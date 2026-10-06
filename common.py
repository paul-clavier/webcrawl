"""Shared helpers for the pipeline: paths, config, IDs, files, HTTP, chunking, RRF."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SHARDS_DIR = RAW_DIR / "shards"
CDX_CACHE_DIR = RAW_DIR / "cdx"
RAW_DATA_FILE = RAW_DIR / "data.jsonl"
CLEAN_DIR = DATA_DIR / "clean"
PAGES_FILE = CLEAN_DIR / "pages.jsonl"

load_dotenv(PROJECT_DIR / ".env")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """Connection and model settings, read from environment variables."""

    contact_email: str
    elasticsearch_url: str
    elasticsearch_index: str
    postgres_dsn: str
    embedding_model: str
    embedding_dimensions: int

    @property
    def user_agent(self) -> str:
        """User-Agent header identifying this project to Common Crawl."""
        return f"eurosport-search/0.1 (educational project; contact: {self.contact_email})"


def load_settings() -> Settings:
    """Build Settings from the environment, with local defaults."""
    return Settings(
        contact_email=os.getenv("CONTACT_EMAIL", "you@example.com"),
        elasticsearch_url=os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"),
        elasticsearch_index=os.getenv("ELASTICSEARCH_INDEX", "eurosport_pages"),
        postgres_dsn=os.getenv(
            "POSTGRES_DSN", "postgresql://eurosport:eurosport@localhost:5432/eurosport"
        ),
        embedding_model=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
        embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "384")),
    )


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure root logging and return the pipeline logger."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy_logger_name in ("urllib3", "elastic_transport", "httpx"):
        logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)
    logging.getLogger("trafilatura").setLevel(logging.ERROR)  # "discarding data" is routine
    return logging.getLogger("eurosport")


def print_summary(title: str, counters: dict[str, int | str]) -> None:
    """Print an aligned end-of-run summary block."""
    print(f"\n== {title} ==")
    label_width = max(len(label) for label in counters)
    for label, value in counters.items():
        print(f"  {label:<{label_width}}  {value}")


def detect_device() -> str:
    """Pick the fastest available torch device: cuda, then mps, else cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Deterministic IDs
# ---------------------------------------------------------------------------


def sha1_hex(*parts: str) -> str:
    """SHA-1 hex digest of the concatenated string parts."""
    return hashlib.sha1("".join(parts).encode("utf-8")).hexdigest()


def make_record_id(crawl_id: str, url: str, fetch_timestamp: str) -> str:
    """ID of a raw capture: sha1(crawl_id + url + fetch_timestamp)."""
    return sha1_hex(crawl_id, url, fetch_timestamp)


def make_text_hash(page_text: str) -> str:
    """Hash of a page's main text: sha1(text)."""
    return sha1_hex(page_text)


def make_page_id(url: str, text_hash: str) -> str:
    """ID of a clean page: sha1(url + text_hash)."""
    return sha1_hex(url, text_hash)


def make_chunk_id(page_id: str, chunk_index: int) -> str:
    """ID of a chunk: sha1(page_id + chunk_index)."""
    return sha1_hex(page_id, str(chunk_index))


# ---------------------------------------------------------------------------
# JSONL files: atomic writes, streaming reads, ordering
# ---------------------------------------------------------------------------


def to_json_line(record: dict[str, Any]) -> str:
    """Serialize a record as one JSON line, keeping the dict's key order."""
    return json.dumps(record, ensure_ascii=False, separators=(", ", ": ")) + "\n"


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream records from a JSONL file, one dict per line."""
    with path.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            if line.strip():
                yield json.loads(line)


def count_lines(path: Path) -> int:
    """Count non-empty lines in a file without parsing them."""
    if not path.exists():
        return 0
    with path.open("rb") as binary_file:
        return sum(1 for line in binary_file if line.strip())


def atomic_write_lines(path: Path, lines: Iterable[str], temporary_path: Path | None = None) -> int:
    """Write lines to <path>.tmp then os.replace() it into place; return the line count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = temporary_path or path.with_name(path.name + ".tmp")
    line_count = 0
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for line in lines:
            output_file.write(line)
            line_count += 1
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, path)
    return line_count


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Atomically write records as JSONL; return the record count."""
    return atomic_write_lines(path, (to_json_line(record) for record in records))


def repair_partial_jsonl(path: Path) -> None:
    """Truncate a trailing incomplete line left by a crash during an append."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as binary_file:
        binary_file.seek(-1, os.SEEK_END)
        if binary_file.read(1) == b"\n":
            return
        file_size = binary_file.seek(0, os.SEEK_END)
        read_position = file_size
        while read_position > 0:
            block_size = min(65536, read_position)
            read_position -= block_size
            binary_file.seek(read_position)
            block = binary_file.read(block_size)
            newline_index = block.rfind(b"\n")
            if newline_index != -1:
                binary_file.truncate(read_position + newline_index + 1)
                return
        binary_file.truncate(0)


def record_sort_key(record: dict[str, Any]) -> tuple[str, str, str]:
    """Canonical output order: crawl_id, then url, then fetch_timestamp."""
    return (record["crawl_id"], record["url"], record["fetch_timestamp"])


# Raw records are serialized with "html" as the last key, so everything before
# it can be decoded without parsing the (large) HTML string. Inside a JSON
# string a quote is always escaped, so the first unescaped ', "html": ' is the key.
HTML_KEY_MARKER = ', "html": '


def parse_record_header(line: str) -> dict[str, Any]:
    """Decode all fields of a raw-record JSON line except the trailing 'html'."""
    marker_index = line.find(HTML_KEY_MARKER)
    if marker_index == -1:
        return json.loads(line)
    return json.loads(line[:marker_index] + "}")


def iter_record_headers_with_offsets(path: Path) -> Iterator[tuple[dict[str, Any], int, int]]:
    """Yield (header, byte_offset, byte_length) for every line of a raw JSONL file."""
    with path.open("rb") as binary_file:
        byte_offset = 0
        for raw_line in binary_file:
            if raw_line.strip():
                yield parse_record_header(raw_line.decode("utf-8")), byte_offset, len(raw_line)
            byte_offset += len(raw_line)


def normalize_date(date_text: str | None) -> str | None:
    """Return a valid ISO date (YYYY-MM-DD) or None for missing/invalid input."""
    if not date_text:
        return None
    try:
        return date.fromisoformat(date_text[:10]).isoformat()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Common Crawl helpers
# ---------------------------------------------------------------------------

HTML_MIME_TYPES = frozenset({"text/html", "application/xhtml+xml"})
CRAWL_ID_PATTERN = re.compile(r"CC-MAIN-\d{4}-\d{2}")


def is_indexed_warc_crawl(crawl_id: str) -> bool:
    """True for WARC-era crawls (CC-MAIN-YYYY-WW); false for the 2008–2012 ARC archives."""
    return CRAWL_ID_PATTERN.fullmatch(crawl_id) is not None


def is_successful_html_capture(cdx_row: dict[str, Any]) -> bool:
    """Keep only HTTP 200 captures with an HTML MIME type (mime-detected preferred)."""
    if str(cdx_row.get("status", "")) != "200":
        return False
    mime_type = cdx_row.get("mime-detected") or cdx_row.get("mime") or ""
    return mime_type.split(";")[0].strip().lower() in HTML_MIME_TYPES


def cdx_timestamp_to_iso(cdx_timestamp: str) -> str:
    """Convert a CDX timestamp (YYYYMMDDhhmmss) to ISO 8601 UTC."""
    padded_timestamp = cdx_timestamp.ljust(14, "0")
    return (
        f"{padded_timestamp[0:4]}-{padded_timestamp[4:6]}-{padded_timestamp[6:8]}"
        f"T{padded_timestamp[8:10]}:{padded_timestamp[10:12]}:{padded_timestamp[12:14]}Z"
    )


# ---------------------------------------------------------------------------
# HTTP with retries and rate limiting
# ---------------------------------------------------------------------------

RETRYABLE_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})


class RetriesExhaustedError(RuntimeError):
    """Raised when an HTTP request still fails after all retry attempts."""


class RateLimiter:
    """Thread-safe limiter enforcing a minimum interval between calls."""

    def __init__(self, min_interval_seconds: float) -> None:
        self.min_interval_seconds = min_interval_seconds
        self._lock = threading.Lock()
        self._next_allowed_time = 0.0

    def wait(self) -> None:
        """Block until the next call is allowed."""
        with self._lock:
            now = time.monotonic()
            sleep_seconds = self._next_allowed_time - now
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            self._next_allowed_time = max(now, self._next_allowed_time) + self.min_interval_seconds


def make_http_session(user_agent: str) -> requests.Session:
    """Create a requests session carrying the project User-Agent."""
    session = requests.Session()
    session.headers["User-Agent"] = user_agent
    return session


def get_with_retries(
    session: requests.Session,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    rate_limiter: RateLimiter | None = None,
    accepted_statuses: frozenset[int] = frozenset({200}),
    max_attempts: int = 6,
    timeout_seconds: float = 60.0,
    response_validator: Callable[[requests.Response], str | None] | None = None,
) -> requests.Response:
    """GET with exponential backoff on retryable statuses, network errors, or invalid bodies."""
    last_error: str = ""
    for attempt_number in range(1, max_attempts + 1):
        if rate_limiter is not None:
            rate_limiter.wait()
        try:
            response = session.get(url, headers=headers, params=params, timeout=timeout_seconds)
        except (requests.ConnectionError, requests.Timeout) as network_error:
            last_error = f"network error: {network_error}"
        else:
            if response.status_code in accepted_statuses:
                validation_error = (
                    response_validator(response) if response_validator is not None else None
                )
                if validation_error is None:
                    return response
                last_error = f"invalid response: {validation_error}"
                response.close()
            elif response.status_code not in RETRYABLE_HTTP_STATUSES:
                response.raise_for_status()
                return response
            else:
                last_error = f"HTTP {response.status_code}"
                retry_after_header = response.headers.get("Retry-After", "")
                if retry_after_header.isdigit():
                    time.sleep(min(int(retry_after_header), 120))
        if attempt_number < max_attempts:
            backoff_seconds = min(2**attempt_number, 60) + random.uniform(0, 1)
            logging.getLogger("eurosport").debug(
                "retry %d/%d for %s in %.1fs (%s)",
                attempt_number,
                max_attempts,
                url,
                backoff_seconds,
                last_error,
            )
            time.sleep(backoff_seconds)
    raise RetriesExhaustedError(f"{url}: giving up after {max_attempts} attempts ({last_error})")


# ---------------------------------------------------------------------------
# Chunking and fusion (pure functions, unit-tested)
# ---------------------------------------------------------------------------


def chunk_words(
    page_text: str, title: str | None, chunk_size_words: int = 200, overlap_words: int = 40
) -> list[str]:
    """Split text into overlapping word windows, each prefixed with the title."""
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")
    words = page_text.split()
    step_words = chunk_size_words - overlap_words
    title_prefix = f"{title.strip()}\n\n" if title and title.strip() else ""
    chunk_texts: list[str] = []
    start_index = 0
    while True:
        window = words[start_index : start_index + chunk_size_words]
        chunk_texts.append(title_prefix + " ".join(window))
        if start_index + chunk_size_words >= len(words):
            break
        start_index += step_words
    return chunk_texts


RRF_RANK_CONSTANT = 60


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[str]], rank_constant: int = RRF_RANK_CONSTANT
) -> list[tuple[str, float, list[str]]]:
    """Fuse ranked ID lists: score = Σ 1/(k + rank). Returns (id, score, sources), best first."""
    scores: dict[str, float] = {}
    sources: dict[str, list[str]] = {}
    for source_name, ranked_ids in ranked_lists.items():
        for rank, item_id in enumerate(ranked_ids, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rank_constant + rank)
            sources.setdefault(item_id, []).append(source_name)
    return sorted(
        ((item_id, score, sources[item_id]) for item_id, score in scores.items()),
        key=lambda fused_item: (-fused_item[1], fused_item[0]),
    )
