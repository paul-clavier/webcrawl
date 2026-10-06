"""Step 1: download every successful HTML capture of eurosport.fr from Common Crawl.

Output: data/raw/shards/{crawl_id}.jsonl (one per crawl), merged into data/raw/data.jsonl.
"""

from __future__ import annotations

import argparse
import codecs
import io
import json
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from charset_normalizer import from_bytes
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, TypeAdapter
from tqdm import tqdm
from warcio.archiveiterator import ArchiveIterator

from common import (
    CDX_CACHE_DIR,
    RAW_DATA_FILE,
    RAW_DIR,
    SHARDS_DIR,
    RateLimiter,
    RetriesExhaustedError,
    atomic_write_jsonl,
    atomic_write_lines,
    cdx_timestamp_to_iso,
    get_with_retries,
    is_indexed_warc_crawl,
    is_successful_html_capture,
    iter_jsonl,
    iter_record_headers_with_offsets,
    load_settings,
    make_http_session,
    make_record_id,
    print_summary,
    repair_partial_jsonl,
    setup_logging,
    to_json_line,
)

COLLINFO_URL = "https://index.commoncrawl.org/collinfo.json"
CDX_API_TEMPLATE = "https://index.commoncrawl.org/{crawl_id}-index"
WARC_DATA_URL_TEMPLATE = "https://data.commoncrawl.org/{warc_filename}"
TARGET_DOMAIN = "eurosport.fr"
CHARSET_PATTERN = re.compile(r"charset\s*=\s*[\"']?([\w.:-]+)", re.IGNORECASE)

logger = setup_logging()
settings = load_settings()
cdx_rate_limiter = RateLimiter(min_interval_seconds=1.0)
thread_local_storage = threading.local()


class CrawlInfo(BaseModel):
    """One entry of collinfo.json."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    timegate: HttpUrl
    cdx_api: HttpUrl = Field(alias="cdx-api")
    from_: datetime = Field(alias="from")
    to: datetime


crawl_list_adapter = TypeAdapter(list[CrawlInfo])


class CaptureSkippedError(Exception):
    """A capture that can never be turned into a record (bad WARC payload, not a response)."""


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-crawl", help="first crawl ID to process, e.g. CC-MAIN-2015-11")
    parser.add_argument("--to-crawl", help="last crawl ID to process, e.g. CC-MAIN-2016-07")
    parser.add_argument(
        "--limit", type=int, default=0, help="max captures in total, for smoke tests (0 = all)"
    )
    parser.add_argument("--workers", type=int, default=4, help="parallel WARC fetches")
    parser.add_argument(
        "--keep-all-captures",
        action="store_true",
        help="keep captures whose content digest was already seen (rerun with --force)",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download crawls even if their shard is complete"
    )
    return parser.parse_args()


def shared_session() -> requests.Session:
    """Return this thread's HTTP session (sessions are not shared across threads)."""
    if not hasattr(thread_local_storage, "session"):
        thread_local_storage.session = make_http_session(settings.user_agent)
    return thread_local_storage.session


# ---------------------------------------------------------------------------
# Crawl list and CDX index
# ---------------------------------------------------------------------------


def list_crawl_ids(from_crawl: str | None, to_crawl: str | None) -> list[str]:
    """Indexed WARC crawls from collinfo.json, oldest first, within the requested range."""
    collinfo_cache_path = RAW_DIR / "collinfo.json"
    try:
        response = get_with_retries(shared_session(), COLLINFO_URL, rate_limiter=cdx_rate_limiter)
        crawls = crawl_list_adapter.validate_json(response.content)
        atomic_write_lines(collinfo_cache_path, [response.text])
    except RetriesExhaustedError:
        if not collinfo_cache_path.exists():
            raise
        logger.warning("index server unreachable: using cached %s", collinfo_cache_path.name)
        crawls = crawl_list_adapter.validate_json(collinfo_cache_path.read_bytes())
    crawl_ids = sorted(crawl.id for crawl in crawls if is_indexed_warc_crawl(crawl.id))
    return [
        crawl_id
        for crawl_id in crawl_ids
        if (from_crawl is None or crawl_id >= from_crawl)
        and (to_crawl is None or crawl_id <= to_crawl)
    ]


def is_no_captures_response(response: requests.Response) -> bool:
    """True when the CDX server says the crawl has no capture for the query."""
    return response.status_code == 404 and "No Captures found" in response.text


def validate_cdx_response(response: requests.Response) -> str | None:
    """Return an error for a truncated or otherwise malformed successful CDX response."""
    if response.status_code != 200:
        return None
    has_rows = False
    for line_number, line in enumerate(response.text.splitlines(), start=1):
        if not line.strip():
            continue
        has_rows = True
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            return f"malformed JSON on line {line_number}: {error.msg}"
        if not isinstance(row, dict):
            return f"JSON line {line_number} is not an object"
    return None if has_rows else "empty body"


def fetch_cdx_page(crawl_id: str, extra_params: dict[str, Any]) -> requests.Response:
    """One rate-limited CDX request for eurosport.fr (all subdomains)."""
    query_params = {"url": TARGET_DOMAIN, "matchType": "domain", "output": "json", **extra_params}
    return get_with_retries(
        shared_session(),
        CDX_API_TEMPLATE.format(crawl_id=crawl_id),
        params=query_params,
        rate_limiter=cdx_rate_limiter,
        accepted_statuses=frozenset({200, 404}),
        timeout_seconds=180.0,
        response_validator=validate_cdx_response,
    )


def cdx_row_to_candidate(crawl_id: str, cdx_row: dict[str, Any]) -> dict[str, Any]:
    """Keep the CDX fields needed to fetch and describe a capture."""
    fetch_timestamp = cdx_timestamp_to_iso(cdx_row["timestamp"])
    return {
        "record_id": make_record_id(crawl_id, cdx_row["url"], fetch_timestamp),
        "url": cdx_row["url"],
        "fetch_timestamp": fetch_timestamp,
        "content_digest": cdx_row.get("digest", ""),
        "warc_filename": cdx_row["filename"],
        "warc_record_offset": int(cdx_row["offset"]),
        "warc_record_length": int(cdx_row["length"]),
    }


def list_crawl_candidates(crawl_id: str, use_cache: bool) -> tuple[list[dict[str, Any]], int]:
    """HTML 200 captures of a crawl (cached on disk); returns (candidates, total CDX rows)."""
    cache_path = CDX_CACHE_DIR / f"{crawl_id}.jsonl"
    if use_cache and cache_path.exists():
        cached_rows = list(iter_jsonl(cache_path))
        header_row, candidates = cached_rows[0], cached_rows[1:]
        return candidates, header_row["total_cdx_rows"]

    page_count_response = fetch_cdx_page(crawl_id, {"showNumPages": "true"})
    page_count = (
        0
        if is_no_captures_response(page_count_response)
        else int(page_count_response.json().get("pages", 0))
    )
    candidates: list[dict[str, Any]] = []
    total_cdx_rows = 0
    for page_number in range(page_count):
        page_response = fetch_cdx_page(crawl_id, {"page": page_number})
        if is_no_captures_response(page_response):
            continue
        for line in page_response.text.splitlines():
            if not line.strip():
                continue
            total_cdx_rows += 1
            cdx_row = json.loads(line)
            if is_successful_html_capture(cdx_row):
                candidates.append(cdx_row_to_candidate(crawl_id, cdx_row))
    candidates.sort(key=lambda candidate: (candidate["url"], candidate["fetch_timestamp"]))
    atomic_write_jsonl(cache_path, [{"total_cdx_rows": total_cdx_rows}, *candidates])
    return candidates, total_cdx_rows


def select_unique_candidates(
    candidates: list[dict[str, Any]], seen_digests: set[str], keep_all_captures: bool
) -> list[dict[str, Any]]:
    """Drop repeated record IDs and, unless keep_all_captures, already-seen content digests."""
    best_by_key: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if keep_all_captures:
            dedup_key = candidate["record_id"]
        else:
            if candidate["content_digest"] in seen_digests:
                continue
            dedup_key = candidate["content_digest"] or candidate["record_id"]
        current_best = best_by_key.get(dedup_key)
        candidate_order = (candidate["fetch_timestamp"], candidate["url"])
        if current_best is None or candidate_order < (
            current_best["fetch_timestamp"],
            current_best["url"],
        ):
            best_by_key[dedup_key] = candidate
    return sorted(
        best_by_key.values(),
        key=lambda candidate: (candidate["url"], candidate["fetch_timestamp"]),
    )


# ---------------------------------------------------------------------------
# WARC fetching
# ---------------------------------------------------------------------------


def decode_html(body_bytes: bytes, content_type_header: str | None) -> str:
    """Decode HTML bytes with the HTTP charset, else a detected one, replacing bad bytes."""
    charset_match = CHARSET_PATTERN.search(content_type_header or "")
    if charset_match:
        try:
            return body_bytes.decode(codecs.lookup(charset_match.group(1)).name, errors="replace")
        except LookupError:
            pass
    detected_match = from_bytes(body_bytes).best()
    if detected_match is not None and detected_match.encoding:
        return body_bytes.decode(detected_match.encoding, errors="replace")
    return body_bytes.decode("utf-8", errors="replace")


def fetch_capture(crawl_id: str, candidate: dict[str, Any]) -> dict[str, Any]:
    """Range-request one WARC record and build the raw record for it."""
    warc_record_offset = candidate["warc_record_offset"]
    warc_record_length = candidate["warc_record_length"]
    warc_last_byte = warc_record_offset + warc_record_length - 1
    response = get_with_retries(
        shared_session(),
        WARC_DATA_URL_TEMPLATE.format(warc_filename=candidate["warc_filename"]),
        headers={"Range": f"bytes={warc_record_offset}-{warc_last_byte}"},
        accepted_statuses=frozenset({206}),
    )
    try:
        for warc_record in ArchiveIterator(io.BytesIO(response.content)):
            if warc_record.rec_type != "response":
                continue
            http_headers = warc_record.http_headers
            content_type_header = http_headers.get_header("Content-Type") if http_headers else None
            body_bytes = warc_record.content_stream().read()
            break
        else:
            raise CaptureSkippedError("no WARC response record in range")
    except CaptureSkippedError:
        raise
    except Exception as parse_error:
        raise CaptureSkippedError(f"unreadable WARC record: {parse_error}") from parse_error

    return {
        "record_id": candidate["record_id"],
        "crawl_id": crawl_id,
        "url": candidate["url"],
        "fetch_timestamp": candidate["fetch_timestamp"],
        "content_digest": candidate["content_digest"],
        "warc_filename": candidate["warc_filename"],
        "warc_record_offset": warc_record_offset,
        "warc_record_length": warc_record_length,
        "http_status": 200,
        "html": decode_html(body_bytes, content_type_header),  # must stay the last key
    }


# ---------------------------------------------------------------------------
# Shards
# ---------------------------------------------------------------------------


def finished_shard_path(crawl_id: str) -> Path:
    """Path of a crawl's completed shard."""
    return SHARDS_DIR / f"{crawl_id}.jsonl"


def partial_shard_path(crawl_id: str) -> Path:
    """Path of a crawl's in-progress shard."""
    return SHARDS_DIR / f"{crawl_id}.jsonl.tmp"


def existing_shard_path(crawl_id: str) -> Path | None:
    """The finished shard if any, else the partial one, else None."""
    for shard_path in (finished_shard_path(crawl_id), partial_shard_path(crawl_id)):
        if shard_path.exists():
            return shard_path
    return None


def read_shard_headers(shard_path: Path) -> list[dict[str, Any]]:
    """Record headers (all fields but html) of a shard."""
    return [header for header, _, _ in iter_record_headers_with_offsets(shard_path)]


def iter_sorted_shard_lines(shard_path: Path) -> list[tuple[dict[str, Any], int, int]]:
    """Shard line positions sorted by (url, fetch_timestamp), first copy of each record only."""
    positions_by_record_id: dict[str, tuple[dict[str, Any], int, int]] = {}
    for header, byte_offset, byte_length in iter_record_headers_with_offsets(shard_path):
        positions_by_record_id.setdefault(header["record_id"], (header, byte_offset, byte_length))
    return sorted(
        positions_by_record_id.values(),
        key=lambda position: (position[0]["url"], position[0]["fetch_timestamp"]),
    )


def read_lines_at(shard_path: Path, positions: list[tuple[dict[str, Any], int, int]]):
    """Yield the raw lines stored at the given byte positions of a file."""
    with shard_path.open("rb") as binary_file:
        for _, byte_offset, byte_length in positions:
            binary_file.seek(byte_offset)
            yield binary_file.read(byte_length).decode("utf-8")


def finalize_shard(crawl_id: str) -> None:
    """Sort the partial shard into the finished shard atomically, then drop the partial."""
    partial_path = partial_shard_path(crawl_id)
    final_path = finished_shard_path(crawl_id)
    if partial_path.exists():
        positions = iter_sorted_shard_lines(partial_path)
        atomic_write_lines(
            final_path,
            read_lines_at(partial_path, positions),
            temporary_path=SHARDS_DIR / f"{crawl_id}.jsonl.sorting.tmp",
        )
        partial_path.unlink()
    else:
        atomic_write_lines(final_path, [])


def download_crawl(
    crawl_id: str,
    seen_digests: set[str],
    remaining_budget: int | None,
    arguments: argparse.Namespace,
    counters: dict[str, int],
) -> int:
    """Download one crawl into its shard; return how many records the shard now holds."""
    if arguments.force:
        for stale_path in (finished_shard_path(crawl_id), partial_shard_path(crawl_id)):
            stale_path.unlink(missing_ok=True)
    elif finished_shard_path(crawl_id).exists():
        shard_record_count = len(read_shard_headers(finished_shard_path(crawl_id)))
        counters["crawls skipped (complete)"] += 1
        counters["records already present"] += shard_record_count
        return shard_record_count

    candidates, total_cdx_rows = list_crawl_candidates(crawl_id, use_cache=not arguments.force)
    unique_candidates = select_unique_candidates(
        candidates, seen_digests, arguments.keep_all_captures
    )
    counters["CDX rows listed"] += total_cdx_rows
    counters["HTML 200 captures"] += len(candidates)
    counters["skipped (duplicate digest)"] += len(candidates) - len(unique_candidates)

    partial_path = partial_shard_path(crawl_id)
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    repair_partial_jsonl(partial_path)
    present_record_ids = (
        {header["record_id"] for header in read_shard_headers(partial_path)}
        if partial_path.exists()
        else set()
    )
    pending_candidates = [
        candidate
        for candidate in unique_candidates
        if candidate["record_id"] not in present_record_ids
    ]
    counters["records already present"] += len(present_record_ids)
    budget_truncated = False
    if remaining_budget is not None:
        new_record_budget = max(remaining_budget - len(present_record_ids), 0)
        budget_truncated = len(pending_candidates) > new_record_budget
        pending_candidates = pending_candidates[:new_record_budget]

    written_count = 0
    failed_count = 0
    with (
        partial_path.open("a", encoding="utf-8", newline="\n") as shard_file,
        ThreadPoolExecutor(max_workers=arguments.workers) as executor,
        tqdm(total=len(pending_candidates), desc=crawl_id, unit="page", leave=False) as progress,
    ):
        future_to_candidate: dict[Future, dict[str, Any]] = {
            executor.submit(fetch_capture, crawl_id, candidate): candidate
            for candidate in pending_candidates
        }
        try:
            for future in as_completed(future_to_candidate):
                candidate = future_to_candidate[future]
                progress.update(1)
                try:
                    raw_record = future.result()
                except CaptureSkippedError as skip_reason:
                    counters["skipped (unreadable WARC)"] += 1
                    logger.debug("skip %s: %s", candidate["url"], skip_reason)
                    continue
                except (RetriesExhaustedError, requests.RequestException) as fetch_error:
                    failed_count += 1
                    logger.warning("failed %s: %s", candidate["url"], fetch_error)
                    continue
                shard_file.write(to_json_line(raw_record))
                shard_file.flush()
                written_count += 1
        except KeyboardInterrupt:
            executor.shutdown(wait=False, cancel_futures=True)
            raise

    counters["downloaded"] += written_count
    counters["failed (will retry on rerun)"] += failed_count
    if failed_count == 0 and not budget_truncated:
        finalize_shard(crawl_id)
        counters["crawls completed"] += 1
    else:
        counters["crawls left incomplete"] += 1
    logger.info(
        "%s: %d HTML captures, %d unique, %d new, %d failed%s",
        crawl_id,
        len(candidates),
        len(unique_candidates),
        written_count,
        failed_count,
        " (stopped by --limit)" if budget_truncated else "",
    )
    return len(present_record_ids) + written_count


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def merge_shards(crawl_ids: list[str], keep_all_captures: bool) -> int:
    """Merge shards into data.jsonl: sorted, atomic, one record per digest (earliest capture)."""
    shard_paths = [path for crawl_id in crawl_ids if (path := existing_shard_path(crawl_id))]
    winning_record_ids: set[str] | None = None
    if not keep_all_captures:
        earliest_by_digest: dict[str, tuple[str, str, str, str]] = {}
        for shard_path in shard_paths:
            for header in read_shard_headers(shard_path):
                dedup_key = header["content_digest"] or header["record_id"]
                capture_order = (
                    header["fetch_timestamp"],
                    header["crawl_id"],
                    header["url"],
                    header["record_id"],
                )
                if (
                    dedup_key not in earliest_by_digest
                    or capture_order < earliest_by_digest[dedup_key]
                ):
                    earliest_by_digest[dedup_key] = capture_order
        winning_record_ids = {capture[3] for capture in earliest_by_digest.values()}

    def iter_merged_lines():
        """Shard lines in crawl, url, timestamp order, filtered to the winning captures."""
        for shard_path in shard_paths:
            positions = [
                position
                for position in iter_sorted_shard_lines(shard_path)
                if winning_record_ids is None or position[0]["record_id"] in winning_record_ids
            ]
            yield from read_lines_at(shard_path, positions)

    return atomic_write_lines(RAW_DATA_FILE, iter_merged_lines())


def main() -> None:
    """Download all requested crawls, then merge their shards."""
    arguments = parse_arguments()
    crawl_ids = list_crawl_ids(arguments.from_crawl, arguments.to_crawl)
    logger.info("%d crawls to process (%s → %s)", len(crawl_ids), crawl_ids[0], crawl_ids[-1])

    counter_names = [
        "crawls completed",
        "crawls skipped (complete)",
        "crawls left incomplete",
        "CDX rows listed",
        "HTML 200 captures",
        "skipped (duplicate digest)",
        "records already present",
        "downloaded",
        "skipped (unreadable WARC)",
        "failed (will retry on rerun)",
    ]
    counters: dict[str, int] = dict.fromkeys(counter_names, 0)
    seen_digests: set[str] = set()
    total_records = 0
    processed_crawl_ids: list[str] = []
    for crawl_id in crawl_ids:
        if arguments.limit and total_records >= arguments.limit:
            break
        remaining_budget = arguments.limit - total_records if arguments.limit else None
        try:
            total_records += download_crawl(
                crawl_id, seen_digests, remaining_budget, arguments, counters
            )
        except RetriesExhaustedError as cdx_error:
            logger.error("%s: CDX listing failed, will retry on rerun (%s)", crawl_id, cdx_error)
            counters["crawls left incomplete"] += 1
        processed_crawl_ids.append(crawl_id)
        if shard_path := existing_shard_path(crawl_id):
            seen_digests.update(
                header["content_digest"] for header in read_shard_headers(shard_path)
            )

    merged_record_count = merge_shards(processed_crawl_ids, arguments.keep_all_captures)
    counters["records written to data.jsonl"] = merged_record_count
    print_summary("1_download", counters)
    if counters["crawls left incomplete"]:
        print("  Some crawls are incomplete: rerun the same command to resume them.")


if __name__ == "__main__":
    main()
