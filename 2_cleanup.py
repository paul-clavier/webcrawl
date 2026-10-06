"""Step 2: extract main text and metadata from raw HTML with trafilatura.

Input: data/raw/data.jsonl. Output: data/clean/pages.jsonl (sorted, deduplicated, atomic).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Iterator
from multiprocessing import Pool
from typing import Any

import trafilatura
from tqdm import tqdm

from common import (
    PAGES_FILE,
    RAW_DATA_FILE,
    atomic_write_jsonl,
    count_lines,
    iter_jsonl,
    make_page_id,
    make_text_hash,
    print_summary,
    setup_logging,
)

HTML_LANG_PATTERN = re.compile(r"<html[^>]*?\blang\s*=\s*[\"']?([a-zA-Z]{2,3})", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
BATCH_SIZE_PER_WORKER = 16

logger = setup_logging()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--min-words", type=int, default=50, help="drop pages with fewer words (default 50)"
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count() or 1, help="parallel extraction processes"
    )
    parser.add_argument(
        "--force", action="store_true", help="rebuild even if pages.jsonl is newer than its input"
    )
    return parser.parse_args()


def normalize_text(raw_text: str) -> str:
    """Collapse whitespace inside lines and drop empty lines."""
    normalized_lines = (WHITESPACE_PATTERN.sub(" ", line).strip() for line in raw_text.splitlines())
    return "\n".join(line for line in normalized_lines if line)


def extract_main_content(url_and_html: tuple[str, str]) -> dict[str, Any] | None:
    """Run trafilatura on one page; return its text and metadata, or None if nothing extracted."""
    url, html = url_and_html
    extraction_json = trafilatura.extract(
        html, url=url, output_format="json", with_metadata=True, favor_precision=True
    )
    if not extraction_json:
        return None
    extraction = json.loads(extraction_json)
    page_text = normalize_text(extraction.get("text") or "")
    if not page_text:
        return None
    html_lang_match = HTML_LANG_PATTERN.search(html[:5000])
    return {
        "title": extraction.get("title") or None,
        "author": extraction.get("author") or None,
        "published_date": extraction.get("date") or None,
        "language": extraction.get("language")
        or (html_lang_match.group(1).lower() if html_lang_match else None),
        "description": extraction.get("excerpt") or extraction.get("description") or None,
        "text": page_text,
    }


def iter_batches(batch_size: int) -> Iterator[list[dict[str, Any]]]:
    """Stream raw records in fixed-size batches (never the whole file in memory)."""
    batch: list[dict[str, Any]] = []
    for raw_record in iter_jsonl(RAW_DATA_FILE):
        batch.append(raw_record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_clean_pages(
    arguments: argparse.Namespace, counters: dict[str, int]
) -> Iterator[dict[str, Any]]:
    """Yield clean pages in input order (crawl, url, timestamp), keeping the earliest duplicate."""
    seen_page_ids: set[str] = set()
    batch_size = max(arguments.workers, 1) * BATCH_SIZE_PER_WORKER
    with (
        Pool(processes=arguments.workers) as worker_pool,
        tqdm(total=count_lines(RAW_DATA_FILE), desc="cleanup", unit="page") as progress,
    ):
        for raw_batch in iter_batches(batch_size):
            extractions = worker_pool.map(
                extract_main_content,
                [(raw_record["url"], raw_record["html"]) for raw_record in raw_batch],
                chunksize=BATCH_SIZE_PER_WORKER // 4 or 1,
            )
            progress.update(len(raw_batch))
            for raw_record, extraction in zip(raw_batch, extractions, strict=True):
                counters["records read"] += 1
                if extraction is None:
                    counters["skipped (no text)"] += 1
                    continue
                word_count = len(extraction["text"].split())
                if word_count < arguments.min_words:
                    counters["skipped (too short)"] += 1
                    continue
                text_hash = make_text_hash(extraction["text"])
                page_id = make_page_id(raw_record["url"], text_hash)
                if page_id in seen_page_ids:
                    counters["skipped (duplicate text)"] += 1
                    continue
                seen_page_ids.add(page_id)
                yield {
                    "page_id": page_id,
                    "url": raw_record["url"],
                    "crawl_id": raw_record["crawl_id"],
                    "fetch_timestamp": raw_record["fetch_timestamp"],
                    "title": extraction["title"],
                    "author": extraction["author"],
                    "published_date": extraction["published_date"],
                    "language": extraction["language"],
                    "description": extraction["description"],
                    "text": extraction["text"],
                    "text_hash": text_hash,
                    "word_count": word_count,
                }


def main() -> None:
    """Rebuild pages.jsonl from data.jsonl unless it is already up to date."""
    arguments = parse_arguments()
    if not RAW_DATA_FILE.exists():
        raise SystemExit(f"{RAW_DATA_FILE} not found: run 1_download.py first")
    if (
        not arguments.force
        and PAGES_FILE.exists()
        and PAGES_FILE.stat().st_mtime > RAW_DATA_FILE.stat().st_mtime
    ):
        print_summary(
            "2_cleanup",
            {"status": "up to date (use --force to rebuild)", "pages": count_lines(PAGES_FILE)},
        )
        return

    counter_names = [
        "records read",
        "skipped (no text)",
        "skipped (too short)",
        "skipped (duplicate text)",
    ]
    counters: dict[str, int] = dict.fromkeys(counter_names, 0)
    # Input is sorted by (crawl_id, url, fetch_timestamp) and order is preserved: output is sorted.
    pages_written = atomic_write_jsonl(PAGES_FILE, iter_clean_pages(arguments, counters))
    counters["pages written"] = pages_written
    print_summary("2_cleanup", counters)


if __name__ == "__main__":
    main()
