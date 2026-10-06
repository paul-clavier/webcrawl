"""Step 3: index clean pages in Elasticsearch for keyword (BM25) search.

Input: data/clean/pages.jsonl. Output: the Elasticsearch index, mirroring the file exactly.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from typing import Any

from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

from common import (
    PAGES_FILE,
    count_lines,
    iter_jsonl,
    load_settings,
    normalize_date,
    print_summary,
    setup_logging,
)

INDEX_SETTINGS: dict[str, Any] = {"number_of_shards": 1, "number_of_replicas": 0}
INDEX_MAPPINGS: dict[str, Any] = {
    "dynamic": "strict",
    "properties": {
        "page_id": {"type": "keyword"},
        "url": {"type": "keyword"},
        "crawl_id": {"type": "keyword"},
        "fetch_timestamp": {"type": "date", "ignore_malformed": True},
        "title": {"type": "text", "analyzer": "french"},
        "author": {"type": "keyword"},
        "published_date": {"type": "date", "ignore_malformed": True},
        "language": {"type": "keyword"},
        "description": {"type": "text", "analyzer": "french"},
        "text": {"type": "text", "analyzer": "french"},
        "text_hash": {"type": "keyword"},
        "word_count": {"type": "integer"},
    },
}

logger = setup_logging()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--index", default=settings.elasticsearch_index, help="index name (default from .env)"
    )
    parser.add_argument("--batch-size", type=int, default=500, help="documents per bulk request")
    parser.add_argument(
        "--recreate", action="store_true", help="drop and rebuild the index (after mapping changes)"
    )
    return parser.parse_args()


def ensure_index(client: Elasticsearch, index_name: str, recreate: bool) -> bool:
    """Create the index with its explicit mapping if missing; return True if it was created."""
    if recreate and client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
    if client.indices.exists(index=index_name):
        return False
    client.indices.create(index=index_name, settings=INDEX_SETTINGS, mappings=INDEX_MAPPINGS)
    return True


def page_to_document(page: dict[str, Any]) -> dict[str, Any]:
    """Map a clean page to its Elasticsearch document (invalid dates become null)."""
    return {
        "page_id": page["page_id"],
        "url": page["url"],
        "crawl_id": page["crawl_id"],
        "fetch_timestamp": page["fetch_timestamp"],
        "title": page["title"],
        "author": page["author"],
        "published_date": normalize_date(page["published_date"]),
        "language": page["language"],
        "description": page["description"],
        "text": page["text"],
        "text_hash": page["text_hash"],
        "word_count": page["word_count"],
    }


def iter_index_actions(index_name: str, file_page_ids: set[str]) -> Iterator[dict[str, Any]]:
    """Bulk 'index' actions keyed by page_id (an upsert); records each ID seen in the file."""
    for page in iter_jsonl(PAGES_FILE):
        file_page_ids.add(page["page_id"])
        yield {
            "_op_type": "index",
            "_index": index_name,
            "_id": page["page_id"],
            "_source": page_to_document(page),
        }


def delete_stale_documents(client: Elasticsearch, index_name: str, file_page_ids: set[str]) -> int:
    """Delete documents whose page_id is no longer in pages.jsonl; return how many."""
    stale_document_ids = [
        hit["_id"]
        for hit in helpers.scan(
            client, index=index_name, query={"query": {"match_all": {}}}, _source=False
        )
        if hit["_id"] not in file_page_ids
    ]
    if stale_document_ids:
        helpers.bulk(
            client,
            (
                {"_op_type": "delete", "_index": index_name, "_id": document_id}
                for document_id in stale_document_ids
            ),
        )
    return len(stale_document_ids)


def main() -> None:
    """Upsert every page, remove stale ones, and check the index mirrors the file."""
    arguments = parse_arguments()
    if not PAGES_FILE.exists():
        raise SystemExit(f"{PAGES_FILE} not found: run 2_cleanup.py first")
    settings = load_settings()
    client = Elasticsearch(settings.elasticsearch_url, request_timeout=120)
    index_created = ensure_index(client, arguments.index, arguments.recreate)

    file_page_ids: set[str] = set()
    indexed_count = 0
    with tqdm(total=count_lines(PAGES_FILE), desc="index", unit="doc") as progress:
        for is_success, _bulk_item in helpers.streaming_bulk(
            client,
            iter_index_actions(arguments.index, file_page_ids),
            chunk_size=arguments.batch_size,
            raise_on_error=True,
        ):
            indexed_count += int(is_success)
            progress.update(1)
    client.indices.refresh(index=arguments.index)
    deleted_count = delete_stale_documents(client, arguments.index, file_page_ids)
    client.indices.refresh(index=arguments.index)

    document_count = client.count(index=arguments.index)["count"]
    print_summary(
        "3_index",
        {
            "index": arguments.index + (" (created)" if index_created else ""),
            "pages read": len(file_page_ids),
            "documents upserted": indexed_count,
            "stale documents deleted": deleted_count,
            "documents in index": document_count,
        },
    )
    if document_count != len(file_page_ids):
        raise SystemExit(
            f"index has {document_count} documents but pages.jsonl has {len(file_page_ids)}"
        )


if __name__ == "__main__":
    main()
