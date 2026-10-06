"""Step 4: chunk pages, embed chunks, and store them in Postgres with pgvector.

Input: data/clean/pages.jsonl. Output: tables `pages` and `chunks`, mirroring the file exactly.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from tqdm import tqdm

from common import (
    PAGES_FILE,
    Settings,
    chunk_words,
    count_lines,
    detect_device,
    iter_jsonl,
    load_settings,
    make_chunk_id,
    make_text_hash,
    normalize_date,
    print_summary,
    setup_logging,
)

PAGES_PER_TRANSACTION = 256
PASSAGE_PREFIX = "passage: "

logger = setup_logging()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch-size", type=int, default=64, help="chunks per embedding batch")
    parser.add_argument(
        "--device", choices=["cpu", "cuda", "mps"], default=None, help="torch device (auto)"
    )
    parser.add_argument(
        "--recreate", action="store_true", help="drop and rebuild tables (after a model change)"
    )
    return parser.parse_args()


class LazyEmbeddingModel:
    """Loads the SentenceTransformer on first use, so up-to-date reruns skip the load."""

    def __init__(self, model_name: str, device: str | None) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None

    def encode(self, texts: list[str], **encode_options: Any) -> Any:
        """Embed texts, loading the model if needed."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            device = self.device or detect_device()
            logger.info("loading %s on %s", self.model_name, device)
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model.encode(texts, **encode_options)


def create_schema(connection: psycopg.Connection, settings: Settings, recreate: bool) -> None:
    """Create extension, tables and HNSW index if missing; refuse a silent model change."""
    with connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        if recreate:
            cursor.execute("DROP TABLE IF EXISTS chunks, pages, pipeline_metadata CASCADE")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS pages (
                page_id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                title TEXT,
                published_date DATE,
                crawl_id TEXT NOT NULL,
                fetch_timestamp TIMESTAMPTZ NOT NULL
            )
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                page_id TEXT NOT NULL REFERENCES pages ON DELETE CASCADE,
                chunk_index INT NOT NULL,
                chunk_text TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                embedding vector({settings.embedding_dimensions}) NOT NULL
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS chunks_page_id_idx ON chunks (page_id)")
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
            ON chunks USING hnsw (embedding vector_cosine_ops)
            """
        )
        cursor.execute(
            """
            INSERT INTO pipeline_metadata (key, value) VALUES ('embedding_model', %s)
            ON CONFLICT (key) DO NOTHING
            """,
            (settings.embedding_model,),
        )
        cursor.execute("SELECT value FROM pipeline_metadata WHERE key = 'embedding_model'")
        stored_model_name = cursor.fetchone()[0]
    connection.commit()
    if stored_model_name != settings.embedding_model:
        raise SystemExit(
            f"tables were built with {stored_model_name}, config says {settings.embedding_model}:"
            " rerun with --recreate"
        )


def iter_page_batches(batch_size: int) -> Iterator[list[dict[str, Any]]]:
    """Stream pages from pages.jsonl in fixed-size batches."""
    batch: list[dict[str, Any]] = []
    for page in iter_jsonl(PAGES_FILE):
        batch.append(page)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_chunks(page: dict[str, Any]) -> list[dict[str, Any]]:
    """Chunk one page into rows (without embeddings)."""
    return [
        {
            "chunk_id": make_chunk_id(page["page_id"], chunk_index),
            "page_id": page["page_id"],
            "chunk_index": chunk_index,
            "chunk_text": chunk_text,
            "text_hash": make_text_hash(chunk_text),
        }
        for chunk_index, chunk_text in enumerate(chunk_words(page["text"], page["title"]))
    ]


def store_page_batch(
    connection: psycopg.Connection,
    embedding_model: Any,
    pages: list[dict[str, Any]],
    embedding_batch_size: int,
    counters: dict[str, int],
) -> None:
    """Upsert pages and their chunks in one transaction, embedding only new/changed chunks."""
    page_chunks = [chunk for page in pages for chunk in build_chunks(page)]
    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO pages (page_id, url, title, published_date, crawl_id, fetch_timestamp)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (page_id) DO UPDATE SET
                url = EXCLUDED.url, title = EXCLUDED.title,
                published_date = EXCLUDED.published_date, crawl_id = EXCLUDED.crawl_id,
                fetch_timestamp = EXCLUDED.fetch_timestamp
            """,
            [
                (
                    page["page_id"],
                    page["url"],
                    page["title"],
                    normalize_date(page["published_date"]),
                    page["crawl_id"],
                    page["fetch_timestamp"],
                )
                for page in pages
            ],
        )
        cursor.execute(
            "SELECT chunk_id, text_hash FROM chunks WHERE page_id = ANY(%s)",
            ([page["page_id"] for page in pages],),
        )
        stored_hash_by_chunk_id = dict(cursor.fetchall())
        chunks_to_embed = [
            chunk
            for chunk in page_chunks
            if stored_hash_by_chunk_id.get(chunk["chunk_id"]) != chunk["text_hash"]
        ]
        current_chunk_ids = {chunk["chunk_id"] for chunk in page_chunks}
        obsolete_chunk_ids = [
            chunk_id for chunk_id in stored_hash_by_chunk_id if chunk_id not in current_chunk_ids
        ]
        if obsolete_chunk_ids:
            cursor.execute("DELETE FROM chunks WHERE chunk_id = ANY(%s)", (obsolete_chunk_ids,))

        if chunks_to_embed:
            embeddings = embedding_model.encode(
                [PASSAGE_PREFIX + chunk["chunk_text"] for chunk in chunks_to_embed],
                batch_size=embedding_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            cursor.executemany(
                """
                INSERT INTO chunks
                    (chunk_id, page_id, chunk_index, chunk_text, text_hash, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    page_id = EXCLUDED.page_id, chunk_index = EXCLUDED.chunk_index,
                    chunk_text = EXCLUDED.chunk_text, text_hash = EXCLUDED.text_hash,
                    embedding = EXCLUDED.embedding
                """,
                [
                    (
                        chunk["chunk_id"],
                        chunk["page_id"],
                        chunk["chunk_index"],
                        chunk["chunk_text"],
                        chunk["text_hash"],
                        embedding,
                    )
                    for chunk, embedding in zip(chunks_to_embed, embeddings, strict=True)
                ],
            )
    connection.commit()
    counters["pages read"] += len(pages)
    counters["chunks total"] += len(page_chunks)
    counters["chunks embedded"] += len(chunks_to_embed)
    counters["chunks unchanged (skipped)"] += len(page_chunks) - len(chunks_to_embed)
    counters["obsolete chunks deleted"] += len(obsolete_chunk_ids)


def delete_stale_pages(connection: psycopg.Connection, file_page_ids: set[str]) -> int:
    """Delete pages (and, by cascade, chunks) absent from pages.jsonl; return how many."""
    with connection.cursor() as cursor:
        cursor.execute("SELECT page_id FROM pages")
        stale_page_ids = [row[0] for row in cursor.fetchall() if row[0] not in file_page_ids]
        if stale_page_ids:
            cursor.execute("DELETE FROM pages WHERE page_id = ANY(%s)", (stale_page_ids,))
    connection.commit()
    return len(stale_page_ids)


def main() -> None:
    """Sync the pages/chunks tables with pages.jsonl."""
    arguments = parse_arguments()
    if not PAGES_FILE.exists():
        raise SystemExit(f"{PAGES_FILE} not found: run 2_cleanup.py first")
    settings = load_settings()

    with psycopg.connect(settings.postgres_dsn) as connection:
        create_schema(connection, settings, arguments.recreate)
        register_vector(connection)

        embedding_model = LazyEmbeddingModel(settings.embedding_model, arguments.device)

        counter_names = [
            "pages read",
            "chunks total",
            "chunks embedded",
            "chunks unchanged (skipped)",
            "obsolete chunks deleted",
        ]
        counters: dict[str, int] = dict.fromkeys(counter_names, 0)
        file_page_ids: set[str] = set()
        with tqdm(total=count_lines(PAGES_FILE), desc="store", unit="page") as progress:
            for page_batch in iter_page_batches(PAGES_PER_TRANSACTION):
                file_page_ids.update(page["page_id"] for page in page_batch)
                store_page_batch(
                    connection, embedding_model, page_batch, arguments.batch_size, counters
                )
                progress.update(len(page_batch))

        counters["stale pages deleted"] = delete_stale_pages(connection, file_page_ids)
        with connection.cursor() as cursor:
            cursor.execute("SELECT (SELECT count(*) FROM pages), (SELECT count(*) FROM chunks)")
            page_row_count, chunk_row_count = cursor.fetchone()
    counters["rows in pages"] = page_row_count
    counters["rows in chunks"] = chunk_row_count
    print_summary("4_store", counters)


if __name__ == "__main__":
    main()
