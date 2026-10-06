"""Step 5: hybrid search: a natural-language prompt in, matching Eurosport pages out.

Combines Elasticsearch (BM25 keyword) and pgvector (semantic) results with Reciprocal Rank Fusion.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

import psycopg
from elasticsearch import Elasticsearch
from pgvector.psycopg import register_vector

from common import (
    Settings,
    detect_device,
    load_settings,
    reciprocal_rank_fusion,
    setup_logging,
)

QUERY_PREFIX = "query: "
SNIPPET_LENGTH = 200
CHUNKS_PER_CANDIDATE_PAGE = 5
WHITESPACE_PATTERN = re.compile(r"\s+")

logger = setup_logging()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prompt", nargs="+", help="what to search for, in natural language")
    parser.add_argument("--mode", choices=["hybrid", "keyword", "semantic"], default="hybrid")
    parser.add_argument(
        "--candidates", type=int, default=100, help="max pages taken from each leg (default 100)"
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.85,
        help="cosine similarity a semantic hit needs to count as a match (default 0.85)",
    )
    parser.add_argument(
        "--top-k", type=int, default=20, help="max pages in the final list (0 = no cap)"
    )
    parser.add_argument("--format", choices=["text", "jsonl"], default="text")
    parser.add_argument("--index", default=settings.elasticsearch_index, help="index name")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    return parser.parse_args()


def make_snippet(source_text: str | None) -> str:
    """Single-line excerpt of at most SNIPPET_LENGTH characters."""
    flat_text = WHITESPACE_PATTERN.sub(" ", source_text or "").strip()
    if len(flat_text) <= SNIPPET_LENGTH:
        return flat_text
    return flat_text[: SNIPPET_LENGTH - 1].rsplit(" ", 1)[0] + "…"


def keyword_search(
    settings: Settings, index_name: str, prompt: str, candidate_count: int
) -> list[dict[str, Any]]:
    """BM25 multi_match over title^2 and text; every Elasticsearch hit counts as a match."""
    client = Elasticsearch(settings.elasticsearch_url, request_timeout=60)
    response = client.search(
        index=index_name,
        query={"multi_match": {"query": prompt, "fields": ["title^2", "text"]}},
        size=candidate_count,
        source_excludes=["text"],
        highlight={
            "fields": {"text": {"fragment_size": SNIPPET_LENGTH, "number_of_fragments": 1}},
            "pre_tags": [""],
            "post_tags": [""],
        },
    )
    keyword_hits = []
    for hit in response["hits"]["hits"]:
        document = hit["_source"]
        highlight_fragments = hit.get("highlight", {}).get("text", [])
        keyword_hits.append(
            {
                "page_id": document["page_id"],
                "url": document["url"],
                "title": document.get("title"),
                "published_date": document.get("published_date"),
                "crawl_id": document["crawl_id"],
                "keyword_score": hit["_score"],
                "snippet": make_snippet(
                    highlight_fragments[0] if highlight_fragments else document.get("description")
                ),
            }
        )
    return keyword_hits


def embed_query(settings: Settings, prompt: str, device: str | None) -> Any:
    """Embed the prompt with the e5 'query: ' prefix, normalized."""
    from sentence_transformers import SentenceTransformer

    device = device or detect_device()
    embedding_model = SentenceTransformer(settings.embedding_model, device=device)
    return embedding_model.encode([QUERY_PREFIX + prompt], normalize_embeddings=True)[0]


def semantic_search(
    settings: Settings,
    prompt: str,
    candidate_count: int,
    min_similarity: float,
    device: str | None,
) -> list[dict[str, Any]]:
    """Nearest chunks by cosine distance, collapsed to pages (best chunk), above min_similarity."""
    query_embedding = embed_query(settings, prompt, device)
    chunk_limit = candidate_count * CHUNKS_PER_CANDIDATE_PAGE
    with psycopg.connect(settings.postgres_dsn) as connection:
        register_vector(connection)
        with connection.cursor() as cursor:
            cursor.execute(f"SET hnsw.ef_search = {min(max(chunk_limit, 40), 1000)}")
            cursor.execute(
                """
                WITH nearest_chunks AS (
                    SELECT page_id, chunk_text, embedding <=> %(query)s AS cosine_distance
                    FROM chunks
                    ORDER BY embedding <=> %(query)s
                    LIMIT %(chunk_limit)s
                )
                SELECT nc.page_id, nc.chunk_text, 1 - nc.cosine_distance AS similarity,
                       p.url, p.title, p.published_date, p.crawl_id
                FROM nearest_chunks nc JOIN pages p USING (page_id)
                ORDER BY nc.cosine_distance, nc.page_id
                """,
                {"query": query_embedding, "chunk_limit": chunk_limit},
            )
            chunk_rows = cursor.fetchall()

    semantic_hits: list[dict[str, Any]] = []
    seen_page_ids: set[str] = set()
    for page_id, chunk_text, similarity, url, title, published_date, crawl_id in chunk_rows:
        if page_id in seen_page_ids or similarity < min_similarity:
            continue
        seen_page_ids.add(page_id)
        has_title_prefix = bool(title) and "\n\n" in chunk_text
        chunk_body = chunk_text.split("\n\n", 1)[1] if has_title_prefix else chunk_text
        semantic_hits.append(
            {
                "page_id": page_id,
                "url": url,
                "title": title,
                "published_date": published_date.isoformat() if published_date else None,
                "crawl_id": crawl_id,
                "similarity": round(float(similarity), 4),
                "snippet": make_snippet(chunk_body),
            }
        )
        if len(semantic_hits) == candidate_count:
            break
    return semantic_hits


def fuse_results(
    keyword_hits: list[dict[str, Any]], semantic_hits: list[dict[str, Any]], top_k: int
) -> list[dict[str, Any]]:
    """Merge both legs with RRF at page level and label which leg(s) found each page."""
    hits_by_source = {"keyword": keyword_hits, "semantic": semantic_hits}
    ranked_page_ids = {
        source_name: [hit["page_id"] for hit in hits]
        for source_name, hits in hits_by_source.items()
        if hits
    }
    rank_by_source = {
        source_name: {page_id: rank for rank, page_id in enumerate(page_ids, start=1)}
        for source_name, page_ids in ranked_page_ids.items()
    }
    hit_by_source = {
        source_name: {hit["page_id"]: hit for hit in hits}
        for source_name, hits in hits_by_source.items()
    }
    fused_pages = []
    for final_rank, (page_id, rrf_score, sources) in enumerate(
        reciprocal_rank_fusion(ranked_page_ids), start=1
    ):
        keyword_hit = hit_by_source["keyword"].get(page_id)
        semantic_hit = hit_by_source["semantic"].get(page_id)
        primary_hit = keyword_hit or semantic_hit
        fused_pages.append(
            {
                "rank": final_rank,
                "page_id": page_id,
                "rrf_score": round(rrf_score, 6),
                "found_by": "both" if len(sources) == 2 else sources[0],
                "keyword_rank": rank_by_source.get("keyword", {}).get(page_id),
                "semantic_rank": rank_by_source.get("semantic", {}).get(page_id),
                "similarity": semantic_hit["similarity"] if semantic_hit else None,
                "title": primary_hit["title"],
                "published_date": primary_hit["published_date"],
                "url": primary_hit["url"],
                "crawl_id": primary_hit["crawl_id"],
                "snippet": (semantic_hit or primary_hit)["snippet"] or primary_hit["snippet"],
            }
        )
        if top_k and final_rank == top_k:
            break
    return fused_pages


def print_text_results(fused_pages: list[dict[str, Any]], match_counts: dict[str, int]) -> None:
    """Human-readable result list."""
    counts_label = ", ".join(f"{name}: {count}" for name, count in match_counts.items())
    print(f"{len(fused_pages)} pages shown ({counts_label} matched)\n")
    for page in fused_pages:
        found_by_details = []
        if page["keyword_rank"]:
            found_by_details.append(f"kw #{page['keyword_rank']}")
        if page["semantic_rank"]:
            found_by_details.append(f"sem #{page['semantic_rank']}, sim {page['similarity']:.2f}")
        print(f"{page['rank']:>3}. {page['title'] or '(untitled)'}")
        print(
            f"     {page['published_date'] or '----------'} · {page['url']}\n"
            f"     found by {page['found_by']} ({'; '.join(found_by_details)})"
        )
        if page["snippet"]:
            print(f"     {page['snippet']}")
        print()


def main() -> None:
    """Run the requested search legs, fuse them, and print the results."""
    arguments = parse_arguments()
    settings = load_settings()
    prompt = " ".join(arguments.prompt)

    keyword_hits = (
        keyword_search(settings, arguments.index, prompt, arguments.candidates)
        if arguments.mode in ("hybrid", "keyword")
        else []
    )
    semantic_hits = (
        semantic_search(
            settings, prompt, arguments.candidates, arguments.min_similarity, arguments.device
        )
        if arguments.mode in ("hybrid", "semantic")
        else []
    )
    fused_pages = fuse_results(keyword_hits, semantic_hits, arguments.top_k)

    if arguments.format == "jsonl":
        for page in fused_pages:
            sys.stdout.write(json.dumps(page, ensure_ascii=False) + "\n")
        return
    match_counts = {}
    if arguments.mode != "semantic":
        match_counts["keyword"] = len(keyword_hits)
    if arguments.mode != "keyword":
        match_counts["semantic"] = len(semantic_hits)
    print_text_results(fused_pages, match_counts)


if __name__ == "__main__":
    main()
