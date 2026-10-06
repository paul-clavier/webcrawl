# Eurosport.fr hybrid search (Common Crawl → Elasticsearch + pgvector)

A local, educational search engine over every Eurosport.fr page available in
[Common Crawl](https://commoncrawl.org), from the earliest indexed crawl (2013) to the latest.
It is a pipeline of five numbered Python scripts. You run each one from the CLI, and each one
is idempotent.

```
1_download.py  →  data/raw/data.jsonl        raw HTML captures from Common Crawl
2_cleanup.py   →  data/clean/pages.jsonl     main text + metadata (trafilatura)
3_index.py     →  Elasticsearch index        keyword search (BM25, French analyzer)
4_store.py     →  Postgres + pgvector        chunk embeddings for semantic search
5_query.py     →  stdout                     hybrid search: prompt in, matching pages out
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you)
- Docker with Compose (Elasticsearch 9 + Postgres 17/pgvector)
- About 2 GB of free RAM for Elasticsearch, plus disk space for the raw HTML (see step 1)

## Quickstart

```bash
uv sync
cp .env.example .env          # then set CONTACT_EMAIL to your own address
docker compose up -d          # wait until both services are "healthy": docker compose ps
```

Smoke test first, with a small download:

```bash
uv run python 1_download.py --limit 200
uv run python 2_cleanup.py
uv run python 3_index.py
uv run python 4_store.py
uv run python 5_query.py "Thibaut Pinot sur le Tour d'Espagne"
```

`--limit` takes captures from the oldest crawls first. To smoke-test a specific period, add a
crawl range, e.g. `--from-crawl CC-MAIN-2019-04 --to-crawl CC-MAIN-2019-04`.

Full run (all crawls, which takes many hours and needs lots of disk space):

```bash
uv run python 1_download.py
uv run python 2_cleanup.py && uv run python 3_index.py && uv run python 4_store.py
```

Every command can be interrupted and rerun. Development checks:

```bash
uv run ruff check . && uv run pytest
```

## Project layout

```
common.py        shared helpers: paths, config, IDs, atomic writes, HTTP retries, chunking, RRF
1_download.py … 5_query.py
tests/           pytest unit tests for the pure functions in common.py
data/            gitignored
  raw/collinfo.json        last crawl list fetched (used as fallback when the index is down)
  raw/cdx/{crawl}.jsonl    cached CDX listing of HTML captures per crawl
  raw/shards/{crawl}.jsonl one file per finished crawl ({crawl}.jsonl.tmp while in progress)
  raw/data.jsonl           merged raw captures
  clean/pages.jsonl        clean pages
```

The script names start with a digit, so other code can't import them as modules. Everything
they share lives in `common.py`.

## Idempotency rules (all scripts)

- **Deterministic IDs**: `record_id = sha1(crawl_id + url + fetch_timestamp)`,
  `page_id = sha1(url + text_hash)`, `chunk_id = sha1(page_id + chunk_index)`. No auto-increment
  IDs and no random UUIDs. Every write to Elasticsearch and Postgres is an upsert on these IDs.
- **Atomic files**: outputs are written to `<file>.tmp`, then `os.replace()`d into place.
- **Deterministic order**: files are sorted by `crawl_id`, `url`, `fetch_timestamp`, so a rerun
  produces byte-identical files.
- **Mirrors**: the Elasticsearch index and the Postgres tables are kept identical to
  `pages.jsonl`. Pages that disappear from the file are deleted from both stores.

---

## 1. `1_download.py`: Common Crawl → raw HTML

**Input:** the Common Crawl index server. **Output:** `data/raw/shards/*.jsonl` and
`data/raw/data.jsonl`.

1. Reads the crawl list from `https://index.commoncrawl.org/collinfo.json`. It keeps WARC
   crawls (`CC-MAIN-YYYY-WW`) and processes them oldest first.
2. For each crawl, it lists every eurosport.fr capture (all subdomains) through the CDX API
   (`url=eurosport.fr&matchType=domain`), page by page, at about 1 request per second. A 404
   "No Captures found" just means the crawl has no Eurosport pages. It keeps `status == 200`
   with an HTML MIME type (`mime-detected` when present, otherwise `mime`). The filtered list
   is cached in `data/raw/cdx/`.
3. It skips captures whose CDX `digest` (a hash of the content) has already been downloaded,
   so a page recaptured unchanged in later crawls is fetched only once.
4. It fetches each capture with an HTTP range request on `data.commoncrawl.org`, parses it
   with `warcio`, and decodes the HTML using the HTTP charset (falling back to
   `charset-normalizer`, then UTF-8, with `errors="replace"`).

| option | meaning |
|---|---|
| `--from-crawl`, `--to-crawl` | crawl ID range, e.g. `CC-MAIN-2015-11` |
| `--limit N` | at most N captures in total (smoke tests). Deterministic: always the same N |
| `--workers` | parallel WARC fetches (default 4) |
| `--keep-all-captures` | don't deduplicate by digest (use with `--force` on already-downloaded crawls) |
| `--force` | re-list and re-download crawls even if their shard is complete |

**Resume:** each crawl is appended to `shards/{crawl}.jsonl.tmp` and renamed to
`{crawl}.jsonl` (sorted) only when every capture of the crawl has been fetched. On rerun:

- complete shards are skipped;
- an unfinished shard is repaired (a line truncated by a crash is dropped) and resumed,
  skipping record IDs it already contains;
- captures that failed after 6 retries are retried, and their crawl stays unfinished until
  they succeed.

At the end, the shards are merged into `data.jsonl`. When the same digest appears more than
once, only the earliest capture is kept.

> **Disk space.** Raw HTML for all crawls can reach tens of GB. Use `--limit` and the crawl range
> options to stay small while experimenting.

Record fields: `record_id`, `crawl_id`, `url`, `fetch_timestamp` (ISO 8601), `content_digest`,
`warc_filename`, `warc_record_offset`, `warc_record_length`, `http_status`, `html`. `html` is
always the last key, so the scripts can read the other fields without parsing the HTML string.

## 2. `2_cleanup.py`: raw HTML → clean text

**Input:** `data/raw/data.jsonl` (streamed). **Output:** `data/clean/pages.jsonl`.

- Runs `trafilatura.extract(..., output_format="json", with_metadata=True, favor_precision=True)`
  in parallel processes, preserving input order.
- Normalizes whitespace, then drops pages with no text or fewer than `--min-words` (default 50).
- Deduplicates on `(url, text_hash)`: the same article captured with different ads or menus has
  a different digest but the same main text. The earliest capture is kept.
- Page fields: `page_id`, `url`, `crawl_id`, `fetch_timestamp`, `title`, `author`,
  `published_date`, `language`, `description`, `text`, `text_hash`, `word_count`. `language` is
  often null because Eurosport pages don't declare one.

Options: `--min-words`, `--workers` (default: CPU count), `--force`.
**Idempotency:** the output depends only on the input file. If `pages.jsonl` is newer than
`data.jsonl`, the script does nothing. Use `--force` after changing `--min-words`.

## 3. `3_index.py`: Elasticsearch keyword index

**Input:** `pages.jsonl`. **Output:** index `eurosport_pages` (set with `ELASTICSEARCH_INDEX` or
`--index`).

- Creates the index only if it's missing, with an explicit `strict` mapping. `title`,
  `description` and `text` use the built-in `french` analyzer, which handles elision
  ("l'équipe" → "équipe"), French stopwords and stemming. `url`, `crawl_id`, `language` and
  `author` are `keyword` fields, the dates are `date` fields (invalid dates are stored as
  null), and `word_count` is an `integer`.
- Bulk-indexes with `_id = page_id` (op type `index`, which is an upsert), then deletes
  documents whose `page_id` is no longer in the file.
- Prints the final document count and exits with an error if it differs from the number of
  lines in `pages.jsonl`.

Options: `--index`, `--batch-size` (default 500), `--recreate` (drop and rebuild, needed after a
mapping change).

## 4. `4_store.py`: pgvector semantic index

**Input:** `pages.jsonl`. **Output:** Postgres tables `pages` and `chunks`, plus a
`pipeline_metadata` table that records which embedding model built the vectors.

- **Chunking:** about 200 words per chunk with a 40-word overlap. The title is prepended to
  every chunk.
- **Embeddings:** `intfloat/multilingual-e5-small` (384 dimensions, good French support).
  Chunks get the `"passage: "` prefix and queries get `"query: "`, as e5 expects. Vectors are
  normalized.
- **Index:** HNSW with `vector_cosine_ops`.
- **Cheap reruns:** before embedding, chunks whose `chunk_id` already exists with the same
  `text_hash` are skipped. When nothing changed, the model isn't even loaded. Work is committed
  every 256 pages, so an interrupted run resumes.
- Pages that disappear from `pages.jsonl` are deleted, and their chunks go with them
  (`ON DELETE CASCADE`).

Options: `--batch-size` (default 64), `--device cpu|cuda|mps` (auto-detected by default),
`--recreate`. If you change `EMBEDDING_MODEL`, the script refuses to run until you pass
`--recreate`, because the vector dimension may change.

## 5. `5_query.py`: hybrid search

```bash
uv run python 5_query.py "Roland-Garros 2015 finale Wawrinka Djokovic"
uv run python 5_query.py "victoire d'étape en montagne" --mode semantic --top-k 5
uv run python 5_query.py "Pinot Vuelta" --format jsonl | jq .url
```

- **Keyword leg:** Elasticsearch `multi_match` over `title^2` and `text`. Every hit counts as a
  match.
- **Semantic leg:** embeds the prompt, fetches the nearest chunks with pgvector's cosine
  distance `<=>`, and collapses them to pages by keeping each page's best chunk. A page counts
  as a match only if its similarity is at least `--min-similarity`.
- **Fusion:** Reciprocal Rank Fusion at page level, `score = Σ 1 / (60 + rank)` over both lists.
  Each result is labeled `keyword`, `semantic` or `both`.

| option | default | meaning |
|---|---|---|
| `--mode` | `hybrid` | `hybrid`, `keyword` or `semantic`, to compare the two methods |
| `--candidates` | 100 | max pages taken from each leg |
| `--min-similarity` | 0.85 | cosine threshold for a semantic hit to count |
| `--top-k` | 20 | cap on the final list (`0` = no cap) |
| `--format` | `text` | `text` (rank, title, date, URL, found-by, snippet) or `jsonl` |

**Why 0.85?** e5 models produce compressed similarity scores, so almost every French sports text
scores above 0.75 against any French query. On the smoke-test corpus (2018 Vuelta pages):

| query | best similarity |
|---|---|
| off-topic ("recette de gâteau au chocolat") | 0.815 |
| a topic absent from the sample ("Rafael Nadal à Roland-Garros") | 0.840 |
| relevant ("Thibaut Pinot sur le Tour d'Espagne") | 0.89 (13 pages ≥ 0.86) |

0.85 separates these cleanly. The plan's starting value of 0.80 accepted everything. Re-tune
on your own corpus with `--mode semantic --min-similarity 0 --format jsonl` and look at the
`similarity` field.

---

## Concepts, briefly

- **Inverted index (Elasticsearch):** a map from each word (after analysis: lowercasing, removing
  elisions and stopwords, stemming "victoires" → "victoir") to the list of documents that contain
  it. It answers "which documents contain these words?" very fast.
- **BM25:** the default ranking function in Elasticsearch. A document scores higher when it
  contains the query terms often (with diminishing returns), when those terms are rare across
  the corpus, and when the document is short.
- **Embeddings:** a neural model maps a text to a vector (here, 384 numbers). Texts with similar
  meaning end up close together, even when they share no words ("Vuelta" ≈ "Tour d'Espagne").
  Closeness is measured with cosine similarity.
- **Vector index (pgvector HNSW):** a graph structure that finds approximate nearest neighbours
  among millions of vectors without comparing against every one.
- **Chunking:** embedding models read a limited amount of text, and one vector for a whole long
  article blurs its meaning. Splitting pages into overlapping ~200-word chunks keeps each
  vector focused. The overlap keeps a sentence from being cut off from its context.
- **Hybrid search and RRF:** keyword search is precise on names, scores and dates. Semantic
  search catches paraphrases. Their scores live on different scales, so Reciprocal Rank Fusion
  ignores the scores and combines ranks instead: a page ranked well by both legs rises to the
  top.

## Limitations

- **No URL index before 2013:** Common Crawl's 2008–2012 archives are ARC files.
  `collinfo.json` lists them (`CC-MAIN-2008-2009`, `CC-MAIN-2009-2010`, `CC-MAIN-2012`), but
  they are out of scope and skipped.
- **Coverage is partial:** Common Crawl samples the web and doesn't capture every Eurosport
  page. In recent crawls (e.g. `CC-MAIN-2026-39`), many captures are redirects (HTTP 307) to
  Eurosport's geoblocking page, so expect fewer usable captures there.
- **Copyright:** the content belongs to Eurosport. This project is for educational use only.
  Keep the data local and don't redistribute it.

## Troubleshooting

- **Elasticsearch exits or never turns healthy:** give Docker at least 4 GB of RAM. On Linux,
  run `sudo sysctl -w vm.max_map_count=262144`. The heap is set to 1 GB in
  `docker-compose.yml` (`ES_JAVA_OPTS`).
- **CDX errors (429/502/503/504) or "Remote end closed connection":** the Common Crawl index
  server is rate-limited and sometimes down. The script retries with exponential backoff, and
  a failed crawl stays unfinished, so rerun the same command later. If the server is down, the
  crawl list falls back to `data/raw/collinfo.json`.
- **Slow embeddings on CPU:** use `--device mps` (Apple Silicon) or `--device cuda`, or lower
  the corpus size with `1_download.py --limit` or a crawl range. Reruns only embed new chunks.
- **"tables were built with X, config says Y":** you changed `EMBEDDING_MODEL`. Run
  `4_store.py --recreate`.
- **Hugging Face rate-limit warnings:** the model is downloaded once and then cached. Setting
  `HF_TOKEN` silences the warning.
