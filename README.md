# Chat-with-your-docs — Metadata-filtered Hybrid Pipeline

A production-shaped RAG pipeline over your slide-level corpus (1 record = 1 PowerPoint slide,
~13.9k records sourced from SharePoint / OneDrive). It combines:

- **Lexical leg** — your existing **Algolia** index (BM25-style keyword + faceting), called via MCP/REST.
- **Semantic leg** — **Postgres + pgvector** on **Railway**, embeddings via a bi-encoder.
- **Fusion** — Reciprocal Rank Fusion (RRF) over the two legs.
- **Rerank** — a **cross-encoder** reranker over the fused candidate set.
- **Generation** — **Gemini 2.5 Flash** (`gemini-2.5-flash`) via the Gemini Developer API + LlamaIndex.
- **Metadata filtering** — a single filter spec is translated to *both* Algolia facet filters
  *and* a pgvector `WHERE` clause, so the two legs stay consistent.

```
                         ┌──────────────────────────────┐
   user query  ─────────▶│   query understanding         │  (Gemini 2.5 Flash, structured output)
   + filters             │   → filters, lexical_query,   │
                         │     semantic_query, intent_type│
                         └────────────┬───────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                                 ▼
   ┌────────────────────┐                          ┌────────────────────────┐
   │ Algolia (lexical)  │  lexical_query            │ pgvector (semantic)    │
   │ BM25 + facetFilters│  + facetFilters           │ semantic_query         │
   │                    │                          │ cosine + WHERE filter  │
   └─────────┬──────────┘                          └───────────┬────────────┘
             │ ranked hits                                      │ ranked hits
             └───────────────────┬──────────────────────────────┘
                                 ▼
                       ┌───────────────────┐
                       │  RRF fusion       │   top-K candidates
                       └─────────┬─────────┘
                                 ▼
                       ┌───────────────────┐
                       │  cross-encoder    │   rerank → top-N
                       │  reranker         │
                       └─────────┬─────────┘
                                 ▼
                       ┌───────────────────┐
                       │  Gemini 2.5 Flash │   grounded answer + slide citations
                       └───────────────────┘
```

## Model choices

| Role | Model | Why |
|------|-------|-----|
| LLM | `gemini-2.5-flash` | Your pick. Fast, cheap grounded synthesis + structured filter extraction via the Gemini Developer API. `gemini-3.5-flash` exists but needs a separate access grant on this project (see `.env.example`). |
| Bi-encoder (embeddings) | `BAAI/bge-m3` | Multilingual (your corpus is FR/EN/DE), 1024-dim, strong retrieval, good on short slide text. |
| Cross-encoder (rerank) | `BAAI/bge-reranker-v2-m3` | Multilingual reranker matched to bge-m3; big precision lift on the fused set. |

Both embedding models run locally via `sentence-transformers` / `FlagEmbedding`, so there is no
extra embedding-API cost or data-egress. Swap to a hosted embedding API by editing `app/embeddings.py`.

> **Multilingual matters here.** Your `language` facet shows French (majority), English, German,
> and Other. A monolingual English encoder would underperform — bge-m3 is the deliberate choice.

## Layout

```
docchat/
├── app/
│   ├── config.py          # env-driven settings
│   ├── schema.py          # Pydantic: MetadataFilterSpec, QueryUnderstanding, Candidate, etc.
│   ├── taxonomy.py        # decode SharePoint ";#Label|GUID" facet encoding
│   ├── embeddings.py      # bge-m3 bi-encoder wrapper
│   ├── reranker.py        # bge-reranker-v2-m3 cross-encoder wrapper
│   ├── algolia_leg.py     # lexical retrieval + facet filter translation
│   ├── pg_leg.py          # pgvector retrieval + WHERE filter translation
│   ├── fusion.py          # reciprocal rank fusion
│   ├── query_understanding.py  # NL query -> {filters, lexical_query, semantic_query, intent_type} via Gemini
│   ├── pipeline.py        # orchestration: retrieve -> fuse -> rerank -> generate
│   └── server.py          # FastAPI /chat endpoint
├── scripts/
│   ├── 01_create_schema.py   # DDL: slides table + ivfflat/hnsw index
│   ├── 02_index_from_algolia.py  # pull records from Algolia, embed, upsert to pg
│   └── 03_smoke_test.py      # end-to-end query against the live pipeline
├── migrations/001_init.sql
├── requirements.txt
├── .env.example
├── railway.json
└── Dockerfile
```

## Quick start

```bash
# 0. Python 3.11+
pip install -r requirements.txt

# 1. Provision Postgres + pgvector on Railway, then set env (see .env.example)
cp .env.example .env   # fill in values

# 2. Create the schema (enables pgvector, creates table + vector index)
python scripts/01_create_schema.py

# 3. Backfill: pull every record from Algolia, embed, upsert into pgvector
python scripts/02_index_from_algolia.py

# 4. Smoke-test the full pipeline
python scripts/03_smoke_test.py "What did we propose to Airbus on supply chain?"

# 5. Serve
uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8000}
```

## Railway notes

- Add the **Postgres** plugin, then in a shell: `CREATE EXTENSION IF NOT EXISTS vector;`
  (scripts/01 does this for you if the role has rights).
- Set the service env vars from `.env.example` in the Railway dashboard — `GEMINI_API_KEY`
  ([aistudio.google.com/apikey](https://aistudio.google.com/apikey)) as a secret, the rest as
  plain variables. `DATABASE_URL` should use Railway's `${{<PostgresServiceName>.DATABASE_URL}}`
  reference picker rather than the public proxy URL, so it resolves over the private network.
- The included `Dockerfile` / `railway.json` deploy the FastAPI service. The first boot downloads
  the bge models (~2.3 GB); use a Railway volume mounted at `/models` and set `HF_HOME=/models`
  to avoid re-downloading on every deploy.
- If you raise the service's vCPU limit (Settings -> Resource Limits) to speed up reranking, also
  set `RERANK_NUM_THREADS` to that same number. `os.cpu_count()` reports the underlying host's
  core count, not the container's cgroup quota, so without this the reranker will oversubscribe
  threads against the real limit and get slower, not faster.
