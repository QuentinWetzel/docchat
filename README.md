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
│   └── server.py          # FastAPI /chat + /chat/stream endpoints
├── scripts/
│   ├── 01_create_schema.py   # DDL: slides table + ivfflat/hnsw index
│   ├── 02_index_from_algolia.py  # pull records from Algolia, embed, upsert to pg
│   ├── 03_smoke_test.py      # end-to-end query against the live pipeline
│   └── stream_demo.py        # CLI consumer of /chat/stream, prints tokens + step latencies live
├── ui/                    # Gradio chat UI, deployed as its own container (no ML deps)
│   ├── app.py             # talks to the backend's /chat/stream over HTTP
│   ├── requirements.txt
│   ├── Dockerfile
│   └── railway.json
├── migrations/001_init.sql
├── requirements.txt
├── .env.example
├── railway.json
├── Dockerfile
└── docker-compose.yml     # backend + ui, for local dev
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

## Chat UI (Gradio)

A minimal chat UI lives in `ui/` and runs as its **own container**, separate from the backend —
it has no ML dependencies (no torch/transformers), it just calls the backend's `/chat/stream`
SSE endpoint over HTTP. Each assistant reply shows a collapsible **Steps** trace with per-stage
latency: query understanding (intent + extracted filters/queries), retrieval (lexical/semantic/
fusion/rerank hit counts), and the streamed answer (first-token + total latency). The trace
collapses once the answer finishes, with all latencies summarized in its (still-visible) title.

```bash
docker compose up --build
# backend: http://localhost:8000  (first boot downloads ~4.5GB of models, can take a few minutes)
# UI:      http://localhost:7860
```

The UI is stateless on the backend side (no conversation memory) — each turn sends only the
latest message; chat history is for display only, matching how `/chat/stream` already works.

## Railway notes

- Add the **Postgres** plugin, then in a shell: `CREATE EXTENSION IF NOT EXISTS vector;`
  (scripts/01 does this for you if the role has rights).
- Set the service env vars from `.env.example` in the Railway dashboard — `GEMINI_API_KEY`
  ([aistudio.google.com/apikey](https://aistudio.google.com/apikey)) as a secret, the rest as
  plain variables. `DATABASE_URL` should use Railway's `${{<PostgresServiceName>.DATABASE_URL}}`
  reference picker rather than the public proxy URL, so it resolves over the private network.
- The included `Dockerfile` / `railway.json` deploy the FastAPI service. Every boot downloads the
  bge-m3 + reranker weights (~4.5 GB combined) to the container's ephemeral disk — there's no
  persistent volume backing `HF_HOME=/models`, since that exceeds Railway Hobby's 5 GB volume cap.
  On a Pro plan (or higher volume limit), add a volume mounted at `/models` to avoid the
  re-download on every restart/redeploy.
- If you raise the service's vCPU limit (Settings -> Resource Limits) to speed up reranking, also
  set `RERANK_NUM_THREADS` to that same number. `os.cpu_count()` reports the underlying host's
  core count, not the container's cgroup quota, so without this the reranker will oversubscribe
  threads against the real limit and get slower, not faster.

### Deploying the Gradio UI as a second service

The backend (`docchat`, say) and the UI are two separate Railway services in the **same
project**, sharing the repo:

1. In the Railway project, **New Service -> GitHub Repo** -> pick this repo again.
2. Service Settings -> **Root Directory** -> `ui`. Railway will pick up `ui/Dockerfile` and
   `ui/railway.json` automatically (same `DOCKERFILE` builder as the backend).
3. Set one variable on the new service, using Railway's reference picker so it resolves over the
   private network rather than the public URL (same pattern as `DATABASE_URL` above):
   ```
   BACKEND_URL=http://${{<backend-service-name>.RAILWAY_PRIVATE_DOMAIN}}:${{<backend-service-name>.PORT}}
   ```
   (Replace `<backend-service-name>` with whatever the backend service is actually named —
   `docchat` if you haven't renamed it.)
4. Deploy. The UI's own `railway.json` sets `healthcheckPath: "/"`, which Gradio serves with a
   200 once it's up — no extra config needed there.

The UI container is tiny (`gradio` + `httpx`, no torch) so it builds and boots in seconds —
unlike the backend, it has nothing to do with the bge-m3/reranker cold-start cost above.
