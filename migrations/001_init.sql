-- Semantic store schema for chat-with-your-docs.
-- 1 row = 1 slide. Typed columns for common filter facets; JSONB `meta` for the rest.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS slides (
    object_id          TEXT PRIMARY KEY,          -- Algolia objectID
    file_id            TEXT,
    file_name          TEXT,
    title              TEXT,
    content            TEXT,
    slide_number       INTEGER,
    web_url            TEXT,

    -- filterable metadata (mirror Algolia stored values verbatim, including ";#Label|GUID")
    source             TEXT,
    drive_name         TEXT,
    site_display_name  TEXT,
    language           TEXT,
    client             TEXT,
    industry_sector    TEXT,
    service_line       TEXT,
    function           TEXT,
    document_purpose   TEXT,

    spo_created        BIGINT,                    -- epoch ms

    meta               JSONB DEFAULT '{}'::jsonb, -- background, path, timestamps, etc.

    embedding          vector(1024)               -- bge-m3 dense
);

-- B-tree indexes on the hot filter columns (these get ANDed into the KNN query's WHERE).
CREATE INDEX IF NOT EXISTS slides_client_idx           ON slides (client);
CREATE INDEX IF NOT EXISTS slides_industry_sector_idx  ON slides (industry_sector);
CREATE INDEX IF NOT EXISTS slides_service_line_idx     ON slides (service_line);
CREATE INDEX IF NOT EXISTS slides_language_idx         ON slides (language);
CREATE INDEX IF NOT EXISTS slides_source_idx           ON slides (source);
CREATE INDEX IF NOT EXISTS slides_drive_name_idx       ON slides (drive_name);
CREATE INDEX IF NOT EXISTS slides_document_purpose_idx ON slides (document_purpose);
CREATE INDEX IF NOT EXISTS slides_spo_created_idx       ON slides (spo_created);

-- Vector index. HNSW gives better recall/latency than ivfflat for ~14k rows and scales further.
-- cosine ops to match `1 - (embedding <=> q)` similarity used in queries.
CREATE INDEX IF NOT EXISTS slides_embedding_hnsw
    ON slides USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
