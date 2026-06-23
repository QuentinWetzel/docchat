"""Backfill the pgvector store from the Algolia index.

Algolia's regular search paginates only up to ~1000 hits, so to copy the *whole* corpus
(~13.9k slides) we use `browse_objects`, which cursors through every record. Each record's
text (title+content+background+file_name) is embedded with bge-m3 and upserted into pg.

Idempotent: re-running updates existing rows by object_id.
Run after 01_create_schema.py.
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.config import settings          # noqa: E402
from app.embeddings import get_embedder  # noqa: E402

BATCH = 256


def browse_all_records(on_page: Any) -> None:
    from algoliasearch.search.client import SearchClientSync

    client = SearchClientSync(settings.algolia_app_id, settings.algolia_api_key)
    # browse_objects cursors through the entire index regardless of size, calling
    # `on_page` (the aggregator) once per page with a BrowseResponse.
    client.browse_objects(index_name=settings.algolia_index_name, aggregator=on_page)


def record_text(h: dict[str, Any]) -> str:
    parts = [h.get("title"), h.get("content"), h.get("background"), h.get("file_name")]
    return "\n".join(p for p in parts if p)


def to_row(h: dict[str, Any], vec: list[float]) -> tuple:
    meta = {
        "background": h.get("background"),
        "path": h.get("path"),
        "drive_type": h.get("drive_type"),
        "drive_owner_id": h.get("drive_owner_id"),
        "number_of_slides": h.get("number_of_slides"),
        "essential": h.get("Essential"),
        "spo_modified": h.get("spo_modified"),
        "pptx_created": h.get("pptx_created"),
        "pptx_modified": h.get("pptx_modified"),
    }
    return (
        h["objectID"],
        h.get("file_id"),
        h.get("file_name"),
        h.get("title"),
        h.get("content"),
        h.get("slide_number"),
        h.get("web_url"),
        h.get("source"),
        h.get("drive_name"),
        h.get("site_display_name"),
        h.get("language"),
        h.get("Client"),
        h.get("IndustrySector"),
        h.get("ServiceLine"),
        h.get("Function"),
        h.get("DocumentPurpose"),
        h.get("spo_created"),
        json.dumps(meta),
        vec,
    )


UPSERT = f"""
INSERT INTO {settings.pg_table}
    (object_id, file_id, file_name, title, content, slide_number, web_url,
     source, drive_name, site_display_name, language,
     client, industry_sector, service_line, function, document_purpose,
     spo_created, meta, embedding)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON CONFLICT (object_id) DO UPDATE SET
    file_id=EXCLUDED.file_id, file_name=EXCLUDED.file_name, title=EXCLUDED.title,
    content=EXCLUDED.content, slide_number=EXCLUDED.slide_number, web_url=EXCLUDED.web_url,
    source=EXCLUDED.source, drive_name=EXCLUDED.drive_name,
    site_display_name=EXCLUDED.site_display_name, language=EXCLUDED.language,
    client=EXCLUDED.client, industry_sector=EXCLUDED.industry_sector,
    service_line=EXCLUDED.service_line, function=EXCLUDED.function,
    document_purpose=EXCLUDED.document_purpose, spo_created=EXCLUDED.spo_created,
    meta=EXCLUDED.meta, embedding=EXCLUDED.embedding;
"""


def flush(conn, rows: list[tuple]) -> None:
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(UPSERT, rows)


def main() -> None:
    embedder = get_embedder()
    conn = psycopg.connect(settings.database_url, autocommit=False)
    register_vector(conn)

    buf_records: list[dict[str, Any]] = []
    total = 0

    def process_batch():
        nonlocal total
        texts = [record_text(h) or (h.get("file_name") or "") for h in buf_records]
        vecs = embedder.encode(texts)
        rows = [to_row(h, v) for h, v in zip(buf_records, vecs)]
        flush(conn, rows)
        conn.commit()
        total += len(rows)
        print(f"  upserted {total} records...", flush=True)
        buf_records.clear()

    def on_page(resp: Any) -> None:
        for hit in resp.hits:
            h = hit if isinstance(hit, dict) else hit.to_dict()
            if "objectID" not in h:
                continue
            buf_records.append(h)
            if len(buf_records) >= BATCH:
                process_batch()

    print("Browsing Algolia and indexing into pgvector...")
    browse_all_records(on_page)
    if buf_records:
        process_batch()

    conn.close()
    print(f"Done. {total} records indexed.")


if __name__ == "__main__":
    main()
