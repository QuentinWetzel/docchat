"""Semantic leg: Postgres + pgvector.

Embeds the query with bge-m3 and runs a cosine-distance KNN search, applying the *same*
MetadataFilterSpec as a SQL WHERE clause so the two legs are constrained identically.

We store metadata both as typed columns (for the common, indexable facets) and as a JSONB blob
(for everything else / future facets). Filters here use the typed columns.

Note on encoded facets: the values stored in pg are the SAME raw strings as in Algolia
(we copy them verbatim during indexing), so a filter value resolved to its raw form works
unchanged against both legs.
"""
from __future__ import annotations

from typing import Any

from .config import settings
from .schema import Candidate, MetadataFilterSpec, SlideRecord

# MetadataFilterSpec list-field -> pg column
_COL_MAP = {
    "client": "client",
    "industry_sector": "industry_sector",
    "service_line": "service_line",
    "function": "function",
    "document_purpose": "document_purpose",
    "language": "language",
    "source": "source",
    "drive_name": "drive_name",
    "site_display_name": "site_display_name",
}


def build_where(spec: MetadataFilterSpec) -> tuple[str, list[Any]]:
    """Return (where_sql, params). where_sql excludes the leading WHERE."""
    clauses: list[str] = []
    params: list[Any] = []
    for field, col in _COL_MAP.items():
        values = getattr(spec, field)
        if values:
            # OR within a field -> column = ANY(%s)
            clauses.append(f"{col} = ANY(%s)")
            params.append(values)
    if spec.created_after is not None:
        clauses.append("spo_created >= %s")
        params.append(spec.created_after)
    if spec.created_before is not None:
        clauses.append("spo_created <= %s")
        params.append(spec.created_before)
    return (" AND ".join(clauses), params)


def _row_to_record(row: dict[str, Any]) -> SlideRecord:
    meta = row.get("meta") or {}
    return SlideRecord(
        object_id=row["object_id"],
        file_id=row.get("file_id"),
        file_name=row.get("file_name"),
        title=row.get("title"),
        content=row.get("content"),
        background=meta.get("background"),
        slide_number=row.get("slide_number"),
        number_of_slides=meta.get("number_of_slides"),
        web_url=row.get("web_url"),
        path=meta.get("path"),
        source=row.get("source"),
        drive_name=row.get("drive_name"),
        drive_type=meta.get("drive_type"),
        drive_owner_id=meta.get("drive_owner_id"),
        site_display_name=row.get("site_display_name"),
        language=row.get("language"),
        client=row.get("client"),
        industry_sector=row.get("industry_sector"),
        service_line=row.get("service_line"),
        function=row.get("function"),
        document_purpose=row.get("document_purpose"),
        essential=meta.get("essential"),
        spo_created=row.get("spo_created"),
        spo_modified=meta.get("spo_modified"),
        pptx_created=meta.get("pptx_created"),
        pptx_modified=meta.get("pptx_modified"),
    )


class PgLeg:
    def __init__(self):
        import psycopg
        from pgvector.psycopg import register_vector

        self._psycopg = psycopg
        self.conn = psycopg.connect(settings.database_url, autocommit=True)
        register_vector(self.conn)
        self.table = settings.pg_table

    def search(self, query: str, spec: MetadataFilterSpec, top_k: int | None = None) -> list[Candidate]:
        from .embeddings import get_embedder

        top_k = top_k or settings.semantic_top_k
        qvec = get_embedder().encode_one(query)

        where_sql, where_params = build_where(spec)
        where_clause = f"WHERE {where_sql}" if where_sql else ""

        # <=> is cosine distance in pgvector; similarity = 1 - distance.
        sql = f"""
            SELECT object_id, file_id, file_name, title, content, slide_number, web_url,
                   source, drive_name, site_display_name, language,
                   client, industry_sector, service_line, function, document_purpose,
                   spo_created, meta,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM {self.table}
            {where_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params = [qvec, *where_params, qvec, top_k]

        with self.conn.cursor(row_factory=self._psycopg.rows.dict_row) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        candidates: list[Candidate] = []
        for rank, row in enumerate(rows, start=1):
            rec = _row_to_record(row)
            candidates.append(
                Candidate(
                    object_id=rec.object_id,
                    record=rec,
                    semantic_rank=rank,
                    semantic_score=float(row["similarity"]),
                )
            )
        return candidates
