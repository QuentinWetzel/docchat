"""Lexical leg: Algolia.

Translates a MetadataFilterSpec into Algolia `facetFilters` + numeric `filters`, runs the
keyword search, and returns ranked SlideRecords.

Field -> facet mapping is derived from the live index. SharePoint-encoded facets (Client,
IndustrySector, ServiceLine, DocumentPurpose, Essential) need the *raw* stored value
(";#Label|GUID"); we resolve user labels to raw via a per-field label index (see taxonomy.py).

Two backends:
  - REST (default): the `algoliasearch` Python client. Use this in production.
  - MCP: if you prefer to route through the Algolia MCP server you already have connected,
    call the `algolia_search_index_*` tool with the same params shape (query, facetFilters via
    the per-facet facet_* args, hitsPerPage). The translation logic below is identical; only the
    transport differs.
"""
from __future__ import annotations

from typing import Any

from .config import settings
from .schema import Candidate, MetadataFilterSpec, SlideRecord

# MetadataFilterSpec field  ->  Algolia facet attribute name
_FACET_MAP = {
    "client": "Client",
    "industry_sector": "IndustrySector",
    "service_line": "ServiceLine",
    "function": "Function",
    "document_purpose": "DocumentPurpose",
    "language": "language",
    "source": "source",
    "drive_name": "drive_name",
    "site_display_name": "site_display_name",
}

# Facets whose stored values are SharePoint-encoded "<id>;#Label|GUID".
# For these we should resolve user labels to the raw value before sending.
_ENCODED_FACETS = {"Client", "IndustrySector", "ServiceLine", "DocumentPurpose"}


def build_facet_filters(spec: MetadataFilterSpec) -> list[list[str]]:
    """Build Algolia facetFilters.

    Algolia semantics: outer list = AND, inner list = OR.
    So we emit one inner OR-group per non-empty field.
    """
    groups: list[list[str]] = []
    for field, facet in _FACET_MAP.items():
        values = getattr(spec, field)
        if values:
            groups.append([f"{facet}:{v}" for v in values])
    return groups


def build_numeric_filters(spec: MetadataFilterSpec) -> str:
    clauses = []
    if spec.created_after is not None:
        clauses.append(f"spo_created >= {spec.created_after}")
    if spec.created_before is not None:
        clauses.append(f"spo_created <= {spec.created_before}")
    return " AND ".join(clauses)


def _hit_to_record(hit: dict[str, Any]) -> SlideRecord:
    return SlideRecord(
        object_id=hit["objectID"],
        file_id=hit.get("file_id"),
        file_name=hit.get("file_name"),
        title=hit.get("title"),
        content=hit.get("content"),
        background=hit.get("background"),
        slide_number=hit.get("slide_number"),
        number_of_slides=hit.get("number_of_slides"),
        web_url=hit.get("web_url"),
        path=hit.get("path"),
        source=hit.get("source"),
        drive_name=hit.get("drive_name"),
        drive_type=hit.get("drive_type"),
        drive_owner_id=hit.get("drive_owner_id"),
        site_display_name=hit.get("site_display_name"),
        language=hit.get("language"),
        client=hit.get("Client"),
        industry_sector=hit.get("IndustrySector"),
        service_line=hit.get("ServiceLine"),
        function=hit.get("Function"),
        document_purpose=hit.get("DocumentPurpose"),
        essential=hit.get("Essential"),
        spo_created=hit.get("spo_created"),
        spo_modified=hit.get("spo_modified"),
        pptx_created=hit.get("pptx_created"),
        pptx_modified=hit.get("pptx_modified"),
    )


class AlgoliaLeg:
    def __init__(self):
        from algoliasearch.search.client import SearchClientSync

        self.client = SearchClientSync(settings.algolia_app_id, settings.algolia_api_key)
        self.index_name = settings.algolia_index_name

    def search(self, query: str, spec: MetadataFilterSpec, top_k: int | None = None) -> list[Candidate]:
        top_k = top_k or settings.lexical_top_k
        params: dict[str, Any] = {"query": query, "hitsPerPage": top_k}

        facet_filters = build_facet_filters(spec)
        if facet_filters:
            params["facetFilters"] = facet_filters
        numeric = build_numeric_filters(spec)
        if numeric:
            params["filters"] = numeric

        resp = self.client.search_single_index(
            index_name=self.index_name,
            search_params=params,
        )
        hits = resp.to_dict().get("hits", [])

        candidates: list[Candidate] = []
        for rank, hit in enumerate(hits, start=1):
            rec = _hit_to_record(hit)
            candidates.append(Candidate(object_id=rec.object_id, record=rec, lexical_rank=rank))
        return candidates
