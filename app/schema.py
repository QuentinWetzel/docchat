"""Pydantic models shared across the pipeline.

The field set mirrors the Algolia record (1 record = 1 slide) discovered from the index:
searchable text lives in title/content/background/file_name; the rest are metadata used for
filtering and citation.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class SlideRecord(BaseModel):
    """One slide. Mirrors the Algolia document; all non-text fields are metadata."""

    object_id: str                      # Algolia objectID, e.g. "01..s1" — our primary key
    file_id: Optional[str] = None
    file_name: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    background: Optional[str] = None
    slide_number: Optional[int] = None
    number_of_slides: Optional[int] = None
    web_url: Optional[str] = None
    path: Optional[str] = None

    # --- filterable metadata ---
    source: Optional[str] = None            # e.g. "OneShelf > Proposals Library"
    drive_name: Optional[str] = None
    drive_type: Optional[str] = None
    drive_owner_id: Optional[str] = None
    site_display_name: Optional[str] = None
    language: Optional[str] = None          # French / English / German / Other

    # SharePoint managed-metadata facets (raw values may be ";#Label|GUID" encoded)
    client: Optional[str] = None
    industry_sector: Optional[str] = None
    service_line: Optional[str] = None
    function: Optional[str] = None
    document_purpose: Optional[str] = None
    essential: Optional[str] = None

    # timestamps (epoch ms)
    spo_created: Optional[int] = None
    spo_modified: Optional[int] = None
    pptx_created: Optional[int] = None
    pptx_modified: Optional[int] = None

    def text_for_embedding(self) -> str:
        """Concatenated text used both for embedding and for the LLM context block."""
        parts = [self.title, self.content, self.background, self.file_name]
        return "\n".join(p for p in parts if p)


class MetadataFilterSpec(BaseModel):
    """A single, leg-agnostic filter description.

    Translated into Algolia facetFilters AND a pgvector SQL WHERE clause so both legs
    apply the *same* constraints. List values mean OR within a field; different fields are AND-ed.
    """

    client: list[str] = Field(default_factory=list)
    industry_sector: list[str] = Field(default_factory=list)
    service_line: list[str] = Field(default_factory=list)
    function: list[str] = Field(default_factory=list)
    document_purpose: list[str] = Field(default_factory=list)
    language: list[str] = Field(default_factory=list)
    source: list[str] = Field(default_factory=list)
    drive_name: list[str] = Field(default_factory=list)
    site_display_name: list[str] = Field(default_factory=list)

    # numeric range on spo_created (epoch ms); None = unbounded
    created_after: Optional[int] = None
    created_before: Optional[int] = None

    def is_empty(self) -> bool:
        return (
            not any(
                [
                    self.client, self.industry_sector, self.service_line, self.function,
                    self.document_purpose, self.language, self.source, self.drive_name,
                    self.site_display_name,
                ]
            )
            and self.created_after is None
            and self.created_before is None
        )


class QueryUnderstanding(BaseModel):
    """The query, decomposed for a hybrid retrieval system.

    `filters` are exact constraints. `lexical_query` and `semantic_query` are the residue once
    filter terms are pulled out — what's left to search the lexical leg and semantic leg with,
    respectively. They may overlap (a term can be both an exact phrase to match and part of the
    concept to embed), and either may be None when that leg has nothing useful to search on
    (e.g. a pure browse request has no semantic_query; a pure concept question has no
    lexical_query).
    """

    intent_type: Literal["aggregate", "lookup", "filtered_browse", "semantic_search", "hybrid"]
    filters: MetadataFilterSpec = Field(default_factory=MetadataFilterSpec)
    lexical_query: Optional[str] = None
    semantic_query: Optional[str] = None


class Candidate(BaseModel):
    """A retrieval hit flowing through fusion and rerank."""

    object_id: str
    record: SlideRecord
    lexical_rank: Optional[int] = None   # 1-based rank in Algolia results (None if absent)
    semantic_rank: Optional[int] = None  # 1-based rank in pgvector results (None if absent)
    semantic_score: Optional[float] = None  # cosine similarity
    fused_score: float = 0.0
    rerank_score: Optional[float] = None


class ChatRequest(BaseModel):
    query: str
    # Optional explicit filters from the UI; merged with LLM-extracted filters.
    filters: Optional[MetadataFilterSpec] = None
    # If True, skip LLM filter extraction and use only `filters`.
    explicit_filters_only: bool = False
    # If False, skip the Gemini synthesis and return citations only (retrieval-only callers).
    generate: bool = True
    # Overrides settings.rerank_top_n for this request; None uses the server default.
    top_k: Optional[int] = None


class Citation(BaseModel):
    object_id: str
    file_name: Optional[str]
    slide_number: Optional[int]
    web_url: Optional[str]
    source: Optional[str]
    snippet: str
    rerank_score: Optional[float] = None
    # Decoded Client label (human-readable) — provenance for leak attribution.
    client: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    used_filters: MetadataFilterSpec
    intent_type: str
    lexical_query: Optional[str]
    semantic_query: Optional[str]
    n_lexical: int
    n_semantic: int
    n_after_fusion: int
    n_after_rerank: int
    # Stage latencies in ms: query_understanding_ms, retrieval_ms, generation_ms, total_ms.
    timings: dict[str, float] = Field(default_factory=dict)
