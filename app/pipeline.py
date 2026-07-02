"""End-to-end pipeline: filter -> dual retrieval -> RRF -> cross-encoder rerank -> Gemini.

Resolution of encoded facet values (Client, IndustrySector, etc.) to their raw stored form is
done lazily against Algolia's facet endpoint and cached, so the same MetadataFilterSpec drives
both legs consistently.
"""
from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any, Iterator

from google.genai import types as genai_types
from llama_index.core.llms import ChatMessage
from llama_index.llms.google_genai import GoogleGenAI

from .algolia_leg import AlgoliaLeg
from .config import settings
from .fusion import reciprocal_rank_fusion
from .pg_leg import PgLeg
from .query_understanding import understand_query
from .reranker import get_reranker
from .schema import (
    Candidate,
    ChatRequest,
    ChatResponse,
    Citation,
    MetadataFilterSpec,
    QueryUnderstanding,
)
from .taxonomy import build_label_index, resolve_to_raw

# Fields whose Algolia stored values are SharePoint-encoded and need label->raw resolution.
_ENCODED_FIELDS = {
    "client": "Client",
    "industry_sector": "IndustrySector",
    "service_line": "ServiceLine",
    "document_purpose": "DocumentPurpose",
}

logger = logging.getLogger(__name__)

_ANSWER_SYSTEM = """You are a precise assistant answering questions strictly from the provided
slide excerpts (each from a PowerPoint deck in the company's document libraries). Rules:
- Use ONLY the supplied excerpts. If they do not contain the answer, say so plainly.
- Cite slides inline as [n] matching the numbered excerpts you used.
- Be concise and concrete. Prefer specifics (client, figures, approach) found in the slides.
- The corpus is multilingual (French/English/German). Answer in the user's language."""

_NO_RESULTS_MSG = "I couldn't find any slides matching that query and the applied filters."
_GENERATION_FAILED_MSG = (
    "I found relevant slides but couldn't generate an answer from them just now. "
    "Please try again."
)


class Pipeline:
    def __init__(self):
        self.algolia = AlgoliaLeg()
        self.pg = PgLeg()
        self.reranker = get_reranker()
        self.llm = GoogleGenAI(
            model=settings.llm_model,
            api_key=settings.gemini_api_key,
            max_tokens=2048,
            # Gemini 2.5's dynamic thinking budget eats into max_output_tokens and can exhaust
            # it before any visible answer text is produced, which llama-index surfaces as a
            # RuntimeError("...MAX_TOKENS") with no partial content recoverable. This is a
            # grounded-extraction task, not multi-step reasoning, so disable thinking outright.
            generation_config=genai_types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2048,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )

    # --- encoded-facet resolution -------------------------------------------------
    @lru_cache(maxsize=8)
    def _label_index(self, facet: str) -> dict:
        """Fetch distinct raw values for a facet from Algolia and build a label->raw index."""
        resp = self.algolia.client.search_for_facet_values(
            index_name=self.algolia.index_name,
            facet_name=facet,
            search_for_facet_values_request={"maxFacetHits": 100},
        )
        raws = [h["value"] if isinstance(h, dict) else h.value
                for h in resp.to_dict().get("facetHits", [])]
        return build_label_index(raws)

    def _resolve_spec(self, spec: MetadataFilterSpec) -> MetadataFilterSpec:
        """Replace decoded labels with raw stored values for encoded facets."""
        resolved = spec.model_copy(deep=True)
        for field, facet in _ENCODED_FIELDS.items():
            values = getattr(resolved, field)
            if not values:
                continue
            idx = self._label_index(facet)
            setattr(resolved, field, [raw for v in values for raw in resolve_to_raw(v, idx)])
        return resolved

    # --- main entrypoint ----------------------------------------------------------
    def run(self, req: ChatRequest) -> ChatResponse:
        qu, lexical, semantic, fused, top, timings = self._retrieve(req)

        # 5. Generate grounded answer.
        t0 = time.monotonic()
        answer, citations = self._generate(req.query, top)
        timings["generation_ms"] = round((time.monotonic() - t0) * 1000, 1)
        timings["total_ms"] = round(sum(timings.values()), 1)

        return ChatResponse(
            answer=answer,
            citations=citations,
            used_filters=qu.filters,  # report the human-label spec, not the raw-resolved one
            intent_type=qu.intent_type,
            lexical_query=qu.lexical_query,
            semantic_query=qu.semantic_query,
            n_lexical=len(lexical),
            n_semantic=len(semantic),
            n_after_fusion=len(fused),
            n_after_rerank=len(top),
            timings=timings,
        )

    def run_stream(self, req: ChatRequest) -> Iterator[dict[str, Any]]:
        """Same pipeline as `run`, but the final Gemini call is streamed token-by-token, and the
        two retrieval stages are surfaced as soon as each is ready rather than bundled together.

        Yields dicts: a `qu` event the moment query understanding finishes, a `retrieval` event
        once dual retrieval + fusion + rerank finish (these can be seconds apart from `qu` --
        rerank in particular isn't free), then one `delta` event per generated token, then a
        closing `done` event carrying generation/first-token/total latency.
        """
        qu, resolved_spec, qu_ms = self._understand(req)
        yield {
            "type": "qu",
            "intent_type": qu.intent_type,
            "used_filters": qu.filters.model_dump(),
            "lexical_query": qu.lexical_query,
            "semantic_query": qu.semantic_query,
            "timings": {"query_understanding_ms": qu_ms},
        }

        lexical, semantic, fused, top, retrieval_ms = self._search_fuse_rerank(req, qu, resolved_spec)
        user_msg, citations = self._build_context(req.query, top)
        timings = {"query_understanding_ms": qu_ms, "retrieval_ms": retrieval_ms}

        yield {
            "type": "retrieval",
            "citations": [c.model_dump() for c in citations],
            "n_lexical": len(lexical),
            "n_semantic": len(semantic),
            "n_after_fusion": len(fused),
            "n_after_rerank": len(top),
            "timings": {"retrieval_ms": retrieval_ms},
        }

        if user_msg is None:
            yield {"type": "delta", "text": _NO_RESULTS_MSG}
            yield {
                "type": "done",
                "timings": {
                    **timings,
                    "first_token_ms": None,
                    "generation_ms": 0.0,
                    "total_ms": timings["query_understanding_ms"] + timings["retrieval_ms"],
                },
            }
            return

        gen_start = time.monotonic()
        first_token_ms: float | None = None
        try:
            for chunk in self.llm.stream_chat(
                [
                    ChatMessage(role="system", content=_ANSWER_SYSTEM),
                    ChatMessage(role="user", content=user_msg),
                ]
            ):
                if chunk.delta:
                    if first_token_ms is None:
                        first_token_ms = round((time.monotonic() - gen_start) * 1000, 1)
                    yield {"type": "delta", "text": chunk.delta}
        except Exception:
            # Same degrade-don't-fail rule as the non-streaming path (see _generate).
            logger.exception("LLM stream_chat failed")
            yield {"type": "delta", "text": _GENERATION_FAILED_MSG}
        generation_ms = round((time.monotonic() - gen_start) * 1000, 1)
        yield {
            "type": "done",
            "timings": {
                **timings,
                "first_token_ms": first_token_ms,
                "generation_ms": generation_ms,
                "total_ms": round(
                    timings["query_understanding_ms"] + timings["retrieval_ms"] + generation_ms, 1
                ),
            },
        }

    # --- shared retrieval steps -----------------------------------------------------
    def _understand(self, req: ChatRequest) -> tuple[QueryUnderstanding, MetadataFilterSpec, float]:
        """Step 1 alone: explicit UI filters merged with LLM-extracted ones, plus the
        lexical/semantic text each leg should actually search with. Split out from
        `_search_fuse_rerank` so streaming callers can surface this the moment it's ready,
        instead of waiting on retrieval too."""
        t0 = time.monotonic()

        if req.explicit_filters_only:
            qu = QueryUnderstanding(
                intent_type="hybrid",
                filters=req.filters or MetadataFilterSpec(),
                lexical_query=req.query,
                semantic_query=req.query,
            )
        else:
            qu = understand_query(req.query)
            if req.filters:
                qu.filters = _merge_specs(qu.filters, req.filters)

        resolved_spec = self._resolve_spec(qu.filters)
        qu_ms = round((time.monotonic() - t0) * 1000, 1)
        return qu, resolved_spec, qu_ms

    def _search_fuse_rerank(
        self, req: ChatRequest, qu: QueryUnderstanding, resolved_spec: MetadataFilterSpec
    ) -> tuple[list[Candidate], list[Candidate], list[Candidate], list[Candidate], float]:
        """Steps 2-4: dual retrieval, each on its own slot of the query (each leg applies the
        same filter), fuse, then cross-encoder rerank.

        A null slot means that leg has nothing useful to search on: the lexical leg falls back
        to a filters-only browse (empty query text); the semantic leg is skipped outright, since
        "nearest neighbors of nothing" isn't meaningful.
        """
        t0 = time.monotonic()

        lexical = self.algolia.search(qu.lexical_query or "", resolved_spec, settings.lexical_top_k)
        semantic = (
            self.pg.search(qu.semantic_query, resolved_spec, settings.semantic_top_k)
            if qu.semantic_query
            else []
        )

        fused = reciprocal_rank_fusion(lexical, semantic)

        reranked = self._rerank(req.query, fused)
        top = reranked[: settings.rerank_top_n]

        retrieval_ms = round((time.monotonic() - t0) * 1000, 1)
        return lexical, semantic, fused, top, retrieval_ms

    def _retrieve(
        self, req: ChatRequest
    ) -> tuple[
        QueryUnderstanding, list[Candidate], list[Candidate], list[Candidate], list[Candidate],
        dict[str, float],
    ]:
        """Convenience wrapper over `_understand` + `_search_fuse_rerank` for non-streaming
        callers that don't need the two stages surfaced separately."""
        qu, resolved_spec, qu_ms = self._understand(req)
        lexical, semantic, fused, top, retrieval_ms = self._search_fuse_rerank(req, qu, resolved_spec)
        timings = {"query_understanding_ms": qu_ms, "retrieval_ms": retrieval_ms}
        return qu, lexical, semantic, fused, top, timings

    def _rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        if not candidates:
            return []
        passages = [c.record.text_for_embedding() for c in candidates]
        scores = self.reranker.score(query, passages)
        for c, s in zip(candidates, scores):
            c.rerank_score = float(s)
        return sorted(candidates, key=lambda x: x.rerank_score, reverse=True)

    def _build_context(
        self, query: str, top: list[Candidate]
    ) -> tuple[str | None, list[Citation]]:
        """Build the grounded-generation prompt + citations. user_msg is None when there's
        nothing to generate from (caller should short-circuit with _NO_RESULTS_MSG)."""
        if not top:
            return (None, [])

        blocks = []
        citations: list[Citation] = []
        for i, c in enumerate(top, start=1):
            r = c.record
            text = r.text_for_embedding()
            loc = f"{r.file_name or 'unknown file'} (slide {r.slide_number})"
            blocks.append(f"[{i}] {loc} | source: {r.source}\n{text}")
            citations.append(
                Citation(
                    object_id=r.object_id,
                    file_name=r.file_name,
                    slide_number=r.slide_number,
                    web_url=r.web_url,
                    source=r.source,
                    snippet=(text[:300] + "…") if len(text) > 300 else text,
                    rerank_score=c.rerank_score,
                )
            )

        context = "\n\n".join(blocks)
        user_msg = (
            f"Question:\n{query}\n\n"
            f"Slide excerpts:\n{context}\n\n"
            "Answer the question using only these excerpts, with inline [n] citations."
        )
        return (user_msg, citations)

    def _generate(self, query: str, top: list[Candidate]) -> tuple[str, list[Citation]]:
        user_msg, citations = self._build_context(query, top)
        if user_msg is None:
            return (_NO_RESULTS_MSG, citations)
        try:
            resp = self.llm.chat(
                [
                    ChatMessage(role="system", content=_ANSWER_SYSTEM),
                    ChatMessage(role="user", content=user_msg),
                ]
            )
        except Exception:
            # e.g. the model hit a non-STOP finish reason (MAX_TOKENS, SAFETY, ...); no partial
            # content is recoverable from that path, so degrade instead of failing the request.
            logger.exception("LLM chat failed")
            return (_GENERATION_FAILED_MSG, citations)
        return (resp.message.content or "", citations)


def _merge_specs(a: MetadataFilterSpec, b: MetadataFilterSpec) -> MetadataFilterSpec:
    """Union list fields; explicit (b) wins for numeric ranges if set."""
    out = a.model_copy(deep=True)
    for field in [
        "client", "industry_sector", "service_line", "function", "document_purpose",
        "language", "source", "drive_name", "site_display_name",
    ]:
        merged = list(dict.fromkeys(getattr(a, field) + getattr(b, field)))
        setattr(out, field, merged)
    out.created_after = b.created_after if b.created_after is not None else a.created_after
    out.created_before = b.created_before if b.created_before is not None else a.created_before
    return out


@lru_cache(maxsize=1)
def get_pipeline() -> Pipeline:
    return Pipeline()
