"""Reciprocal Rank Fusion (RRF).

RRF combines two ranked lists using only ranks (not raw scores), which makes it robust to the
fact that Algolia relevance and pgvector cosine are not on the same scale. For a document d:

    score(d) = sum over legs of  weight_leg / (k + rank_leg(d))

Documents present in both legs naturally score higher. k (default 60) damps the influence of
very high ranks.
"""
from __future__ import annotations

from .config import settings
from .schema import Candidate


def reciprocal_rank_fusion(
    lexical: list[Candidate],
    semantic: list[Candidate],
    k: int | None = None,
    w_lex: float | None = None,
    w_sem: float | None = None,
) -> list[Candidate]:
    k = k or settings.rrf_k
    w_lex = settings.rrf_lexical_weight if w_lex is None else w_lex
    w_sem = settings.rrf_semantic_weight if w_sem is None else w_sem

    merged: dict[str, Candidate] = {}

    def _merge(c: Candidate) -> Candidate:
        if c.object_id not in merged:
            merged[c.object_id] = Candidate(object_id=c.object_id, record=c.record)
        existing = merged[c.object_id]
        # carry over ranks/scores from whichever leg this came from
        if c.lexical_rank is not None:
            existing.lexical_rank = c.lexical_rank
        if c.semantic_rank is not None:
            existing.semantic_rank = c.semantic_rank
            existing.semantic_score = c.semantic_score
        return existing

    for c in lexical:
        _merge(c)
    for c in semantic:
        _merge(c)

    for c in merged.values():
        score = 0.0
        if c.lexical_rank is not None:
            score += w_lex / (k + c.lexical_rank)
        if c.semantic_rank is not None:
            score += w_sem / (k + c.semantic_rank)
        c.fused_score = score

    ranked = sorted(merged.values(), key=lambda x: x.fused_score, reverse=True)
    return ranked[: settings.fused_top_k]
