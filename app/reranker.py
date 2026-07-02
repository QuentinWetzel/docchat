"""Cross-encoder reranker using BAAI/bge-reranker-v2-m3 (multilingual).

A cross-encoder jointly encodes (query, passage) and scores relevance directly, which is far
more precise than the bi-encoder's independent embeddings — at the cost of running once per
candidate. So we only rerank the fused candidate set (FUSED_TOP_K), not the whole corpus.
"""
from __future__ import annotations

import os
import threading
from functools import lru_cache

from .config import settings


class Reranker:
    def __init__(self, model_name: str | None = None):
        import torch
        from FlagEmbedding import FlagReranker  # heavy import, kept local

        # PyTorch defaults to a fraction of available cores on some platforms; this is a
        # CPU-bound cross-encoder, so give it every available core for inference. os.cpu_count()
        # reports the host's physical core count, not a container's cgroup quota, so in a
        # deployed container set RERANK_NUM_THREADS to match the service's actual vCPU limit --
        # otherwise torch oversubscribes threads against the real quota and inference gets
        # slower, not faster.
        torch.set_num_threads(settings.rerank_num_threads or os.cpu_count() or 1)

        self.model = FlagReranker(
            model_name or settings.rerank_model,
            use_fp16=False,
            devices="cpu",
        )
        self._lock = threading.Lock()

    def score(
        self,
        query: str,
        passages: list[str],
        batch_size: int = 16,
        max_length: int | None = None,
    ) -> list[float]:
        """Return a relevance score per passage in [0, 1] (higher = more relevant)."""
        if not passages:
            return []
        pairs = [[query, p] for p in passages]
        with self._lock:
            scores = self.model.compute_score(
                pairs,
                batch_size=batch_size,
                normalize=True,
                max_length=max_length or settings.rerank_max_length,
            )
        # FlagReranker returns a float for a single pair, list otherwise.
        if isinstance(scores, float):
            return [scores]
        return list(scores)


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()
