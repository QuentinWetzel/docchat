"""Bi-encoder embeddings using BAAI/bge-m3 (multilingual, 1024-dim).

bge-m3 covers FR/EN/DE/other, which matches the corpus language mix. Dense vectors only here
(the lexical leg is Algolia, so we don't need bge-m3's sparse output).

For queries, bge-m3 does NOT require a special instruction prefix (unlike bge-v1.5 English models),
so query and passage are embedded the same way.
"""
from __future__ import annotations

import threading
from functools import lru_cache

from .config import settings


class Embedder:
    def __init__(self, model_name: str | None = None):
        from FlagEmbedding import BGEM3FlagModel  # heavy import, kept local

        # Force CPU: on unified-memory Apple Silicon, FlagEmbedding auto-selects MPS,
        # whose allocator doesn't release memory back to the OS between batches, driving
        # the whole machine into swap within a few hundred records. fp16 has no native
        # CPU kernels either, so it's slower and no smaller here than fp32 - keep fp32.
        self.model = BGEM3FlagModel(
            model_name or settings.embed_model,
            use_fp16=False,
            devices="cpu",
        )
        self._lock = threading.Lock()

    def encode(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        """Return dense vectors (1024-dim) for a list of texts."""
        if not texts:
            return []
        with self._lock:
            out = self.model.encode(
                texts,
                batch_size=batch_size,
                max_length=1024,
            )["dense_vecs"]
        return [vec.tolist() for vec in out]

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Process-wide singleton; the model is large, load it once."""
    return Embedder()
