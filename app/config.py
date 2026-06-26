"""Central configuration, loaded from environment / .env."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Gemini (Developer API)
    gemini_api_key: str = ""
    llm_model: str = "gemini-3.5-flash"

    # Algolia
    algolia_app_id: str = ""
    algolia_api_key: str = ""
    algolia_index_name: str = "13fcbda9-b656-4559-911f-ce79be7ee6a2"

    # Postgres / pgvector
    database_url: str = ""
    pg_table: str = "slides"
    embed_dim: int = 1024

    # Models
    embed_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    # Retrieval knobs
    lexical_top_k: int = 50
    semantic_top_k: int = 50
    rrf_k: int = 60
    fused_top_k: int = 20
    rerank_top_n: int = 10
    rrf_lexical_weight: float = 1.0
    rrf_semantic_weight: float = 1.0
    # Cap on tokens-per-pair the cross-encoder sees. Real slide text averages ~230-260 tokens,
    # so 256 barely truncates typical slides while keeping the worst-case (multi-thousand-char
    # outliers) from dominating reranking latency. See app/reranker.py.
    rerank_max_length: int = 256
    # Explicit thread count for the reranker's torch.set_num_threads(). None = auto-detect via
    # os.cpu_count(), which is correct on bare metal but reports the *host's* core count inside
    # a container, not the cgroup quota actually granted -- set this to match whatever vCPU
    # limit is configured for the service (e.g. on Railway) to avoid thread oversubscription.
    rerank_num_threads: int | None = None


settings = Settings()
