"""Smoke-test the full pipeline against live Algolia + pgvector + Opus 4.8.

Usage:
    python scripts/03_smoke_test.py "What did we propose to Airbus on supply chain?"
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.pipeline import get_pipeline  # noqa: E402
from app.schema import ChatRequest      # noqa: E402


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "Show me credentials in aerospace & defense"
    resp = get_pipeline().run(ChatRequest(query=query))

    print("\n=== QUERY UNDERSTANDING ===")
    print(f"intent_type={resp.intent_type}")
    print(f"lexical_query={resp.lexical_query!r}")
    print(f"semantic_query={resp.semantic_query!r}")
    print(resp.used_filters.model_dump_json(indent=2, exclude_defaults=True))
    print("\n=== RETRIEVAL COUNTS ===")
    print(f"lexical={resp.n_lexical} semantic={resp.n_semantic} "
          f"fused={resp.n_after_fusion} reranked={resp.n_after_rerank}")
    print("\n=== ANSWER ===")
    print(resp.answer)
    print("\n=== CITATIONS ===")
    for i, c in enumerate(resp.citations, start=1):
        print(f"[{i}] {c.file_name} slide {c.slide_number} — {c.source}")
        print(f"    {c.web_url}")


if __name__ == "__main__":
    main()
