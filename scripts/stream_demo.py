"""Consume /chat/stream and print the answer growing live, like a chat UI.

Postman shows SSE as a list of discrete events rather than concatenated text, so this is the
quickest way to actually see the token-by-token streaming effect. Talks to a running server
over HTTP rather than importing the pipeline directly, so it doesn't load its own copy of the
embedding/reranker models.

Usage:
    python scripts/stream_demo.py "What did we propose to Airbus on supply chain?"
"""
from __future__ import annotations

import json
import sys

import httpx

SERVER_URL = "http://localhost:8000/chat/stream"


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "Show me credentials in aerospace & defense"

    with httpx.Client(timeout=60.0) as client, client.stream(
        "POST", SERVER_URL, json={"query": query}
    ) as resp:
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[len("data: "):])

            if event["type"] == "meta":
                print(
                    f"=== {len(event['citations'])} citations | "
                    f"lexical={event['n_lexical']} semantic={event['n_semantic']} "
                    f"fused={event['n_after_fusion']} reranked={event['n_after_rerank']} ===\n"
                )
            elif event["type"] == "delta":
                print(event["text"], end="", flush=True)
            elif event["type"] == "done":
                print("\n")


if __name__ == "__main__":
    main()
