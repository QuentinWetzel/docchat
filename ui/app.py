"""Lightweight Gradio chat UI for docchat.

Runs as its own container with no ML dependencies (no torch/transformers): it just talks to the
FastAPI backend's /chat/stream SSE endpoint over HTTP, the same way scripts/stream_demo.py does,
and renders the qu/retrieval/delta/done events as they arrive -- qu and retrieval land as separate
events seconds apart (rerank isn't free), so each updates the Steps panel the moment it's ready
rather than waiting for both. Each turn is stateless on the backend side (no conversation memory),
so only the latest message is sent — history is for display only.
"""
from __future__ import annotations

import json
import os

import gradio as gr
import httpx

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")


def _format_filters(filters: dict) -> str:
    active = {k: v for k, v in filters.items() if v}
    return ", ".join(f"{k}={v}" for k, v in active.items()) or "none"


def _format_citations(citations: list[dict]) -> str:
    if not citations:
        return ""
    lines = []
    for c in citations:
        label = f"{c.get('file_name') or 'slide'} (slide {c.get('slide_number')})"
        lines.append(f"- [{label}]({c['web_url']})" if c.get("web_url") else f"- {label}")
    return "\n\n---\n**Sources**\n" + "\n".join(lines)


def respond(message: str, history: list):
    if not message.strip():
        yield "Please enter a question."
        return

    qu_body = retrieval_body = "_pending…_"
    answer = ""
    citations: list[dict] = []
    summary_bits: list[str] = []
    expanded = True

    def render() -> str:
        summary = "🔎 Steps" + (" · " + " · ".join(summary_bits) if summary_bits else "")
        block = (
            f"<details{' open' if expanded else ''}><summary>{summary}</summary>\n\n"
            f"**1. Query understanding**\n{qu_body}\n\n"
            f"**2. Retrieval**\n{retrieval_body}\n\n"
            "</details>\n\n"
        )
        return block + (answer or "_thinking…_") + _format_citations(citations)

    yield render()

    try:
        with httpx.Client(timeout=120.0) as client, client.stream(
            "POST", f"{BACKEND_URL}/chat/stream", json={"query": message}
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[len("data: "):])
                etype = event["type"]

                if etype == "qu":
                    t = event["timings"]
                    qu_body = (
                        f"- intent: `{event['intent_type']}`\n"
                        f"- lexical query: \"{event['lexical_query'] or '—'}\"\n"
                        f"- semantic query: \"{event['semantic_query'] or '—'}\"\n"
                        f"- filters: {_format_filters(event['used_filters'])}"
                    )
                    summary_bits = [
                        f"query understanding {t['query_understanding_ms']:.0f} ms",
                        "retrieving…",
                    ]
                    yield render()

                elif etype == "retrieval":
                    t = event["timings"]
                    retrieval_body = (
                        f"- lexical hits: {event['n_lexical']}\n"
                        f"- semantic hits: {event['n_semantic']}\n"
                        f"- after fusion: {event['n_after_fusion']}\n"
                        f"- after rerank: {event['n_after_rerank']}"
                    )
                    citations = event.get("citations", [])
                    summary_bits = summary_bits[:1] + [
                        f"retrieval {t['retrieval_ms']:.0f} ms",
                        "answering…",
                    ]
                    yield render()

                elif etype == "delta":
                    answer += event["text"]
                    yield render()

                elif etype == "done":
                    t = event["timings"]
                    gen_bit = f"answer {t['generation_ms']:.0f} ms"
                    if t.get("first_token_ms") is not None:
                        gen_bit += f" (first token {t['first_token_ms']:.0f} ms)"
                    summary_bits = summary_bits[:2] + [gen_bit, f"total {t['total_ms']:.0f} ms"]
                    expanded = False
                    yield render()
    except httpx.HTTPError as e:
        yield render() + f"\n\n*Error talking to backend at {BACKEND_URL}: {e}*"


demo = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="docchat",
    description=(
        "RAG over the slide corpus — Algolia + pgvector hybrid retrieval, cross-encoder reranked, "
        "answered by Gemini. Expand **Steps** on a reply to see query understanding and retrieval "
        "stats with latencies."
    ),
)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
