"""FastAPI service exposing the chat-with-your-docs pipeline."""
from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from .pipeline import get_pipeline
from .schema import ChatRequest, ChatResponse

app = FastAPI(title="chat-with-your-docs", version="1.0.0")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.on_event("startup")
def _warm() -> None:
    # Load models + open connections at boot so the first request isn't slow.
    get_pipeline()


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    return get_pipeline().run(req)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Same pipeline as /chat, but Server-Sent Events: a `qu` event as soon as query
    understanding finishes, a `retrieval` event (citations + retrieval stats) once dual
    retrieval/fusion/rerank finish, then one `delta` event per generated token, then `done`."""

    def events():
        for event in get_pipeline().run_stream(req):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
