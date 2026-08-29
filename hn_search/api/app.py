"""FastAPI app: SSE search endpoint, recent queries, and the built React UI."""

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from sse_starlette import EventSourceResponse

from .. import search_backend
from ..cache_config import redis_client
from .search import job_manager, sse_search

STATS_CACHE_KEY = "hn:stats"
STATS_CACHE_TTL = 300  # the corpus only grows once a day

app = FastAPI(title="HN RAG Search")


@app.get("/api/search")
def search(q: str = ""):
    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty")
    return EventSourceResponse(sse_search(query))


@app.get("/api/recent")
def recent(limit: int = 25):
    return {"queries": job_manager.get_recent_queries(min(limit, 100))}


@app.get("/api/stats")
def stats():
    """Corpus freshness (doc count + newest comment timestamp) for the UI."""
    if redis_client is not None:
        cached = redis_client.get(STATS_CACHE_KEY)
        if isinstance(cached, bytes):
            return json.loads(cached)
    try:
        body = search_backend.stats()
    except Exception as e:  # service down / old binary without /stats
        raise HTTPException(status_code=503, detail=f"stats unavailable: {e}")
    if redis_client is not None:
        redis_client.setex(STATS_CACHE_KEY, STATS_CACHE_TTL, json.dumps(body))
    return body


@app.get("/api/health")
def health():
    return {"status": "ok"}


_static_dir = Path(
    os.environ.get("STATIC_DIR", Path(__file__).parents[2] / "frontend" / "dist")
)
if _static_dir.is_dir():
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
