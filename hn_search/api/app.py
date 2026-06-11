"""FastAPI app: SSE search endpoint, recent queries, and the built React UI."""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from sse_starlette import EventSourceResponse

from .search import job_manager, sse_search

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


@app.get("/api/health")
def health():
    return {"status": "ok"}


_static_dir = Path(
    os.environ.get("STATIC_DIR", Path(__file__).parents[2] / "frontend" / "dist")
)
if _static_dir.is_dir():
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
