"""FastAPI app: SSE search endpoint, recent queries, and the built React UI."""

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from sse_starlette import EventSourceResponse

from .. import search_backend
from ..cache_config import redis_client
from ..logging_config import get_logger
from . import agents, auth
from .search import job_manager, sse_search

STATS_CACHE_KEY = "hn:stats"
STATS_CACHE_TTL = 300  # the corpus only grows once a day


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # The MCP transport keeps its own task group; it lives exactly as long as
    # the app does. Nothing else needs starting: the encoder, the graph and the
    # search client are all lazy singletons.
    async with agents.server.session_manager.run():
        yield


logger = get_logger(__name__)

app = FastAPI(title="HN RAG Search", lifespan=lifespan)


@app.get("/api/search")
def search(q: str = ""):
    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty")
    return EventSourceResponse(sse_search(query))


@app.get("/api/recent")
def recent(limit: int = 25):
    return {"queries": job_manager.get_recent_queries(min(limit, 100))}


@app.delete("/api/recent", status_code=204)
def delete_recent(q: str, request: Request) -> Response:
    """Forget a query — its row in the recent list and its cached answer.

    Admins only (`auth.py`): the list is shared, so anyone could otherwise
    delete anyone's search. Idempotent, and deliberately never 404s: the list
    is a snapshot, so the row you clicked may already have been trimmed or
    deleted from another tab. Reporting that as a failure would leave a row
    nobody can get rid of.
    """
    who = auth.require_admin(request)
    logger.info(f"{who} deleted recent query: {q!r}")
    job_manager.delete_recent_query(q)
    return Response(status_code=204)


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


# The machine-facing surface: /mcp, /api/find, /api/similar, /api/comments and
# /llms.txt. Registered before the static mount below, which would otherwise
# swallow /llms.txt and /mcp as files that do not exist.
app.include_router(agents.router)
app.include_router(auth.router)
app.add_route(agents.MCP_PATH, agents.mcp_app, methods=["GET", "POST", "DELETE"])


_static_dir = Path(
    os.environ.get("STATIC_DIR", Path(__file__).parents[2] / "frontend" / "dist")
)
if _static_dir.is_dir():
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
