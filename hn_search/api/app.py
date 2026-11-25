"""FastAPI app: SSE search endpoint, recent queries, and the built React UI."""

import json
import os
from urllib.parse import quote
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette import EventSourceResponse

from .. import permalinks, search_backend
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


@app.get("/api/answers/{share_id}")
def saved_answer(share_id: str):
    """A saved answer (`permalinks.py`): query, answer, sources, created_at."""
    snapshot = permalinks.load(share_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="No such answer")
    return snapshot


# A saved answer never changes while it exists; a day keeps a crawler's or a
# chat app's copy from outliving its expiry or deletion by much.
_IMAGE_CACHE = "public, max-age=86400"


def _base_url(request: Request) -> str:
    base = request.base_url
    if auth.is_secure(request):
        base = base.replace(scheme="https")
    return str(base).rstrip("/")


def _page(tags: str) -> HTMLResponse:
    index = permalinks.index_html(_static_dir)
    if index is None:  # dev without a frontend build
        raise HTTPException(status_code=404, detail="Frontend not built")
    return HTMLResponse(permalinks.render_page(index, tags))


def _sources_line(snapshot: dict) -> str:
    n = len(snapshot.get("sources") or [])
    return f"{n} Hacker News comment{'s' if n != 1 else ''} cited"


@app.get("/a/{share_id}", response_class=HTMLResponse)
def answer_page(share_id: str, request: Request):
    """The SPA, with this answer's link-preview tags in its head: crawlers
    run no JavaScript, so the tags have to be in the HTML they fetch."""
    snapshot = permalinks.load(share_id)
    base = _base_url(request)
    if snapshot is None:
        # Still the app, which tells a visitor the answer is gone; the 404
        # is for crawlers.
        page = _page(_site_tags(base, f"{base}/a/{share_id}"))
        page.status_code = 404
        return page
    tags = permalinks.meta_tags(
        title=f"{snapshot['query']} — {permalinks.SITE_NAME}",
        description=permalinks.summary(snapshot["answer"]) or _sources_line(snapshot),
        url=f"{base}/a/{share_id}",
        image=f"{base}/a/{share_id}/og.png",
    )
    return _page(tags)


@app.get("/a/{share_id}/og.png")
def answer_image(share_id: str):
    snapshot = permalinks.load(share_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="No such answer")
    png = permalinks.og_image(
        snapshot["query"],
        permalinks.summary(snapshot["answer"], 260),
        _sources_line(snapshot),
    )
    return Response(
        png, media_type="image/png", headers={"Cache-Control": _IMAGE_CACHE}
    )


@app.get("/og.png")
def site_image():
    png = permalinks.og_image(
        "Ask Hacker News anything", permalinks.SITE_DESCRIPTION, "hn.fiodorov.es"
    )
    return Response(
        png, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"}
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request, q: str = ""):
    """The SPA with site-wide preview tags. A `?q=` link is titled with its
    question but keeps the generic image: that text has not been through the
    guard, so it is not drawn onto an image served from this domain."""
    base = _base_url(request)
    query = q.strip()
    url = f"{base}/" + (f"?q={quote(query)}" if query else "")
    return _page(_site_tags(base, url, query))


def _site_tags(base: str, url: str, query: str = "") -> str:
    return permalinks.meta_tags(
        title=f"{query} — {permalinks.SITE_NAME}" if query else permalinks.SITE_NAME,
        description=permalinks.SITE_DESCRIPTION,
        url=url,
        image=f"{base}/og.png",
    )


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
