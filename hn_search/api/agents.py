"""The surface for other agents: MCP tools, plain JSON routes, and llms.txt.

An outside agent — Claude, ChatGPT, anything with an MCP client — does its own
reasoning, so what it needs from this service is retrieval, not the chat. The
tools here are the pieces the RAG planner is built from, handed out one by one:
semantic search, "comments like this one", comment lookup by id, and corpus
freshness. `ask` is kept too, for a caller that would rather have the finished
cited answer and pay for the DeepSeek turn.

Three transports, one set of functions:

- `/mcp`: Streamable HTTP MCP, stateless, JSON responses. What Claude Code,
  claude.ai connectors, the Messages API and the OpenAI Responses API consume.
- `/api/find`, `/api/similar`, `/api/comments`, `/api/stats`: the same as GET
  routes, for an agent that only has a web fetch. FastAPI's own `/openapi.json`
  describes them.
- `/llms.txt`: the map. What this is, what a result looks like, where the
  above live.

Nothing here has auth. `/api/search` was already open, so this widens no door;
it only adds better-shaped ones.
"""

import asyncio
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings

from hn_search import search_backend
from hn_search.logging_config import get_logger
from hn_search.rag.pipeline import search_stream
from hn_search.rag.tools import semantic_search, similar_comments

logger = get_logger(__name__)

MCP_PATH = "/mcp"
MAX_K = 50
MAX_IDS = 50

ABOUT = """\
Semantic search over ~12 million Hacker News comments and stories from 2023 onward, \
ranked by exact cosine similarity of all-mpnet-base-v2 embeddings. A query is \
matched by meaning, not keywords, so describe the discussion you want rather than \
guessing its wording. The corpus grows daily; `stats` says how fresh it is."""

HONESTY = """\
- `distance` is cosine distance: lower is closer. Values under ~0.5 are usually on \
topic; above ~0.7 the match is loose. Say so when quoting a weak hit.
- Results are comments people wrote, not facts. Attribute claims to their author \
and link the id: https://news.ycombinator.com/item?id=<id>.
- Comments from before 2023 are not in the corpus. Absence is not evidence.
- Every result carries `timestamp`; date-bound the search rather than assuming \
recency."""

RESULT_SHAPE = """\
Each hit: {id, author, timestamp (ISO8601), type ("comment" | "story"), text, \
distance, url}. `text` is the comment body with HTML stripped."""


def _url(hn_id: str) -> str:
    return f"https://news.ycombinator.com/item?id={hn_id}"


def _with_urls(hits: list[dict]) -> list[dict]:
    return [{**h, "url": _url(h["id"])} for h in hits]


def _clamp_k(k: int) -> int:
    if k < 1:
        raise ValueError("k must be at least 1")
    return min(k, MAX_K)


def find(
    query: str,
    k: int = 10,
    time_after: str | None = None,
    time_before: str | None = None,
) -> list[dict]:
    """Semantic search, through the same tool (and Redis cache) the planner uses."""
    if not query.strip():
        raise ValueError("query must not be empty")
    hits = semantic_search.invoke(
        {
            "query": query.strip(),
            "k": _clamp_k(k),
            "time_after": time_after or None,
            "time_before": time_before or None,
        }
    )
    return _with_urls(hits)


def similar(hn_id: str, k: int = 10) -> list[dict]:
    if not hn_id.strip().isdigit():
        raise ValueError("hn_id must be a numeric Hacker News item id")
    hits = similar_comments.invoke({"hn_id": hn_id.strip(), "k": _clamp_k(k)})
    return _with_urls(hits)


def comments(hn_ids: list[str]) -> list[dict]:
    """Text and metadata for specific ids. Ids not in the corpus are omitted."""
    ids = [i.strip() for i in hn_ids if i.strip()]
    if not ids:
        raise ValueError("hn_ids must not be empty")
    if len(ids) > MAX_IDS:
        raise ValueError(f"at most {MAX_IDS} ids per call")
    docs = search_backend.get_docs(ids)
    return [
        {
            "id": d["id"],
            "author": d["author"],
            "timestamp": d["timestamp"],
            "type": d["type"],
            "text": d["clean_text"],
            "parent_id": d.get("parent_id"),
            "url": _url(d["id"]),
        }
        for i in ids
        if (d := docs.get(i))
    ]


def stats() -> dict:
    return search_backend.stats()


def ask(question: str) -> dict:
    """The whole pipeline: planner, searches, fusion, DeepSeek. Costs a model call."""
    if not question.strip():
        raise ValueError("question must not be empty")
    answer, sources, refused = "", [], False
    for event in search_stream(question.strip()):
        if event["type"] == "sources":
            sources = event["sources"]
        elif event["type"] == "answer":
            answer = event["text"]
            refused = bool(event.get("refused"))
        elif event["type"] == "error":
            raise RuntimeError(event["message"])
    return {"answer": answer, "sources": sources, "refused": refused}


# --- MCP ---------------------------------------------------------------------

server = MCPServer(
    "hn-search",
    title="Hacker News semantic search",
    instructions=(
        f"{ABOUT}\n\nUse search for a topic, similar for 'more like this "
        "comment', comments to read specific ids in full. "
        + RESULT_SHAPE
        + "\n\n"
        + HONESTY
    ),
)


def _explain(fn, exc: Exception) -> str:
    """A caller-facing message for a failed call. httpx's own text names the
    backend's URL, which an anonymous caller has no business seeing, so only
    the status survives; everything else is reduced to its type. The full
    detail goes to the log."""
    logger.exception(f"{fn.__name__} failed")
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        hint = " (no such id in the corpus)" if status == 404 else ""
        return f"{fn.__name__} failed: the search service returned {status}{hint}"
    if isinstance(exc, httpx.HTTPError):
        return f"{fn.__name__} failed: the search service is unreachable"
    return f"{fn.__name__} failed: {type(exc).__name__}"


async def _tool(fn, *args):
    """Run a blocking call off the event loop, turning a bad argument or a
    backend failure into a ToolError the caller can read."""
    try:
        return await asyncio.to_thread(fn, *args)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    except Exception as exc:
        raise ToolError(_explain(fn, exc)) from exc


@server.tool(name="search")
async def search_tool(
    query: str,
    k: int = 10,
    time_after: str | None = None,
    time_before: str | None = None,
) -> list[dict[str, Any]]:
    """Search Hacker News comments and stories by meaning.

    Describe the discussion you want in a sentence; keywords work less well than
    a paraphrase of the kind of comment you expect. Returns up to k hits (max 50),
    closest first. time_after / time_before are ISO8601 dates (YYYY-MM-DD) that
    bound the results; the corpus starts in 2023.

    Args:
        query: What you are looking for, in natural language.
        k: How many hits, 1-50.
        time_after: Only comments on or after this date.
        time_before: Only comments before this date.
    """
    return await _tool(find, query, k, time_after, time_before)


@server.tool(name="similar")
async def similar_tool(hn_id: str, k: int = 10) -> list[dict[str, Any]]:
    """Comments similar to one specific Hacker News item, given its numeric id
    (the number in a news.ycombinator.com/item?id=... link). Reuses that item's
    own stored embedding, so no description is needed. The item itself is
    excluded. Fails if the id is not in the corpus.

    Args:
        hn_id: The numeric item id.
        k: How many hits, 1-50.
    """
    return await _tool(similar, hn_id, k)


@server.tool(name="comments")
async def comments_tool(hn_ids: list[str]) -> list[dict[str, Any]]:
    """Full text and metadata for specific Hacker News items by numeric id, up to
    50 per call. Use it to read a parent comment a hit replies to (its parent_id),
    or to expand a result you already have. Ids not in the corpus are omitted.

    Args:
        hn_ids: Numeric item ids.
    """
    return await _tool(comments, hn_ids)


@server.tool(name="stats")
async def stats_tool() -> dict[str, Any]:
    """Corpus freshness: how many items are indexed, the highest item id, and the
    timestamp of the newest comment. Check it before claiming something is not
    on Hacker News."""
    return await _tool(stats)


@server.tool(name="ask")
async def ask_tool(question: str) -> dict[str, Any]:
    """Ask the hosted RAG agent and get its finished, cited answer plus the
    sources it used. Slower and lossier than searching yourself; use it when you
    want a second opinion or do not want to run the searches. A question that is
    not about what Hacker News discusses is refused: `refused` is true and
    `answer` says so.

    Args:
        question: A question about what Hacker News thinks.
    """
    return await _tool(ask, question)


# A one-route Starlette app the FastAPI app registers at MCP_PATH; its session
# manager runs inside the outer app's lifespan, see `app.py`.
#
# Stateless: no session ids, every call stands alone, which is what a read-only
# server behind a load balancer wants. JSON responses rather than SSE for the
# same reason, and so `curl` shows something readable.
#
# Behind Caddy or Railway the Host header is the public name, so the SDK's
# localhost-only rebinding guard would reject every real request. There is no
# session or credential here for a rebinding attack to steal.
mcp_app = server.streamable_http_app(
    streamable_http_path=MCP_PATH,
    json_response=True,
    stateless_http=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


# --- Plain HTTP --------------------------------------------------------------

router = APIRouter()


async def _route(fn, *args):
    try:
        return await asyncio.to_thread(fn, *args)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        status = 404 if exc.response.status_code == 404 else 502
        raise HTTPException(status_code=status, detail=_explain(fn, exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_explain(fn, exc)) from exc


@router.get("/api/find")
async def find_route(
    q: str = "",
    k: int = 10,
    time_after: str | None = None,
    time_before: str | None = None,
) -> list[dict]:
    """Semantic search as JSON. The twin of the MCP `search` tool; `/api/search`
    is the RAG pipeline as server-sent events."""
    return await _route(find, q, k, time_after, time_before)


@router.get("/api/similar")
async def similar_route(id: str = "", k: int = 10) -> list[dict]:
    """Items similar to one numeric id. JSON twin of the MCP `similar` tool."""
    return await _route(similar, id, k)


@router.get("/api/comments")
async def comments_route(ids: list[str] = Query(default=[])) -> list[dict]:
    """Full text for numeric ids: `?ids=1&ids=2` or `?ids=1,2`. JSON twin of the
    MCP `comments` tool."""
    flat = [part for chunk in ids for part in chunk.split(",")]
    return await _route(comments, flat)


def llms_txt(base: str) -> str:
    base = base.rstrip("/")
    return f"""\
# HN Search: semantic search over Hacker News

> {ABOUT}

Free, read-only, no auth. Query it directly rather than scraping the chat UI.

## MCP (preferred)

Streamable HTTP endpoint: {base}{MCP_PATH}
Tools: search(query, k, time_after, time_before), similar(hn_id, k), \
comments(hn_ids), stats(), ask(question).

- Claude Code: `claude mcp add --transport http hn-search {base}{MCP_PATH}`
- Claude API: mcp_servers=[{{"type": "url", "url": "{base}{MCP_PATH}", "name": "hn-search"}}]
- OpenAI Responses API: tools=[{{"type": "mcp", "server_label": "hn-search", "server_url": "{base}{MCP_PATH}", "require_approval": "never"}}]

## Plain HTTP

- {base}/api/find?q=why+do+people+leave+google&k=10&time_after=2025-01-01 — semantic search (JSON)
- {base}/api/similar?id=43000000&k=10 — items like one id (JSON)
- {base}/api/comments?ids=43000000,43000001 — full text by id (JSON)
- {base}/api/stats — corpus size and freshness (JSON)
- {base}/api/search?q=... — the hosted RAG agent, as server-sent events
- {base}/openapi.json — the OpenAPI description of all of the above

## Results

{RESULT_SHAPE}

## Before you quote a comment

{HONESTY}
"""


@router.get("/llms.txt", response_class=PlainTextResponse)
def llms(request: Request) -> str:
    # Built from the request so staging and prod each name themselves. Uvicorn
    # only trusts X-Forwarded-Proto from loopback, and neither Caddy nor
    # Railway's edge is that, so the scheme is fixed up here: anything that is
    # not a dev loopback is behind TLS.
    base = request.base_url
    if base.hostname not in ("localhost", "127.0.0.1", "::1"):
        base = base.replace(scheme="https")
    return llms_txt(str(base))
