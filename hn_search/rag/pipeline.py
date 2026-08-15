"""Streaming search pipeline: drives the compiled agentic graph as a flat event
generator, translating its per-node updates into SSE-ready events.

Event types:
    {"type": "progress", "step", "label", "status": "start"|"done", "ms", "hit"}
    {"type": "sources", "sources": [{id, author, timestamp, type, text, url, distance}]}
    {"type": "token", "text"}    -- answer delta (currently emitted as one chunk)
    {"type": "answer", "text"}   -- full answer, always emitted last
    {"type": "error", "message"}
"""

import time
from typing import Iterator, Optional

from hn_search.logging_config import get_logger

from .agent import AgentState, create_agent_workflow

logger = get_logger(__name__)

_NODE_LABELS = {
    "agent": "Planning search",
    "tools": "Searching (agent-requested)",
    "gather_sources": "Searching (baseline) + merging",
    "synthesize_answer": "Asking DeepSeek",
}


def _progress(step: str, status: str, ms: Optional[int] = None) -> dict:
    return {
        "type": "progress",
        "step": step,
        "label": _NODE_LABELS.get(step, step),
        "status": status,
        "ms": ms,
        "hit": None,
    }


def _next_node(node_name: str, delta: dict) -> Optional[str]:
    """Predict the next node from the graph's static topology, so its "start"
    event can be emitted the instant the current node finishes — otherwise
    stream_mode="updates" only ever tells us about a node *after* it completes,
    leaving the client with no spinner during the long synthesize_answer call."""
    if node_name == "agent":
        messages = delta.get("messages") or []
        last = messages[-1] if messages else None
        return "tools" if getattr(last, "tool_calls", None) else "gather_sources"
    if node_name == "tools":
        return "gather_sources"
    if node_name == "gather_sources":
        return "synthesize_answer"
    return None


def search_stream(query: str) -> Iterator[dict]:
    """Drives the compiled tool-calling graph, translating its per-node updates
    into typed SSE events."""
    workflow = create_agent_workflow()
    initial_state = AgentState(
        messages=[],
        query=query,
        tool_calls=[],
        sources=[],
        parent_texts={},
        answer="",
    )

    try:
        yield _progress("agent", "start")
        t0 = time.perf_counter()
        for update in workflow.stream(initial_state, stream_mode="updates"):
            for node_name, delta in update.items():
                ms = round((time.perf_counter() - t0) * 1000)
                logger.info(f"⏱️ {_NODE_LABELS.get(node_name, node_name)}: {ms}ms")
                yield _progress(node_name, "done", ms=ms)

                if node_name == "gather_sources":
                    sources = delta.get("sources", [])
                    logger.info(f"✅ Found {len(sources)} relevant comments/articles")
                    yield {
                        "type": "sources",
                        "sources": [
                            {
                                **s,
                                "url": f"https://news.ycombinator.com/item?id={s['id']}",
                            }
                            for s in sources
                        ],
                    }
                elif node_name == "synthesize_answer":
                    answer = delta.get("answer", "")
                    yield {"type": "token", "text": answer}
                    yield {"type": "answer", "text": answer}

                next_node = _next_node(node_name, delta)
                if next_node:
                    yield _progress(next_node, "start")
                t0 = time.perf_counter()
    except Exception as e:
        logger.exception(f"Agentic pipeline error: {e}")
        yield {"type": "error", "message": str(e)}
