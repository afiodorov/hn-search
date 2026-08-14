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
from typing import Iterator

from hn_search.logging_config import get_logger

from .agent import create_agent_workflow

logger = get_logger(__name__)

_NODE_LABELS = {
    "agent": "Planning search",
    "tools": "Searching (agent-requested)",
    "gather_sources": "Searching (baseline) + merging",
    "synthesize_answer": "Asking DeepSeek",
}


def search_stream(query: str) -> Iterator[dict]:
    """Drives the compiled tool-calling graph, translating its per-node updates
    into typed SSE events."""
    workflow = create_agent_workflow()
    initial_state = {"messages": [], "query": query, "sources": [], "answer": ""}

    try:
        t0 = time.perf_counter()
        for update in workflow.stream(initial_state, stream_mode="updates"):
            for node_name, delta in update.items():
                ms = round((time.perf_counter() - t0) * 1000)
                t0 = time.perf_counter()
                label = _NODE_LABELS.get(node_name, node_name)
                logger.info(f"⏱️ {label}: {ms}ms")
                yield {
                    "type": "progress",
                    "step": node_name,
                    "label": label,
                    "status": "done",
                    "ms": ms,
                    "hit": None,
                }

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
    except Exception as e:
        logger.exception(f"Agentic pipeline error: {e}")
        yield {"type": "error", "message": str(e)}
