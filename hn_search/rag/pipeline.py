"""Streaming search pipeline: drives the compiled agentic graph as a flat event
generator, translating its per-node updates into SSE-ready events.

Event types:
    {"type": "progress", "step", "label", "status": "start"|"done", "ms", "hit"}
    {"type": "sources", "sources": [{id, author, timestamp, type, text, url, distance}]}
    {"type": "token", "text"}    -- answer delta (currently emitted as one chunk)
    {"type": "answer", "text", "refused"}   -- full answer, always emitted last;
                                  refused=True when the scope filter stopped the
                                  query and `text` is the canned refusal
    {"type": "error", "message"}
"""

import time
from typing import Iterator, Optional

from hn_search.logging_config import get_logger

from .agent import create_agent_workflow, initial_state

logger = get_logger(__name__)

_NODE_LABELS = {
    "guard": "Scope check",
    "seed": "Searching your question",
    "agent": "Planning search",
    "tools": "Searching",
    "gather_sources": "Picking sources",
    "synthesize_answer": "Asking DeepSeek",
}
_LABEL_MAX_CHARS = 120


def _progress(
    step: str, status: str, ms: Optional[int] = None, label: Optional[str] = None
) -> dict:
    return {
        "type": "progress",
        "step": step,
        "label": label or _NODE_LABELS.get(step, step),
        "status": status,
        "ms": ms,
        "hit": None,
    }


def _window(args: dict) -> str:
    after, before = args.get("time_after"), args.get("time_before")
    return f" · {after or '…'} → {before or 'now'}" if after or before else ""


def _describe_call(call: dict) -> str:
    args = call.get("args", {})
    name = call.get("name")
    if name == "semantic_search":
        return f"searching “{args.get('query', '')}”{_window(args)}"
    if name == "similar_comments":
        return f"finding comments like {args.get('hn_id')}"
    if name == "keyword_stats":
        return f"counting “{args.get('term', '')}”{_window(args)}"
    if name == "get_comments":
        return f"reading {len(args.get('hn_ids') or [])} comments"
    return str(name)


def _describe_round(tool_calls: list[dict]) -> str:
    """One round of the planner's tool calls, as a progress label: what it is
    looking for is the interesting part of a run, so the log says it."""
    label = "; ".join(_describe_call(c) for c in tool_calls)
    label = label[:1].upper() + label[1:]
    if len(label) > _LABEL_MAX_CHARS:
        label = label[: _LABEL_MAX_CHARS - 1] + "…"
    return label


def _next_node(node_name: str, delta: dict) -> Optional[tuple[str, Optional[str]]]:
    """Predict the next node (and its label) from the graph's topology, so its
    "start" event can be emitted the instant the current node finishes —
    otherwise stream_mode="updates" only ever tells us about a node *after* it
    completes, leaving the client with no spinner during the long
    synthesize_answer call."""
    if node_name == "guard":
        return ("seed", None) if delta.get("on_topic") else None
    if node_name == "seed":
        return "agent", None
    if node_name == "agent":
        messages = delta.get("messages") or []
        calls = getattr(messages[-1], "tool_calls", None) if messages else None
        return ("tools", _describe_round(calls)) if calls else ("gather_sources", None)
    if node_name == "tools":
        return "agent", "Reading results"
    if node_name == "gather_sources":
        return "synthesize_answer", None
    return None


def search_stream(query: str) -> Iterator[dict]:
    """Drives the compiled tool-calling graph, translating its per-node updates
    into typed SSE events."""
    workflow = create_agent_workflow()
    # The UI pairs a step's "done" with its "start" and shows the done event's
    # label, so a step keeps the label it started with.
    labels: dict[str, Optional[str]] = {}

    try:
        yield _progress("guard", "start")
        t0 = time.perf_counter()
        for update in workflow.stream(initial_state(query), stream_mode="updates"):
            for node_name, delta in update.items():
                ms = round((time.perf_counter() - t0) * 1000)
                label = labels.pop(node_name, None)
                logger.info(
                    f"⏱️ {label or _NODE_LABELS.get(node_name, node_name)}: {ms}ms"
                )
                yield _progress(node_name, "done", ms=ms, label=label)

                if node_name == "guard" and not delta.get("on_topic"):
                    # Refused: the guard wrote the answer itself and nothing
                    # else will run, so this is the whole result.
                    yield {"type": "sources", "sources": []}
                    yield {"type": "answer", "text": delta["answer"], "refused": True}
                elif node_name == "gather_sources":
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
                    yield {"type": "answer", "text": answer, "refused": False}

                upcoming = _next_node(node_name, delta)
                if upcoming:
                    next_node, labels[next_node] = upcoming
                    yield _progress(next_node, "start", label=labels[next_node])
                t0 = time.perf_counter()
    except Exception as e:
        logger.exception(f"Agentic pipeline error: {e}")
        yield {"type": "error", "message": str(e)}
