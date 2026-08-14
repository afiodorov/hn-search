"""Agentic graph: a tool-calling planner, a guaranteed baseline search, and a
dedicated synthesis node.

The planner LLM may call semantic_search (optionally date-bounded) or
similar_comments (by HN id/link) zero or more times if it thinks that helps
retrieval — but we don't rely on it following instructions to *also* search the
user's literal query; an LLM can't be fully trusted to follow a "search verbatim"
instruction (verified empirically: an earlier prompt asked for verbatim-only
search and the model rewrote it anyway, drifting retrieval away from legacy
behavior — see eval_judge history). So a plain verbatim semantic_search on the
user's exact question always runs too, deterministically, outside the LLM's
control. gather_sources fuses all result lists by rank (RRF) and caps the pool,
and synthesize_answer drafts the final cited answer from the combined set.

Still single-hop for the planner (it can request 1+ tool calls in one turn, which
ToolNode runs together, but the graph doesn't loop back to the planner after) — a
real multi-turn loop is a later step, for when a tool's result needs to inform a
*subsequent* tool choice (e.g. thread context once parent/child data exists).
"""

import json
from datetime import datetime, timezone
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from hn_search.cache_config import cache_answer, get_cached_answer
from hn_search.logging_config import get_logger

from .nodes import build_context, build_prompt, make_llm
from .tools import semantic_search, similar_comments

logger = get_logger(__name__)

_TOOLS = [semantic_search, similar_comments]
_TOOL_NAMES = {t.name for t in _TOOLS}
_DEFAULT_K = 10
# Cap on the merged (baseline + agent-added) source pool fed to synthesis, so
# context size — and DeepSeek latency/cost — doesn't scale with how many extra
# searches the agent decides to make.
_MAX_SOURCES = 12
# Standard Reciprocal Rank Fusion constant (Cormack et al. 2009); dampens the
# influence of any single list's exact rank so cross-list consensus matters more
# than any one query embedding's raw distance scale.
_RRF_K = 60


def _reciprocal_rank_fusion(
    result_lists: list[list[dict]], k: int = _RRF_K, limit: int = _MAX_SOURCES
) -> list[dict]:
    """Fuse several ranked result lists (e.g. baseline search + each agent-added
    search) by rank rather than raw distance — raw cosine distances from
    different query embeddings aren't on a comparable scale, but rank position
    within a list always is."""
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}
    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            docs.setdefault(doc_id, doc)
    ranked_ids = sorted(scores, key=lambda i: scores[i], reverse=True)
    return [docs[i] for i in ranked_ids[:limit]]


_SYSTEM_PROMPT = (
    "You are the retrieval planner for a Hacker News search assistant. Today's "
    "date is {today}. A plain semantic search on the user's exact question always "
    "runs automatically, regardless of what you do, so you do not need to (and "
    "should not bother to) search the verbatim question yourself. Your job is to "
    "decide whether ANY ADDITIONAL tool calls would surface better results on top "
    "of that baseline:\n\n"
    "- If the question contains a news.ycombinator.com/item?id=... link or a bare "
    "HN comment id, and the user wants comments *like* or *related to* it, call "
    "similar_comments with that id instead of (or in addition to) semantic_search "
    "— it reuses the comment's own embedding directly, no need to describe its "
    "content in words.\n"
    "- If the question mentions a time period ('last 6 months', 'since 2023', "
    "'in 2022'), call semantic_search with time_after/time_before set to the "
    "actual ISO8601 dates you compute from today's date (the automatic baseline "
    "search will pick up and reuse the same bounds automatically).\n"
    "- Otherwise, call semantic_search again with a rewritten or clarified query "
    "if the question is ambiguous or colloquially phrased, a narrower or broader "
    "phrasing, or a couple of separate calls to cover distinct aspects of a "
    "compound question.\n\n"
    "Call tools as many times as genuinely useful (rarely more than two or three "
    "extra calls), or not at all if the baseline is clearly sufficient. Never "
    "attempt to answer the question yourself — a separate step drafts the final "
    "answer from all search results gathered."
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    sources: list[dict]
    answer: str


def _agent_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    if not messages:
        today = datetime.now(timezone.utc).date().isoformat()
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT.format(today=today)),
            HumanMessage(content=state["query"]),
        ]
    # Deterministic tool selection: which extra searches (if any) get made should
    # be repeatable given the same query, so the eval set has a stable target to
    # snapshot against — only the final answer's prose needs variety.
    llm = make_llm(temperature=0).bind_tools(_TOOLS)
    response = llm.invoke(messages)
    return {**state, "messages": messages + [response]}


def _agent_time_bounds(messages: list) -> tuple[str | None, str | None]:
    """If the planner applied a date filter to any of its own searches, the
    guaranteed baseline search should respect it too — otherwise an unfiltered
    baseline leaks stale results back into a deliberately date-scoped query via
    RRF fusion, defeating the point of the filter. Takes the first bound found;
    in practice the planner applies the same window to every call it makes for
    one query."""
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            args = tc.get("args", {})
            after, before = args.get("time_after"), args.get("time_before")
            if after or before:
                return after, before
    return None, None


def _gather_sources(state: AgentState) -> AgentState:
    """Run the guaranteed verbatim baseline search, then fuse it by rank (RRF)
    with whatever the planner agent additionally searched for."""
    time_after, time_before = _agent_time_bounds(state["messages"])
    verbatim_results = semantic_search.invoke(
        {
            "query": state["query"],
            "k": _DEFAULT_K,
            "time_after": time_after,
            "time_before": time_before,
        }
    )

    agent_result_lists: list[list[dict]] = []
    for m in state["messages"]:
        if getattr(m, "type", None) == "tool" and m.name in _TOOL_NAMES:
            content = m.content
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    content = []
            agent_result_lists.append(content)

    merged = _reciprocal_rank_fusion([verbatim_results, *agent_result_lists])

    return {**state, "sources": merged}


def _synthesize_answer(state: AgentState) -> AgentState:
    query = state["query"]
    context = build_context(state["sources"])

    cached_answer = get_cached_answer(query, context)
    if cached_answer:
        return {**state, "answer": cached_answer}

    llm = make_llm()
    answer = llm.invoke(build_prompt(query, context)).content
    cache_answer(query, context, answer)
    return {**state, "answer": answer}


_compiled_agent_workflow = None


def create_agent_workflow():
    """Get or create the singleton compiled agentic workflow."""
    global _compiled_agent_workflow
    if _compiled_agent_workflow is None:
        logger.info("🔧 Compiling agentic RAG workflow...")
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", _agent_node)
        workflow.add_node("tools", ToolNode(_TOOLS))
        workflow.add_node("gather_sources", _gather_sources)
        workflow.add_node("synthesize_answer", _synthesize_answer)

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent", tools_condition, {"tools": "tools", END: "gather_sources"}
        )
        workflow.add_edge("tools", "gather_sources")
        workflow.add_edge("gather_sources", "synthesize_answer")
        workflow.add_edge("synthesize_answer", END)

        _compiled_agent_workflow = workflow.compile()
        logger.info("✅ Agentic RAG workflow compiled")
    return _compiled_agent_workflow
