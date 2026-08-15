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
control — *except* when the planner called similar_comments, in which case the
baseline is redundant by construction (the id-based lookup already covers what
the link/id was pointing at) and skipping it is itself a fixed, code-level rule
keyed off state, not a new judgment call handed to the LLM: verbatim-embedding a
query that's mostly a pasted HN link produces junk (comments that merely
*mention* a URL, not comments related to the linked one), and that junk was
getting RRF-fused right in with the good similar_comments results. gather_sources
fuses all result lists by rank (RRF), caps the pool, and resolves each source's
parent comment (best-effort — most of the corpus predates the parent_id
backfill) so a short reply that's uninterpretable on its own gets context;
synthesize_answer drafts the final cited answer from the combined set.

The planner's tool-call decisions are captured into `AgentState["tool_calls"]`
the moment they're made (mirroring `response.tool_calls`), and any facts
derived from them (e.g. `time_after`/`time_before`, for propagating a date
filter onto the guaranteed baseline) are computed once in the same node and
stored as their own state fields too — rather than having every downstream
node re-derive "what did the planner decide" by re-walking `messages` or
re-parsing `tool_calls` each time it's needed. `messages` stays LangGraph's
carrier for LLM conversational history (`ToolNode` needs it in that shape),
it just isn't pressed into service as the only record of planner decisions.

Still single-hop for the planner (it can request 1+ tool calls in one turn, which
ToolNode runs together, but the graph doesn't loop back to the planner after) — a
real multi-turn loop is a later step, for when a tool's result needs to inform a
*subsequent* tool choice.
"""

import json
from datetime import datetime, timezone
from typing import Annotated, TypedDict, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from hn_search.cache_config import cache_answer, get_cached_answer
from hn_search.logging_config import get_logger
from hn_search.search_backend import get_docs

from .nodes import build_context, build_prompt, make_llm
from .state import SearchResult
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
# Cap how much of a parent comment's text gets pulled into the prompt — enough
# for it to give context, not so much a single long thread derails the budget.
_PARENT_TEXT_MAX_CHARS = 600


def _reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]], k: int = _RRF_K, limit: int = _MAX_SOURCES
) -> list[SearchResult]:
    """Fuse several ranked result lists (e.g. baseline search + each agent-added
    search) by rank rather than raw distance — raw cosine distances from
    different query embeddings aren't on a comparable scale, but rank position
    within a list always is."""
    scores: dict[str, float] = {}
    docs: dict[str, SearchResult] = {}
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
    tool_calls: list[dict]
    time_after: str | None
    time_before: str | None
    sources: list[SearchResult]
    parent_texts: dict[str, str]
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
    # .invoke()'s static return type is the generic BaseMessage; tool_calls is
    # only on AIMessage, which is what a tool-bound chat model always returns.
    tool_calls = getattr(response, "tool_calls", None) or []
    time_after, time_before = _agent_time_bounds(tool_calls)
    return {
        **state,
        "messages": messages + [response],
        "tool_calls": tool_calls,
        "time_after": time_after,
        "time_before": time_before,
    }


def _agent_time_bounds(tool_calls: list[dict]) -> tuple[str | None, str | None]:
    """If the planner applied a date filter to any of its own searches, the
    guaranteed baseline search should respect it too — otherwise an unfiltered
    baseline leaks stale results back into a deliberately date-scoped query via
    RRF fusion, defeating the point of the filter. Takes the first bound found;
    in practice the planner applies the same window to every call it makes for
    one query."""
    for tc in tool_calls:
        args = tc.get("args", {})
        after, before = args.get("time_after"), args.get("time_before")
        if after or before:
            return after, before
    return None, None


def _fetch_parent_texts(sources: list[SearchResult]) -> dict[str, str]:
    """For each source, resolve its parent comment's own text — a short reply
    ("I agree", "this is wrong") is often uninterpretable without knowing what
    it's replying to (validated empirically: adding this surfaced a genuinely
    new cited point the baseline had silently skipped for exactly this reason).
    Two batched round trips: each source's own parent_id, then the parents'
    text. Best-effort — any failure (including comments that predate the
    parent_id backfill and simply have none) just yields no parent context for
    that source, not an error."""
    if not sources:
        return {}
    try:
        own_docs = get_docs([s["id"] for s in sources])
    except Exception:
        logger.exception("parent-context lookup failed (own docs)")
        return {}

    parent_ids = set()
    for s in sources:
        pid = own_docs.get(s["id"], {}).get("parent_id")
        if pid:
            parent_ids.add(pid)
    if not parent_ids:
        return {}

    try:
        parent_docs = get_docs(list(parent_ids))
    except Exception:
        logger.exception("parent-context lookup failed (parent docs)")
        return {}

    result = {}
    for s in sources:
        pid = own_docs.get(s["id"], {}).get("parent_id")
        parent_doc = parent_docs.get(pid) if pid else None
        if parent_doc:
            text = parent_doc["clean_text"]
            if len(text) > _PARENT_TEXT_MAX_CHARS:
                text = text[:_PARENT_TEXT_MAX_CHARS] + "…"
            result[s["id"]] = text
    return result


def _run_baseline_search(
    query: str, time_after: str | None, time_before: str | None
) -> list[SearchResult]:
    return semantic_search.invoke(
        {"query": query, "k": _DEFAULT_K, "time_after": time_after, "time_before": time_before}
    )


def _fetch_referenced_docs(tool_calls: list[dict]) -> list[SearchResult]:
    """similar_comments deliberately excludes the referenced comment from its
    own results (it reuses that comment's embedding to find *other* similar
    ones) — correct for retrieval, but it means synthesis never sees the exact
    comment the user asked about, only comments around it (confirmed: DeepSeek
    noticed the gap and said as much in an answer). Fetch it directly and feed
    it in as its own rank-1 result list, so it merges in like any other source
    and becomes a normal numbered citation instead of a silent omission."""
    ids = [
        tc["args"]["hn_id"]
        for tc in tool_calls
        if tc["name"] == "similar_comments" and tc.get("args", {}).get("hn_id")
    ]
    if not ids:
        return []
    try:
        docs = get_docs(ids)
    except Exception:
        logger.exception("referenced-comment lookup failed")
        return []
    return [
        SearchResult(
            id=d["id"],
            author=d["author"],
            type=d["type"],
            text=d["clean_text"],
            timestamp=d["timestamp"],
            distance=0.0,
        )
        for d in docs.values()
    ]


def _gather_sources(state: AgentState) -> AgentState:
    """Run the guaranteed verbatim baseline search, then fuse it by rank (RRF)
    with whatever the planner agent additionally searched for.

    The baseline is skipped when the planner called similar_comments: the
    baseline would embed the user's raw text (often mostly a pasted HN link) as
    if it were a search query, which produces junk — comments that merely
    mention a URL, not comments related to the linked one — that then pollutes
    the RRF-fused result via similar_comments' own, correct results. Skipping
    is a fixed rule keyed off state (state["tool_calls"]), not a judgment call
    handed back to the LLM. When similar_comments fires, the referenced
    comment's own text is fetched and merged in too (see
    _fetch_referenced_docs) so synthesis has it, not just its neighbors.
    """
    tool_calls = state["tool_calls"]
    time_after, time_before = state["time_after"], state["time_before"]
    used_similar = any(tc["name"] == "similar_comments" for tc in tool_calls)

    verbatim_results: list[SearchResult] = (
        []
        if used_similar
        else _run_baseline_search(state["query"], time_after, time_before)
    )
    referenced_results = _fetch_referenced_docs(tool_calls) if used_similar else []

    agent_result_lists: list[list[SearchResult]] = []
    for m in state["messages"]:
        if getattr(m, "type", None) == "tool" and m.name in _TOOL_NAMES:
            content = m.content
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    content = []
            agent_result_lists.append(content)

    merged = _reciprocal_rank_fusion(
        [referenced_results, verbatim_results, *agent_result_lists]
    )
    if not merged:
        # Safety net: skipping the baseline (or the planner's own tool calls
        # returning nothing) must never leave us with zero sources.
        merged = _reciprocal_rank_fusion(
            [_run_baseline_search(state["query"], time_after, time_before)]
        )
    parent_texts = _fetch_parent_texts(merged)

    return {**state, "sources": merged, "parent_texts": parent_texts}


def _synthesize_answer(state: AgentState) -> AgentState:
    query = state["query"]
    context = build_context(state["sources"], state["parent_texts"])

    cached_answer = get_cached_answer(query, context)
    if cached_answer:
        return {**state, "answer": cached_answer}

    llm = make_llm()
    # DeepSeek's chat completions are text-only, so .content is always a plain
    # str here despite BaseMessage's broader str | list[...] type.
    answer = cast(str, llm.invoke(build_prompt(query, context)).content)
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
