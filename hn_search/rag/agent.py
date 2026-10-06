"""Agentic graph: a planner that searches in a loop, then a dedicated synthesis
node.

    guard → seed → agent ⇄ tools → gather_sources → synthesize_answer

The planner LLM decides what to retrieve. It calls semantic_search (optionally
date-bounded), similar_comments (by HN id/link) and get_comments (full text and
parent ids) as many times as it finds useful, sees every result before choosing
its next step, and finishes by naming the comments worth citing. That replaces
the earlier single-shot design, where code decided how results combined: a
fixed verbatim search fused by rank with whatever the planner asked for, and
hand-written exceptions for cases the fusion got wrong (skip the verbatim
search when a link was pasted, carry the planner's dates over to it). Each new
kind of question needed another exception. Now the planner looks at the
results and judges, which covers those cases and new ones like "the first
mention of X" (search X in an early date window, check, move the window).

`seed` is the one fixed step. A plain semantic search on the user's exact
question runs before the planner's first turn, and its results reach the
planner as if it had made the call itself. We keep it because a planner left
alone rewrote queries and drifted retrieval away from what plain search found
(seen in eval_judge history). The planner starts from those results and can
still drop them: it picks the final sources, so a verbatim search that was
noise (a pasted link, say) simply goes unpicked.

The loop is bounded: after `_MAX_ROUNDS` rounds of tool calls the planner is
asked once more with tools disabled, so it has to finish. If its final reply
names no usable ids, the sources fall back to every result in the order it was
found, seed first, so a confused planner never leaves synthesis with nothing.

`guard` runs before any of that. It is the one node that can end the run on its
own: an off-topic or over-long query routes straight to END with a canned
refusal, so the seed search, the planner and the synthesis call never happen.
See `guard.py` for why that is a separate model call rather than a line in a
prompt.
"""

import functools
import json
import re
from datetime import datetime, timezone
from typing import Annotated, TypedDict, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from hn_search.cache_config import cache_answer, get_cached_answer
from hn_search.logging_config import get_logger
from hn_search.search_backend import get_docs

from . import guard
from .nodes import build_context, build_prompt, make_llm
from .state import SearchResult
from .tools import (
    KeywordStats,
    archive_start,
    get_comments,
    keyword_stats,
    semantic_search,
    similar_comments,
)

logger = get_logger(__name__)

_TOOLS = [semantic_search, similar_comments, get_comments, keyword_stats]
_TOOLS_BY_NAME = {t.name: t for t in _TOOLS}
_DEFAULT_K = 10
# Rounds of tool calls before the planner must finish. One round is one planner
# turn (~2s) plus its searches; synthesis (~20s) dominates either way.
_MAX_ROUNDS = 4
# Cap on the sources fed to synthesis, so context size — and DeepSeek
# latency/cost — doesn't scale with how much the planner searched.
_MAX_SOURCES = 12
# How much of each result the planner sees. Enough to judge relevance; it can
# get_comments for the full text.
_SNIPPET_CHARS = 300
# Cap how much of a parent comment's text gets pulled into the prompt — enough
# for it to give context, not so much a single long thread derails the budget.
_PARENT_TEXT_MAX_CHARS = 600
_SEED_CALL_ID = "seed"

_SYSTEM_PROMPT = """You are the research planner for a search assistant over \
Hacker News comments. Today's date is {today}. The archive holds HN comments and \
stories from {archive_start} to today; nothing older is in it.

Your job is to find the comments that answer the user's question. A separate \
step writes the answer from the comments you pick; never answer the question \
yourself. A plain semantic search for the user's exact words has already run, \
and its results are below. Look at them, then decide what else, if anything, to \
search.

Tools:
- semantic_search(query, k, time_after, time_before): comments similar in \
meaning to the query, best match first. It ranks by meaning, not date. \
time_after/time_before (YYYY-MM-DD, inclusive) return the best matches inside \
that window, however narrow.
- similar_comments(hn_id): comments like a given comment.
- get_comments(hn_ids): full text of comments by id, with each one's parent_id, \
for reading a comment in full or walking up a thread.
- keyword_stats(term, time_after, time_before, k): exact counts from a \
full-text index over the whole archive: how many comments contain the word or \
phrase, per month (with each month's total), and the k earliest that contain \
it. Whole words, case-insensitive; it matches words, not meanings.

How to search well:
- If the results you already have answer the question, finish straight away.
- Rephrase a vague or colloquial question in the words commenters would use. \
Search each side of a comparison, and each part of a compound question, on its own.
- For a time period ("last 3 months", "in 2024", "recently"), set \
time_after/time_before, computed from today's date.
- For a news.ycombinator.com/item?id=... link or a bare comment id, call \
similar_comments with that id; the plain search on a pasted link is usually noise.
- For "how many", "how often", "how popular over time", "the first mention of X" \
or "when did people start talking about X", call keyword_stats with the term as \
commenters would write it (try the common spellings or names, e.g. "chatgpt" and \
"chat gpt", in the same round). Its figures reach the writer exactly as counted, \
so don't restate them in your notes. Its earliest matches are oldest first; check \
they really are about X (a word can have other meanings) and pick from them. If \
the earliest match sits at the archive start, X is older than the archive; say \
so in your notes. Add a semantic_search when the question also asks what people \
said.
- You have at most {max_rounds} rounds of tool calls. Make independent searches \
in the same round.

When you are done, reply with no tool calls and only this JSON:
{{"sources": ["<id>", ...], "notes": "<one or two sentences for the writer>"}}
sources: up to {max_sources} ids from the results you have seen, most useful \
first, only ones that help answer the question; for a first-mention question, \
oldest first. notes: what the writer needs to know that the comments don't say \
themselves, e.g. the date window you searched, or that the topic is older than \
the archive. Leave notes empty if there is nothing to add."""

_FINISH_NOW = (
    "That was your last round of searches. Reply now with only the JSON: "
    '{"sources": [...], "notes": "..."}'
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    # Set by the guard node and read only by the router after it.
    on_topic: bool
    # Rounds of tool calls run so far.
    rounds: int
    # Every tool call made, the seed included, in order: what was searched.
    tool_calls: list[dict]
    # Every result any tool returned, by id, in the order first found.
    pool: dict[str, SearchResult]
    # Exact figures from keyword_stats, passed to the writer verbatim: numbers
    # a planner paraphrases are numbers it can get wrong.
    facts: list[str]
    notes: str
    sources: list[SearchResult]
    parent_texts: dict[str, str]
    answer: str


def initial_state(query: str) -> AgentState:
    return AgentState(
        messages=[],
        query=query,
        on_topic=False,
        rounds=0,
        tool_calls=[],
        pool={},
        facts=[],
        notes="",
        sources=[],
        parent_texts={},
        answer="",
    )


def _guard_node(state: AgentState) -> dict:
    """Admit or refuse. A refusal fills in `answer` itself (and an empty
    `sources`), which is all a downstream consumer needs to render it."""
    query = state["query"]
    if len(query) > guard.MAX_QUERY_CHARS:
        # Free: no model call for a pasted document.
        return {"on_topic": False, "answer": guard.TOO_LONG_TEXT, "sources": []}
    if guard.is_in_scope(query):  # fails open; never raises
        return {"on_topic": True}
    return {"on_topic": False, "answer": guard.REFUSAL_TEXT, "sources": []}


def _admitted(state: AgentState) -> str:
    return "seed" if state["on_topic"] else END


def _format_results(results: list[SearchResult]) -> str:
    """What the planner reads: one compact block per result, id first, so it
    can name ids back in its selection."""
    if not results:
        return "No results."
    blocks = []
    for r in results:
        text = r["text"]
        if len(text) > _SNIPPET_CHARS:
            text = text[:_SNIPPET_CHARS] + "…"
        blocks.append(f"[{r['id']}] {r['author']} · {str(r['timestamp'])[:10]}\n{text}")
    return f"{len(results)} results:\n\n" + "\n\n".join(blocks)


def _add_to_pool(
    pool: dict[str, SearchResult], results: list[SearchResult]
) -> dict[str, SearchResult]:
    merged = dict(pool)
    for r in results:
        merged.setdefault(r["id"], r)
    return merged


def format_keyword_stats(stats: KeywordStats) -> str:
    """The figures as plain text, for the planner and for the writer."""
    window = (
        f" between {stats['time_after'] or 'the archive start'} and "
        f"{stats['time_before'] or 'today'}"
        if stats["time_after"] or stats["time_before"]
        else ""
    )
    share = 100 * stats["count"] / stats["rows"] if stats["rows"] else 0.0
    lines = [
        f'Exact count for "{stats["term"]}"{window}: {stats["count"]:,} of '
        f"{stats['rows']:,} comments ({share:.2f}%) contain it."
    ]
    if stats["count"] and len(stats["months"]) > 1:
        months = ", ".join(
            f"{m['month']}: {m['count']:,}/{m['rows']:,}" for m in stats["months"]
        )
        lines.append(f"By month (comments containing it / all comments): {months}.")
    return "\n".join(lines)


def _run_tool(name: str, args: dict) -> tuple[list[SearchResult], str, str | None]:
    """Run one tool call; return its citable results, what the planner is
    shown, and an exact figure for the writer if the tool produced one. A
    failing call (bad id, service hiccup) is reported to the planner as text,
    so it can try something else, rather than ending the run."""
    tool = _TOOLS_BY_NAME.get(name)
    if tool is None:
        return [], f"Unknown tool {name!r}.", None
    try:
        out = tool.invoke(args)
    except Exception as e:
        logger.warning(f"tool {name} failed: {e}")
        return [], f"The call failed: {e}", None
    if name == keyword_stats.name:
        stats = cast(KeywordStats, out)
        fact = format_keyword_stats(stats)
        earliest = _format_results(stats["first"]) if stats["first"] else ""
        shown = (
            f"{fact}\n\nEarliest matches, oldest first: {earliest}"
            if earliest
            else fact
        )
        return stats["first"], shown, fact
    results = cast(list[SearchResult], out)
    return results, _format_results(results), None


def _seed_node(state: AgentState) -> dict:
    """The verbatim search, recorded as the planner's own first tool call so
    the conversation reads naturally from its first turn."""
    query = state["query"]
    today = datetime.now(timezone.utc).date().isoformat()
    call = {
        "name": semantic_search.name,
        "args": {"query": query, "k": _DEFAULT_K},
        "id": _SEED_CALL_ID,
        "type": "tool_call",
    }
    results, shown, _ = _run_tool(call["name"], call["args"])
    return {
        "messages": [
            SystemMessage(
                content=_SYSTEM_PROMPT.format(
                    today=today,
                    archive_start=archive_start(),
                    max_rounds=_MAX_ROUNDS,
                    max_sources=_MAX_SOURCES,
                )
            ),
            HumanMessage(content=query),
            AIMessage(content="", tool_calls=[call]),
            ToolMessage(content=shown, tool_call_id=_SEED_CALL_ID, name=call["name"]),
        ],
        "tool_calls": [call],
        "pool": _add_to_pool(state["pool"], results),
    }


def _agent_node(state: AgentState) -> dict:
    # Deterministic tool selection: which searches get made should be
    # repeatable given the same query, so the eval set has a stable target to
    # compare against — only the final answer's prose needs variety.
    llm = make_llm(temperature=0, thinking=False)
    if state["rounds"] >= _MAX_ROUNDS:
        bound = llm.bind_tools(_TOOLS, tool_choice="none")
        messages = [*state["messages"], HumanMessage(content=_FINISH_NOW)]
    else:
        bound = llm.bind_tools(_TOOLS)
        messages = state["messages"]
    response = bound.invoke(messages)
    if state["rounds"] >= _MAX_ROUNDS:
        return {"messages": [messages[-1], response]}
    return {"messages": [response]}


def _wants_tools(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "gather_sources"


def _tools_node(state: AgentState) -> dict:
    """Run every call from the planner's last turn (a round), add the results
    to the pool and hand them back to it."""
    last = cast(AIMessage, state["messages"][-1])
    pool = state["pool"]
    facts = list(state["facts"])
    replies: list[BaseMessage] = []
    for call in last.tool_calls:
        results, shown, fact = _run_tool(call["name"], call.get("args", {}))
        pool = _add_to_pool(pool, results)
        if fact and fact not in facts:
            facts.append(fact)
        replies.append(
            ToolMessage(content=shown, tool_call_id=call["id"], name=call["name"])
        )
    return {
        "messages": replies,
        "rounds": state["rounds"] + 1,
        "tool_calls": [*state["tool_calls"], *last.tool_calls],
        "pool": pool,
        "facts": facts,
    }


_ID = re.compile(r"\b\d{5,}\b")


def _parse_selection(content: str) -> tuple[list[str], str]:
    """The planner's final reply → (ids, notes). Expects the JSON the prompt
    asks for, possibly fenced; failing that, takes any ids it mentions, so a
    planner that answers in prose still has its picks honoured."""
    text = content.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            ids = [str(i) for i in data.get("sources", [])]
            return ids, str(data.get("notes") or "").strip()
        except (json.JSONDecodeError, AttributeError):
            pass
    return _ID.findall(text), ""


def _select_sources(
    pool: dict[str, SearchResult], picked: list[str]
) -> list[SearchResult]:
    chosen = [pool[i] for i in dict.fromkeys(picked) if i in pool]
    if not chosen:
        chosen = list(pool.values())
    return chosen[:_MAX_SOURCES]


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


def _gather_sources(state: AgentState) -> dict:
    """The planner's picks, in its order, from everything it found."""
    final = state["messages"][-1]
    picked, notes = _parse_selection(str(getattr(final, "content", "") or ""))
    sources = _select_sources(state["pool"], picked)
    return {
        "sources": sources,
        "notes": notes,
        "parent_texts": _fetch_parent_texts(sources),
    }


def _synthesize_answer(state: AgentState) -> dict:
    query = state["query"]
    notes = state["notes"]
    facts = "\n".join(state["facts"])
    context = build_context(state["sources"], state["parent_texts"])
    # The notes and figures shape the answer as much as the sources do, so they
    # are part of what the cached answer is keyed on.
    cache_context = "\n\n".join(part for part in (facts, notes, context) if part)

    cached_answer = get_cached_answer(query, cache_context)
    if cached_answer:
        return {"answer": cached_answer}

    llm = make_llm()
    prompt = build_prompt(
        query, context, notes=notes, facts=facts, archive_start=archive_start()
    )
    # DeepSeek's chat completions are text-only, so .content is always a plain
    # str here despite BaseMessage's broader str | list[...] type.
    answer = cast(str, llm.invoke(prompt).content)
    cache_answer(query, cache_context, answer)
    return {"answer": answer}


@functools.cache
def create_agent_workflow():
    """Get or create the singleton compiled agentic workflow."""
    logger.info("🔧 Compiling agentic RAG workflow...")
    workflow = StateGraph(AgentState)

    workflow.add_node("guard", _guard_node)
    workflow.add_node("seed", _seed_node)
    workflow.add_node("agent", _agent_node)
    workflow.add_node("tools", _tools_node)
    workflow.add_node("gather_sources", _gather_sources)
    workflow.add_node("synthesize_answer", _synthesize_answer)

    workflow.set_entry_point("guard")
    workflow.add_conditional_edges("guard", _admitted, {"seed": "seed", END: END})
    workflow.add_edge("seed", "agent")
    workflow.add_conditional_edges(
        "agent", _wants_tools, {"tools": "tools", "gather_sources": "gather_sources"}
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("gather_sources", "synthesize_answer")
    workflow.add_edge("synthesize_answer", END)

    compiled = workflow.compile()
    logger.info("✅ Agentic RAG workflow compiled")
    return compiled
