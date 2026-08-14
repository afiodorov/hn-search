"""Stage 1 agentic graph: a tool-calling planner + a dedicated synthesis node.

Single hop only for now (agent decides tool args once, tools execute once,
synthesize_answer drafts the final cited answer) — the point of this stage is to
validate the tool-calling scaffold, not to add new capability. Extending to a real
multi-hop loop (route "tools" back to "agent" instead of straight to
extract_sources, with a max-iteration cap) is Stage 2's job, once there's more than
one tool to choose between.
"""

import json
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from hn_search.cache_config import cache_answer, get_cached_answer
from hn_search.logging_config import get_logger

from .nodes import build_context, build_prompt, make_llm
from .tools import semantic_search

logger = get_logger(__name__)

_TOOLS = [semantic_search]

_SYSTEM_PROMPT = (
    "You are the retrieval planner for a Hacker News search assistant. For the "
    "user's question, call semantic_search exactly once, passing the user's "
    "question to the query argument verbatim, unmodified and unabridged — do not "
    "paraphrase, rewrite, or summarize it, even if you think a rewrite would "
    "retrieve better results. Then stop: do not call any tool more than once, and "
    "do not attempt to answer the question yourself — a separate step drafts the "
    "final answer from your search results."
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    sources: list[dict]
    answer: str


def _agent_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    if not messages:
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=state["query"]),
        ]
    llm = make_llm().bind_tools(_TOOLS)
    response = llm.invoke(messages)
    return {**state, "messages": messages + [response]}


def _extract_sources(state: AgentState) -> AgentState:
    sources: list[dict] = []
    for m in state["messages"]:
        if getattr(m, "type", None) == "tool" and m.name == "semantic_search":
            content = m.content
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    content = []
            sources.extend(content)
    return {**state, "sources": sources}


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
    """Get or create the singleton compiled Stage-1 agentic workflow."""
    global _compiled_agent_workflow
    if _compiled_agent_workflow is None:
        logger.info("🔧 Compiling agentic RAG workflow...")
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", _agent_node)
        workflow.add_node("tools", ToolNode(_TOOLS))
        workflow.add_node("extract_sources", _extract_sources)
        workflow.add_node("synthesize_answer", _synthesize_answer)

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent", tools_condition, {"tools": "tools", END: "extract_sources"}
        )
        workflow.add_edge("tools", "extract_sources")
        workflow.add_edge("extract_sources", "synthesize_answer")
        workflow.add_edge("synthesize_answer", END)

        _compiled_agent_workflow = workflow.compile()
        logger.info("✅ Agentic RAG workflow compiled")
    return _compiled_agent_workflow
