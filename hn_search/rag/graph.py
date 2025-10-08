from langgraph.graph import END, StateGraph

from hn_search.logging_config import get_logger

from .nodes import answer_node, retrieve_node
from .state import RAGState

logger = get_logger(__name__)

# Singleton compiled workflow
_compiled_workflow = None


def create_rag_workflow():
    """Get or create singleton compiled RAG workflow."""
    global _compiled_workflow
    if _compiled_workflow is None:
        logger.info("🔧 Compiling RAG workflow...")
        workflow = StateGraph(RAGState)

        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("answer", answer_node)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "answer")
        workflow.add_edge("answer", END)

        _compiled_workflow = workflow.compile()
        logger.info("✅ RAG workflow compiled")
    return _compiled_workflow
