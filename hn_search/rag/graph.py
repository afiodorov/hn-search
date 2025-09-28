from langgraph.graph import END, StateGraph

from .nodes import answer_node, retrieve_node
from .state import RAGState


def create_rag_workflow():
    workflow = StateGraph(RAGState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("answer", answer_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "answer")
    workflow.add_edge("answer", END)

    return workflow.compile()
