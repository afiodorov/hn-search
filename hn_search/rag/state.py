from typing import List, TypedDict


class SearchResult(TypedDict):
    id: str
    author: str
    type: str
    text: str
    timestamp: str
    distance: float


class RAGState(TypedDict):
    query: str
    search_results: List[SearchResult]
    context: str
    answer: str
    error_message: str
