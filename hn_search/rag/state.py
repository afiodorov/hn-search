from typing import TypedDict


class SearchResult(TypedDict):
    id: str
    author: str
    type: str
    text: str
    timestamp: str
    distance: float
