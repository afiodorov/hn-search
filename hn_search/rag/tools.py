"""Tools for the agentic retrieval loop.

Stage 1 of the agentic migration: exactly one tool, a direct wrapper of the same
search path retrieve_node/search_stream have always used, so the tool-calling
scaffold can be validated without changing retrieval behavior.
"""

from langchain_core.tools import tool

from hn_search.cache_config import cache_vector_search, get_cached_vector_search
from hn_search.common import get_model
from hn_search.search_backend import search

from .nodes import results_to_cache_data, rows_to_results


@tool
def semantic_search(query: str, k: int = 10) -> list[dict]:
    """Search Hacker News comments and stories by semantic similarity to a query.

    Returns up to k results, each with id, author, timestamp, type, text, and
    distance (lower = more relevant).
    """
    cached = get_cached_vector_search(query, k)
    if cached:
        return cached

    embedding = get_model().encode([query])[0]
    rows = search(embedding, k)
    cache_data = results_to_cache_data(rows_to_results(rows))
    if cache_data:
        cache_vector_search(query, cache_data, k)
    return cache_data
