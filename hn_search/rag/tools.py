"""Tools for the agentic retrieval loop."""

from langchain_core.tools import tool

from hn_search.cache_config import cache_vector_search, get_cached_vector_search
from hn_search.common import get_model
from hn_search.search_backend import search, similar

from .nodes import results_to_cache_data, rows_to_results


@tool
def semantic_search(
    query: str,
    k: int = 10,
    time_after: str | None = None,
    time_before: str | None = None,
) -> list[dict]:
    """Search Hacker News comments and stories by semantic similarity to a query.

    time_after/time_before optionally restrict results to a date range — pass
    ISO8601 dates (YYYY-MM-DD) if the user's question mentions a time period
    (e.g. "in the last 6 months", "since 2023", "in 2022"); compute the actual
    dates yourself from today's date. Omit both for an unrestricted search.

    Returns up to k results, each with id, author, timestamp, type, text, and
    distance (lower = more relevant).
    """
    cached = get_cached_vector_search(query, k, time_after, time_before)
    if cached:
        return cached

    embedding = get_model().encode([query])[0]
    rows = search(embedding, k, time_after=time_after, time_before=time_before)
    cache_data = results_to_cache_data(rows_to_results(rows))
    if cache_data:
        cache_vector_search(query, cache_data, k, time_after, time_before)
    return cache_data


@tool
def similar_comments(hn_id: str, k: int = 10) -> list[dict]:
    """Find Hacker News comments similar to a specific comment, given its numeric
    id (e.g. from a news.ycombinator.com/item?id=... link the user pasted, or a
    bare id they mentioned). Reuses that comment's own embedding — no need to
    describe its content in words. Returns up to k results in the same shape as
    semantic_search, excluding the comment itself. Raises if the id isn't found.
    """
    rows = similar(hn_id, k)
    return results_to_cache_data(rows_to_results(rows))
