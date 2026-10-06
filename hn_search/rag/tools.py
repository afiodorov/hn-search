"""Tools for the agentic retrieval loop."""

import functools
from typing import cast

from langchain_core.tools import tool

from hn_search.cache_config import cache_vector_search, get_cached_vector_search
from hn_search.common import get_model
from hn_search.search_backend import get_docs, search, similar, stats

from .nodes import results_to_cache_data, rows_to_results
from .state import SearchResult
from .translate import to_english


@tool
def semantic_search(
    query: str,
    k: int = 10,
    time_after: str | None = None,
    time_before: str | None = None,
) -> list[SearchResult]:
    """Search Hacker News comments and stories by semantic similarity to a query.

    Ranks by meaning, not by date. time_after/time_before (YYYY-MM-DD,
    inclusive) restrict the search to that window and return the best matches
    inside it, however narrow — use them for a time period the user names, and
    for finding when something was first discussed. Omit both for an
    unrestricted search.

    Returns up to k results, each with id, author, timestamp, type, text, and
    distance (lower = more relevant).
    """
    cached = get_cached_vector_search(query, k, time_after, time_before)
    if cached:
        return cast(list[SearchResult], cached)

    # Cached under the original query, so a hit skips the translation too.
    english = to_english(query)
    embedding = get_model().encode([english or query])[0]
    rows = search(embedding, k, time_after=time_after, time_before=time_before)
    cache_data = results_to_cache_data(rows_to_results(rows))
    # An untranslated non-English query searched on noise: serve it, but don't
    # pin it in the cache past the translator's recovery.
    if cache_data and english is not None:
        cache_vector_search(query, cache_data, k, time_after, time_before)
    return cache_data


@tool
def similar_comments(hn_id: str, k: int = 10) -> list[SearchResult]:
    """Find Hacker News comments similar to a specific comment, given its numeric
    id (e.g. from a news.ycombinator.com/item?id=... link the user pasted, or a
    bare id they mentioned). Reuses that comment's own embedding — no need to
    describe its content in words. Returns up to k results in the same shape as
    semantic_search, excluding the comment itself. Raises if the id isn't found.
    """
    rows = similar(hn_id, k)
    return results_to_cache_data(rows_to_results(rows))


@tool
def get_comments(hn_ids: list[str]) -> list[SearchResult]:
    """Fetch comments by id, in full. Use it to read a comment a search only
    showed the start of, or to walk up a thread: each result's text is
    followed by its parent_id, which you can fetch in turn. Unknown ids are
    skipped."""
    docs = get_docs(hn_ids)
    return [
        SearchResult(
            id=d["id"],
            author=d["author"],
            type=d["type"],
            text=d["clean_text"]
            + (f"\n(parent_id: {d['parent_id']})" if d.get("parent_id") else ""),
            timestamp=d["timestamp"],
            distance=0.0,
        )
        for d in docs.values()
    ]


# What the archive held when this was written; used only if the service can't
# say (it predates `earliest_timestamp` in /stats, or it is down).
_FALLBACK_ARCHIVE_START = "2023-01-01"


@functools.cache
def _archive_start_from_service() -> str:
    return stats()["earliest_timestamp"][:10]


def archive_start() -> str:
    """The date of the oldest comment in the archive (YYYY-MM-DD). The service
    is asked once per process; a failure isn't cached, so it is asked again
    next time."""
    try:
        return _archive_start_from_service()
    except Exception:
        return _FALLBACK_ARCHIVE_START
