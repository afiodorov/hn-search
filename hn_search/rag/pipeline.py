"""Streaming search pipeline: the retrieve/answer flow as a flat event generator.

Mirrors retrieve_node + answer_node (same caches, same SQL, same prompt) but
yields typed events after every step so the API can stream progress, timings
and answer tokens to the browser. The LangGraph graph stays for the CLI.

Event types:
    {"type": "progress", "step", "label", "status": "start"|"done", "ms", "hit"}
    {"type": "sources", "sources": [{id, author, timestamp, type, text, url, distance}]}
    {"type": "token", "text"}    -- answer delta (LLM path only)
    {"type": "answer", "text"}   -- full answer, always emitted last
    {"type": "error", "message"}
"""

import time
from typing import Iterator

from hn_search.cache_config import (
    cache_answer,
    cache_vector_search,
    get_cached_answer,
    get_cached_vector_search,
)
from hn_search.common import get_model
from hn_search.logging_config import get_logger

from .nodes import (
    _search,
    build_context,
    build_prompt,
    cached_to_results,
    get_connection_pool,
    make_llm,
    results_to_cache_data,
    rows_to_results,
)

logger = get_logger(__name__)


class _Step:
    """Times a pipeline step and produces its start/done progress events."""

    def __init__(self, step: str, label: str):
        self.step = step
        self.label = label
        self._t0 = None

    def start(self) -> dict:
        self._t0 = time.perf_counter()
        return {
            "type": "progress",
            "step": self.step,
            "label": self.label,
            "status": "start",
            "ms": None,
            "hit": None,
        }

    def done(self, hit: bool | None = None) -> dict:
        ms = round((time.perf_counter() - self._t0) * 1000)
        logger.info(f"⏱️ {self.label}: {ms}ms" + (" (cache hit)" if hit else ""))
        return {
            "type": "progress",
            "step": self.step,
            "label": self.label,
            "status": "done",
            "ms": ms,
            "hit": hit,
        }


def _sources_event(search_results) -> dict:
    return {
        "type": "sources",
        "sources": [
            {
                **r,
                "url": f"https://news.ycombinator.com/item?id={r['id']}",
            }
            for r in results_to_cache_data(search_results)
        ],
    }


def search_stream(query: str, n_results: int = 10) -> Iterator[dict]:
    # --- Retrieve (mirrors retrieve_node) ---
    try:
        step = _Step("vector_cache", "Checking search cache")
        yield step.start()
        cached_results = get_cached_vector_search(query, n_results)
        yield step.done(hit=bool(cached_results))

        if cached_results:
            search_results = cached_to_results(cached_results)
        else:
            step = _Step("embed", "Embedding query")
            yield step.start()
            embedding = get_model().encode([query])[0]
            yield step.done()

            step = _Step("search", "Searching Postgres (pgvector)")
            yield step.start()
            rows = _search(get_connection_pool(), embedding, n_results)
            yield step.done()

            search_results = rows_to_results(rows)
            cache_data = results_to_cache_data(search_results)
            if cache_data:
                cache_vector_search(query, cache_data, n_results)

        context = build_context(search_results)
        logger.info(f"✅ Found {len(search_results)} relevant comments/articles")
        yield _sources_event(search_results)
    except Exception as e:
        logger.exception(f"Database error: {e}")
        yield {"type": "error", "message": "vector type not found in the database"}
        return

    # --- Answer (mirrors answer_node) ---
    try:
        step = _Step("answer_cache", "Checking answer cache")
        yield step.start()
        cached_answer = get_cached_answer(query, context)
        yield step.done(hit=bool(cached_answer))

        if cached_answer:
            yield {"type": "answer", "text": cached_answer}
            return

        step = _Step("llm", "Asking DeepSeek")
        yield step.start()
        parts = []
        for chunk in make_llm().stream(build_prompt(query, context)):
            if chunk.content:
                parts.append(chunk.content)
                yield {"type": "token", "text": chunk.content}
        answer = "".join(parts)
        yield step.done()

        cache_answer(query, context, answer)
        yield {"type": "answer", "text": answer}
    except Exception as e:
        logger.exception(f"LLM error: {e}")
        yield {"type": "error", "message": str(e)}
