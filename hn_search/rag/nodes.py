import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from hn_search.cache_config import (
    cache_answer,
    cache_vector_search,
    get_cached_answer,
    get_cached_vector_search,
)
from hn_search.common import get_model
from hn_search.logging_config import get_logger, log_time
from hn_search.search_backend import search

from .state import RAGState, SearchResult

load_dotenv()

logger = get_logger(__name__)


def cached_to_results(cached: list[dict]) -> list[SearchResult]:
    """Convert cached vector-search dicts back into SearchResults."""
    return [
        SearchResult(
            id=r["id"],
            author=r["author"],
            type=r["type"],
            text=r["text"],
            timestamp=r["timestamp"],
            distance=r["distance"],
        )
        for r in cached
    ]


def rows_to_results(rows) -> list[SearchResult]:
    """Convert _search result rows into SearchResult dicts."""
    return [
        SearchResult(
            id=doc_id,
            author=author,
            type=doc_type,
            text=document,
            timestamp=timestamp,
            distance=distance,
        )
        for doc_id, document, author, timestamp, doc_type, distance in rows
    ]


def results_to_cache_data(results: list[SearchResult]) -> list[dict]:
    """Convert SearchResults into JSON-able dicts for the Redis cache."""
    return [
        {
            "id": r["id"],
            "text": r["text"],
            "author": r["author"],
            "timestamp": r["timestamp"].isoformat()
            if hasattr(r["timestamp"], "isoformat")
            else str(r["timestamp"]),
            "type": r["type"],
            "distance": float(r["distance"]),
        }
        for r in results
    ]


def build_context(search_results: list[SearchResult]) -> str:
    return "\n\n---\n\n".join(
        [
            f"[{i + 1}] Author: {r['author']} ({r['timestamp']})\nLink: https://news.ycombinator.com/item?id={r['id']}\n{r['text']}"
            for i, r in enumerate(search_results)
        ]
    )


def build_prompt(query: str, context: str) -> str:
    return f"""You are a helpful assistant answering questions about Hacker News discussions.

User Question: {query}

Here are relevant comments and articles from Hacker News:

{context}

Please provide a comprehensive answer to the user's question based on the context above.
If the context doesn't contain enough information, say so.

When citing comments, use this format:
- For quotes: As user AuthorName puts it, "quote here" [[1]](link)
- For paraphrasing: User AuthorName explains that... [[2]](link)
- For multiple references: Several users [[3]](link1) [[4]](link2) discuss...

The [number] should match the source number from the context above, and should be a clickable link to the HN comment.

Example response format:
The community has mixed views on this topic. As user john_doe explains, "Python is great for prototyping" [[1]](https://news.ycombinator.com/item?id=12345). Meanwhile, user jane_smith argues that performance can be an issue [[2]](https://news.ycombinator.com/item?id=67890)."""


def make_llm() -> ChatOpenAI:
    # cache=False: opt out of any global LangChain LLM cache (unreliable with
    # .stream()); the explicit get_cached_answer/cache_answer functions are
    # the real answer cache.
    return ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0.7,
        cache=False,
    )


def retrieve_node(state: RAGState) -> RAGState:
    query = state["query"]
    n_results = 10

    try:
        # Check cache first
        with log_time(logger, "cache lookup"):
            cached_results = get_cached_vector_search(query, n_results)

        if cached_results:
            logger.info(f"🔍 Using cached results for: {query}")
            search_results = cached_to_results(cached_results)
            context = build_context(search_results)

            logger.info(
                f"✅ Found {len(search_results)} relevant comments/articles (cached)"
            )

            return {
                **state,
                "search_results": search_results,
                "context": context,
            }

        logger.info(f"🔍 Searching for: {query}")

        # Use singleton embedding model
        with log_time(logger, "query embedding generation"):
            model = get_model()
            query_embedding = model.encode([query])[0]

        # Binary Hamming shortlist + exact cosine rerank, served by the Rust service.
        with log_time(logger, "vector search (shortlist + rerank)"):
            results = search(query_embedding, n_results)

        with log_time(logger, "building results"):
            search_results = rows_to_results(results)
            cache_data = results_to_cache_data(search_results)

        # Cache the results
        if cache_data:
            with log_time(logger, "caching search results"):
                cache_vector_search(query, cache_data, n_results)

        context = build_context(search_results)

        logger.info(f"✅ Found {len(search_results)} relevant comments/articles")

        return {
            **state,
            "search_results": search_results,
            "context": context,
        }
    except Exception as e:
        error_msg = "vector type not found in the database"
        logger.exception(f"Database error: {str(e)}")
        return {
            **state,
            "error_message": error_msg,
            "search_results": [],
            "context": "",
        }


def answer_node(state: RAGState) -> RAGState:
    query = state["query"]
    context = state["context"]

    # Check cache first
    with log_time(logger, "answer cache lookup"):
        cached_answer = get_cached_answer(query, context)

    if cached_answer:
        logger.info("🤖 Using cached answer")
        return {
            **state,
            "answer": cached_answer,
        }

    logger.info("🤖 Generating answer with DeepSeek...")

    with log_time(logger, "LLM answer generation"):
        llm = make_llm()
        response = llm.invoke(build_prompt(query, context))
        answer = response.content

    # Cache the answer
    with log_time(logger, "caching answer"):
        cache_answer(query, context, answer)

    logger.info("✅ Answer generated")

    return {
        **state,
        "answer": answer,
    }
