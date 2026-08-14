import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .state import SearchResult

load_dotenv()


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


def results_to_cache_data(results: list[SearchResult]) -> list[SearchResult]:
    """Convert SearchResults into JSON-able dicts for the Redis cache."""
    return [
        SearchResult(
            id=r["id"],
            text=r["text"],
            author=r["author"],
            # SearchResult declares timestamp: str, but a raw DB row can still
            # hand this a datetime before it's been normalized — handle both.
            timestamp=r["timestamp"].isoformat()  # pyright: ignore[reportAttributeAccessIssue]
            if hasattr(r["timestamp"], "isoformat")
            else str(r["timestamp"]),
            type=r["type"],
            distance=float(r["distance"]),
        )
        for r in results
    ]


def build_context(
    search_results: list[SearchResult], parent_texts: dict[str, str] | None = None
) -> str:
    parent_texts = parent_texts or {}
    blocks = []
    for i, r in enumerate(search_results):
        parent = parent_texts.get(r["id"])
        reply_to = f"In reply to: {parent}\n\n" if parent else ""
        blocks.append(
            f"[{i + 1}] Author: {r['author']} ({r['timestamp']})\n"
            f"Link: https://news.ycombinator.com/item?id={r['id']}\n"
            f"{reply_to}{r['text']}"
        )
    return "\n\n---\n\n".join(blocks)


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


def make_llm(temperature: float = 0.7) -> ChatOpenAI:
    # cache=False: opt out of any global LangChain LLM cache (unreliable with
    # .stream()); the explicit get_cached_answer/cache_answer functions are
    # the real answer cache.
    api_key = os.getenv("DEEPSEEK_API_KEY")
    return ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=SecretStr(api_key) if api_key else None,
        base_url="https://api.deepseek.com",
        temperature=temperature,
        cache=False,
    )
