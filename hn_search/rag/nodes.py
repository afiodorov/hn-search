import os

import psycopg
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from hn_search.cache_config import (
    cache_answer,
    cache_vector_search,
    get_cached_answer,
    get_cached_vector_search,
)
from hn_search.common import get_device
from hn_search.db_config import get_db_config

from .state import RAGState, SearchResult

load_dotenv()


def retrieve_node(state: RAGState) -> RAGState:
    query = state["query"]
    n_results = 10

    try:
        # Check cache first
        cached_results = get_cached_vector_search(query, n_results)
        if cached_results:
            print(f"🔍 Using cached results for: {query}")
            search_results = []
            for result in cached_results:
                search_results.append(
                    SearchResult(
                        id=result["id"],
                        author=result["author"],
                        type=result["type"],
                        text=result["text"],
                        timestamp=result["timestamp"],
                        distance=result["distance"],
                    )
                )

            context = "\n\n---\n\n".join(
                [
                    f"[{i + 1}] Author: {r['author']} ({r['timestamp']})\nLink: https://news.ycombinator.com/item?id={r['id']}\n{r['text']}"
                    for i, r in enumerate(search_results)
                ]
            )

            print(f"✅ Found {len(search_results)} relevant comments/articles (cached)")

            return {
                **state,
                "search_results": search_results,
                "context": context,
            }

        device = get_device()
        model = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2", device=device
        )

        print(f"🔍 Searching for: {query}")

        query_embedding = model.encode([query])[0]

        db_config = get_db_config()
        print(f"Connecting to database at {db_config['host']}:{db_config['port']}")

        conn = psycopg.connect(**db_config)
        print("Database connection established")

        register_vector(conn)
        print("Vector extension registered")

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, clean_text, author, timestamp, type,
                       embedding <=> %s::vector AS distance
                FROM hn_documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding.tolist(), query_embedding.tolist(), n_results),
            )

            results = cur.fetchall()
            search_results = []
            cache_data = []

            for doc_id, document, author, timestamp, doc_type, distance in results:
                search_results.append(
                    SearchResult(
                        id=doc_id,
                        author=author,
                        type=doc_type,
                        text=document,
                        timestamp=timestamp,
                        distance=distance,
                    )
                )
                # Prepare for caching
                cache_data.append(
                    {
                        "id": doc_id,
                        "text": document,
                        "author": author,
                        "timestamp": timestamp.isoformat()
                        if hasattr(timestamp, "isoformat")
                        else str(timestamp),
                        "type": doc_type,
                        "distance": float(distance),
                    }
                )

        conn.close()

        # Cache the results
        if cache_data:
            cache_vector_search(query, cache_data, n_results)

        context = "\n\n---\n\n".join(
            [
                f"[{i + 1}] Author: {r['author']} ({r['timestamp']})\nLink: https://news.ycombinator.com/item?id={r['id']}\n{r['text']}"
                for i, r in enumerate(search_results)
            ]
        )

        print(f"✅ Found {len(search_results)} relevant comments/articles")

        return {
            **state,
            "search_results": search_results,
            "context": context,
        }
    except Exception as e:
        error_msg = f"vector type not found in the database"
        print(f"Database error: {str(e)}")
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
    cached_answer = get_cached_answer(query, context)
    if cached_answer:
        print("🤖 Using cached answer")
        return {
            **state,
            "answer": cached_answer,
        }

    print("🤖 Generating answer with DeepSeek...")

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0.7,
    )

    prompt = f"""You are a helpful assistant answering questions about Hacker News discussions.

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

    response = llm.invoke(prompt)
    answer = response.content

    # Cache the answer
    cache_answer(query, context, answer)

    print("✅ Answer generated")

    return {
        **state,
        "answer": answer,
    }
