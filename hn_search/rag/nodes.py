import os

import psycopg
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from hn_search.common import get_device

from .state import RAGState, SearchResult

load_dotenv()


def retrieve_node(state: RAGState) -> RAGState:
    query = state["query"]
    n_results = 10

    device = get_device()
    model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2", device=device
    )

    print(f"🔍 Searching for: {query}")

    query_embedding = model.encode([query])[0]

    conn = psycopg.connect(
        host="localhost",
        port=5432,
        dbname="hn_search",
        user="postgres",
        password="postgres",
    )
    register_vector(conn)

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

    conn.close()

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


def answer_node(state: RAGState) -> RAGState:
    query = state["query"]
    context = state["context"]

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

    print("✅ Answer generated")

    return {
        **state,
        "answer": response.content,
    }
