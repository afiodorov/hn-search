import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from hn_search.common import COLLECTION_NAME, get_client, get_model

from .state import RAGState, SearchResult

load_dotenv()


def retrieve_node(state: RAGState) -> RAGState:
    query = state["query"]
    n_results = 10

    client = get_client()
    model = get_model()

    print(f"🔍 Searching for: {query}")

    collection = client.get_collection(name=COLLECTION_NAME)
    query_embedding = model.encode([query])

    results = collection.query(
        query_embeddings=query_embedding.tolist(), n_results=n_results
    )

    search_results = []
    for doc_id, document, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        search_results.append(
            SearchResult(
                id=doc_id,
                author=metadata["author"],
                type=metadata["type"],
                text=document,
                timestamp=metadata["timestamp"],
                distance=distance,
            )
        )

    context = "\n\n---\n\n".join(
        [
            f"[{i + 1}] Author: {r['author']} ({r['timestamp']})\n{r['text']}"
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
Include relevant quotes and mention the authors when appropriate."""

    response = llm.invoke(prompt)

    print("✅ Answer generated")

    return {
        **state,
        "answer": response.content,
    }
