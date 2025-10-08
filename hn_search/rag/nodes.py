import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool
from sentence_transformers import SentenceTransformer

from hn_search.cache_config import (
    cache_answer,
    cache_vector_search,
    get_cached_answer,
    get_cached_vector_search,
)
from hn_search.common import get_device, get_model
from hn_search.db_config import get_db_config
from hn_search.logging_config import get_logger, log_time

from .state import RAGState, SearchResult

load_dotenv()

logger = get_logger(__name__)

# Initialize connection pool (singleton)
_connection_pool = None

# Cache partition names (refreshed periodically)
_partition_cache = None
_partition_cache_timestamp = 0
_PARTITION_CACHE_TTL = 3600  # 1 hour


def get_connection_pool():
    global _connection_pool
    if _connection_pool is None:
        db_config = get_db_config()
        # Create connection string from config
        conn_string = f"host={db_config['host']} port={db_config['port']} dbname={db_config['dbname']} user={db_config['user']} password={db_config['password']}"
        _connection_pool = ConnectionPool(
            conn_string,
            min_size=2,
            max_size=20,
            max_idle=300,  # Close connections idle for 5 minutes
            max_lifetime=3600,  # Replace connections after 1 hour
            kwargs={
                "prepare_threshold": None,  # Disable prepared statements for pgvector
                "keepalives": 1,  # Enable TCP keepalive
                "keepalives_idle": 30,  # Send keepalive after 30s idle
                "keepalives_interval": 10,  # Interval between keepalives
                "keepalives_count": 5,  # Failed keepalives before declaring dead
            },
            configure=lambda conn: register_vector(
                conn
            ),  # Register vector on each connection
            check=ConnectionPool.check_connection,  # Check connection health before use
        )
    return _connection_pool


def _query_partition(pool, partition_name: str, query_embedding, n_results: int):
    """Query a single partition for nearest neighbors."""
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, clean_text, author, timestamp, type,
                           embedding <=> %s::vector AS distance
                    FROM {partition_name}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding.tolist(), query_embedding.tolist(), n_results),
                )
                return cur.fetchall()
    except Exception as e:
        logger.warning(f"Error querying partition {partition_name}: {e}")
        return []


def _get_partitions(pool):
    """
    Get list of all partition table names.

    Cached for 1 hour since partitions rarely change (only monthly).
    """
    global _partition_cache, _partition_cache_timestamp

    current_time = time.time()

    # Return cached value if still fresh
    if (
        _partition_cache
        and (current_time - _partition_cache_timestamp) < _PARTITION_CACHE_TTL
    ):
        logger.debug(
            f"Using cached partition list ({len(_partition_cache)} partitions)"
        )
        return _partition_cache

    # Fetch fresh partition list
    logger.info("Fetching partition list from database")
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = 'public'
                  AND tablename LIKE 'hn_documents_%'
                ORDER BY tablename
            """)
            partitions = [row[0] for row in cur.fetchall()]

    # Update cache
    _partition_cache = partitions
    _partition_cache_timestamp = current_time
    logger.info(f"Cached {len(partitions)} partitions")

    return partitions


def _parallel_partition_search(pool, query_embedding, n_results: int):
    """
    Query all partitions in parallel and return combined results.

    Uses ThreadPoolExecutor to query multiple partitions concurrently,
    then merges results. Much faster than sequential partition scanning.
    """
    partitions = _get_partitions(pool)
    logger.info(f"Querying {len(partitions)} partitions in parallel")

    all_results = []

    # Use max_workers based on connection pool size (leave some headroom)
    max_workers = min(15, len(partitions))  # Use up to 15 connections

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all partition queries
        future_to_partition = {
            executor.submit(
                _query_partition, pool, partition, query_embedding, n_results
            ): partition
            for partition in partitions
        }

        # Collect results as they complete
        for future in as_completed(future_to_partition):
            partition = future_to_partition[future]
            try:
                results = future.result()
                all_results.extend(results)
                logger.debug(f"Partition {partition}: {len(results)} results")
            except Exception as e:
                logger.exception(f"Exception querying partition {partition}: {e}")

    logger.info(
        f"Retrieved {len(all_results)} total results from {len(partitions)} partitions"
    )
    return all_results


def retrieve_node(state: RAGState) -> RAGState:
    query = state["query"]
    n_results = 10

    try:
        # Check cache first
        with log_time(logger, "cache lookup"):
            cached_results = get_cached_vector_search(query, n_results)

        if cached_results:
            logger.info(f"🔍 Using cached results for: {query}")
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

        # Get connection from pool
        pool = get_connection_pool()

        # Query partitions in parallel
        with log_time(logger, "parallel vector search across partitions"):
            all_results = _parallel_partition_search(pool, query_embedding, n_results)

        # Merge and sort results from all partitions
        with log_time(logger, "merging partition results"):
            # Sort by distance and take top n_results
            all_results.sort(key=lambda x: x[5])  # Sort by distance (index 5)
            results = all_results[:n_results]

            search_results = []
            cache_data = []

            for (
                doc_id,
                document,
                author,
                timestamp,
                doc_type,
                distance,
            ) in results:
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

        # Cache the results
        if cache_data:
            with log_time(logger, "caching search results"):
                cache_vector_search(query, cache_data, n_results)

        context = "\n\n---\n\n".join(
            [
                f"[{i + 1}] Author: {r['author']} ({r['timestamp']})\nLink: https://news.ycombinator.com/item?id={r['id']}\n{r['text']}"
                for i, r in enumerate(search_results)
            ]
        )

        logger.info(f"✅ Found {len(search_results)} relevant comments/articles")

        return {
            **state,
            "search_results": search_results,
            "context": context,
        }
    except Exception as e:
        error_msg = f"vector type not found in the database"
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
    with log_time(logger, "caching answer"):
        cache_answer(query, context, answer)

    logger.info("✅ Answer generated")

    return {
        **state,
        "answer": answer,
    }
