import sys

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from hn_search.cache_config import (
    cache_vector_search,
    get_cached_vector_search,
)
from hn_search.common import get_device
from hn_search.db_config import get_db_config

load_dotenv()


def query(
    query_text: str,
    n_results: int = 10,
    db_host: str = None,
    db_port: int = None,
    db_name: str = None,
    db_user: str = None,
    db_password: str = None,
):
    # Check cache first
    cached_results = get_cached_vector_search(query_text, n_results)
    if cached_results:
        print("Using cached results...", file=sys.stderr)
        for i, result in enumerate(cached_results, 1):
            print(f"=== Result {i} (distance: {result['distance']:.4f}) ===")
            print(f"ID: {result['id']}")
            print(f"Author: {result['author']}")
            print(f"Date: {result['timestamp']}")
            print(f"Type: {result['type']}")
            print(f"Text: {result['text']}")
            print()
            sys.stdout.flush()
        return

    device = get_device()
    print(f"Loading embedding model on {device}...", file=sys.stderr)
    model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2", device=device
    )

    print("Encoding query...", file=sys.stderr)
    query_embedding = model.encode([query_text])[0]

    print("Querying PostgreSQL...", file=sys.stderr)
    # Use Railway/environment variables if individual params not provided
    if not all([db_host, db_port, db_name, db_user, db_password]):
        db_config = get_db_config()
    else:
        db_config = {
            "host": db_host,
            "port": db_port,
            "dbname": db_name,
            "user": db_user,
            "password": db_password,
        }

    conn = psycopg.connect(**db_config)
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

        print("Fetching results...\n", file=sys.stderr)
        sys.stdout.flush()

        results = cur.fetchall()

        # Prepare results for caching
        cache_data = []

        for i, (doc_id, document, author, timestamp, doc_type, distance) in enumerate(
            results, 1
        ):
            print(f"=== Result {i} (distance: {distance:.4f}) ===")
            print(f"ID: {doc_id}")
            print(f"Author: {author}")
            print(f"Date: {timestamp}")
            print(f"Type: {doc_type}")
            print(f"Text: {document}")
            print()
            sys.stdout.flush()

            # Add to cache data
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
            cache_vector_search(query_text, cache_data, n_results)

    conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m hn_search.query <query_text> [n_results]")
        sys.exit(1)

    query_text = sys.argv[1]
    n_results = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    query(query_text, n_results)
