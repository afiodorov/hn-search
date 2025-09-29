import sys

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from hn_search.common import get_device
from hn_search.db_config import get_db_config


def query(
    query_text: str,
    n_results: int = 10,
    db_host: str = None,
    db_port: int = None,
    db_name: str = None,
    db_user: str = None,
    db_password: str = None,
):
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

    conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m hn_search.query <query_text> [n_results]")
        sys.exit(1)

    query_text = sys.argv[1]
    n_results = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    query(query_text, n_results)
