import sys
from glob import glob
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from hn_search.db_config import get_db_config

load_dotenv()


def get_connection():
    """Get a fresh database connection with vector registration"""
    db_config = get_db_config()
    conn = psycopg.connect(**db_config)
    register_vector(conn)
    return conn


def init_db_from_precomputed(
    embeddings_dir: str = "embeddings",
    test_mode: bool = False,
):
    db_config = get_db_config()
    print(
        f"Connecting to database at {db_config['host']}:{db_config['port']}/{db_config['dbname']}"
    )

    # Initial connection for schema setup
    conn = get_connection()

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS hn_documents (
                id TEXT PRIMARY KEY,
                clean_text TEXT NOT NULL,
                author TEXT,
                timestamp TEXT,
                type TEXT,
                embedding vector(768)
            )
        """)
        conn.commit()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS processed_files (
                filename TEXT PRIMARY KEY,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

    conn.close()

    embeddings_path = Path(embeddings_dir)
    if not embeddings_path.is_absolute():
        embeddings_path = Path.cwd() / embeddings_path

    parquet_files = sorted(glob(str(embeddings_path / "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {embeddings_dir}")

    if test_mode:
        parquet_files = parquet_files[:1]
        print(f"TEST MODE: Processing only first file")

    print(f"Found {len(parquet_files)} parquet files with precomputed embeddings")

    total_loaded = 0

    for file_idx, parquet_file in enumerate(parquet_files, 1):
        file_name = Path(parquet_file).name

        # Fresh connection for each file
        print(f"\n[{file_idx}/{len(parquet_files)}] Processing {file_name}")

        max_retries = 3
        for retry in range(max_retries):
            try:
                conn = get_connection()

                # Check if already processed
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT 1 FROM processed_files WHERE filename = %s",
                        (file_name,),
                    )
                    if cur.fetchone():
                        print(f"  Skipping {file_name} (already done)")
                        conn.close()
                        break

                print(f"  Loading parquet file...")
                df = pd.read_parquet(parquet_file)

                if test_mode:
                    df = df.head(100)
                    print(f"  TEST MODE: Limited to {len(df)} rows")

                print(f"  Processing {len(df)} rows")

                # Start transaction for this file
                conn.execute("BEGIN")

                batch_size = 1000
                file_loaded = 0
                for batch_start in range(0, len(df), batch_size):
                    batch_df = df.iloc[batch_start : batch_start + batch_size]

                    records = [
                        (
                            str(row["id"]),
                            row["clean_text"],
                            str(row["author"]),
                            str(row["timestamp"]),
                            str(row["type"]),
                            row["embedding"].tolist(),
                        )
                        for _, row in batch_df.iterrows()
                    ]

                    with conn.cursor() as cur:
                        cur.executemany(
                            """
                            INSERT INTO hn_documents (id, clean_text, author, timestamp, type, embedding)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO NOTHING
                            """,
                            records,
                        )

                    file_loaded += len(records)
                    total_loaded += len(records)
                    print(f"    Batch: {file_loaded}/{len(df)} | Total: {total_loaded}")

                # Mark file as processed
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO processed_files (filename) VALUES (%s) ON CONFLICT DO NOTHING",
                        (file_name,),
                    )

                # Commit entire file at once
                conn.commit()
                print(f"  ✅ Committed {file_name} ({file_loaded} records)")
                conn.close()
                break  # Success - exit retry loop

            except Exception as e:
                print(f"  ❌ Error (attempt {retry + 1}/{max_retries}): {e}")
                try:
                    conn.rollback()
                    conn.close()
                except:
                    pass  # Connection might already be closed

                if retry < max_retries - 1:
                    print(f"  🔄 Retrying in 5 seconds...")
                    import time

                    time.sleep(5)
                else:
                    print(
                        f"  ❌ Failed after {max_retries} attempts - skipping {file_name}"
                    )
                    continue

    print(f"\n✅ Successfully loaded {total_loaded} documents into PostgreSQL")

    # Final count with fresh connection
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM hn_documents")
        count = cur.fetchone()[0]
        print(f"Total documents in database: {count}")
    conn.close()


if __name__ == "__main__":
    test_mode = "--test" in sys.argv
    init_db_from_precomputed(test_mode=test_mode)
