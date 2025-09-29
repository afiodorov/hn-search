from glob import glob
from pathlib import Path

import pandas as pd
import psycopg
from pgvector.psycopg import register_vector


def init_db_from_precomputed(
    embeddings_dir: str = "embeddings",
    db_host: str = "localhost",
    db_port: int = 5432,
    db_name: str = "hn_search",
    db_user: str = "postgres",
    db_password: str = "postgres",
    test_mode: bool = False,
):
    conn = psycopg.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
    )
    register_vector(conn)

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

        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM processed_files WHERE filename = %s", (file_name,)
            )
            if cur.fetchone():
                print(
                    f"\n[{file_idx}/{len(parquet_files)}] Skipping {file_name} (already done)"
                )
                continue

        print(f"\n[{file_idx}/{len(parquet_files)}] Loading {file_name}")

        df = pd.read_parquet(parquet_file)

        if test_mode:
            df = df.head(100)
            print(f"  TEST MODE: Limited to {len(df)} rows")

        print(f"  Loaded {len(df)} rows")

        batch_size = 1000
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
                conn.commit()

            total_loaded += len(records)
            print(f"  Total loaded: {total_loaded}")

        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO processed_files (filename) VALUES (%s) ON CONFLICT DO NOTHING",
                (file_name,),
            )
            conn.commit()
        print(f"  Marked {file_name} as done")

    print(f"\n✅ Successfully loaded {total_loaded} documents into PostgreSQL")

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM hn_documents")
        count = cur.fetchone()[0]
        print(f"Total documents in database: {count}")

    conn.close()


if __name__ == "__main__":
    import sys

    test_mode = "--test" in sys.argv
    init_db_from_precomputed(test_mode=test_mode)
