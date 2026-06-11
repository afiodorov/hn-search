#!/usr/bin/env python
"""Load embedded monthly parquets into hn_documents_YYYY_MM partitions.

Reads ``data/embedded/month_YYYY_MM.parquet`` (from generate_embeddings_gpu.py)
and loads each into its matching ``hn_documents_YYYY_MM`` partition as
``halfvec(768)``, then builds the binary HNSW index on that partition. The web
path (hn_search/rag/nodes.py) queries exactly these partitions.

Bulk-loads then indexes (faster than maintaining the HNSW graph row-by-row).
Idempotent: rows use ON CONFLICT DO NOTHING and the index is created IF NOT EXISTS,
so it's safe to re-run or resume. Connects via DATABASE_URL / PG* env vars.

Usage:
    uv run python misc/load_embedded_to_partitions.py
    uv run python misc/load_embedded_to_partitions.py --in data/embedded
"""

import argparse
import os
import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from hn_search.db_config import get_db_config

load_dotenv()

DIM = 768
MONTH_RE = re.compile(r"month_(\d{4})_(\d{2})")


def get_conn():
    conn = psycopg.connect(**get_db_config())
    # The vector extension must exist before register_vector can find the type.
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def partition_for(filename: str) -> str | None:
    m = MONTH_RE.search(filename)
    return f"hn_documents_{m.group(1)}_{m.group(2)}" if m else None


def ensure_partition(conn, table: str):
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id TEXT PRIMARY KEY,
                clean_text TEXT NOT NULL,
                author TEXT,
                timestamp TEXT,
                type TEXT,
                embedding halfvec({DIM})
            )
            """
        )
    conn.commit()


def build_index(conn, table: str):
    with conn.cursor() as cur:
        # Managed Postgres (Railway) has a small shared-memory segment, so pgvector's
        # PARALLEL HNSW build fails allocating its DSM ("No space left on device").
        # Build serially (no parallel workers -> no shared-memory segment); the binary
        # index is tiny, so backend-local maintenance_work_mem is plenty.
        cur.execute("SET max_parallel_maintenance_workers = 0")
        cur.execute(
            f"SET maintenance_work_mem = '{os.getenv('MAINTENANCE_WORK_MEM', '256MB')}'"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {table}_bin_idx ON {table} "
            f"USING hnsw ((binary_quantize(embedding)::bit({DIM})) bit_hamming_ops)"
        )
    conn.commit()


def load_file(conn, path: Path, table: str, batch_size: int) -> int:
    df = pd.read_parquet(path)
    df = df[df["clean_text"].notna() & (df["clean_text"].astype(str).str.len() > 0)]
    inserted = 0
    with conn.cursor() as cur:
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start : start + batch_size]
            records = [
                (
                    str(r["id"]),
                    r["clean_text"],
                    str(r["author"]),
                    str(r["timestamp"]),
                    str(r["type"]),
                    np.asarray(r["embedding"], dtype=np.float32),
                )
                for _, r in batch.iterrows()
            ]
            cur.executemany(
                f"INSERT INTO {table} (id, clean_text, author, timestamp, type, embedding) "
                f"VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                records,
            )
            inserted += len(records)
    conn.commit()
    return inserted


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in", dest="in_dir", default="data/embedded", help="dir with month_*.parquet"
    )
    ap.add_argument("--glob", default="month_*.parquet")
    ap.add_argument("--batch-size", type=int, default=1000)
    args = ap.parse_args()

    files = sorted(glob(str(Path(args.in_dir) / args.glob)))
    if not files:
        raise FileNotFoundError(f"No files matching {args.glob} in {args.in_dir}")

    cfg = get_db_config()
    print(f"Connecting to {cfg['host']}:{cfg['port']}/{cfg['dbname']}")
    conn = get_conn()

    grand_total = 0
    for idx, path in enumerate(map(Path, files), 1):
        table = partition_for(path.name)
        if not table:
            print(f"[{idx}/{len(files)}] ⚠️ {path.name}: no YYYY_MM in name, skipping")
            continue
        print(f"[{idx}/{len(files)}] {path.name} -> {table}")
        ensure_partition(conn, table)
        n = load_file(conn, path, table, args.batch_size)
        build_index(conn, table)
        grand_total += n
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {table}")
            cnt = cur.fetchone()[0]
        print(f"  ✅ inserted {n:,} (partition now {cnt:,} rows, binary index built)")

    conn.close()
    print(
        f"\n✅ Loaded {grand_total:,} comments across {len(files)} partition file(s)."
    )


if __name__ == "__main__":
    main()
