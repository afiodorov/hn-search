#!/usr/bin/env python
"""Migrate hn_documents* tables to the RAM-light search layout and build indexes.

For every table named ``hn_documents`` or ``hn_documents_YYYY_MM`` this script:

  1. converts the ``embedding`` column to ``halfvec(768)`` if it's still
     ``vector(768)`` (halves on-disk vector storage; binary_quantize works on both),
  2. drops any old float HNSW index (``*_cosine_ops`` — the ~37 GB RAM hog),
  3. creates the binary-quantized HNSW index
     ``hnsw((binary_quantize(embedding)::bit(768)) bit_hamming_ops)`` (~3.7 GB at 9.4M).

It is idempotent — safe to re-run. Connects via DATABASE_URL / PG* env vars
(see hn_search.db_config), so it works directly against Railway.

Usage:
    uv run python misc/build_binary_index.py            # build/upgrade everything
    uv run python misc/build_binary_index.py --concurrent  # CREATE INDEX CONCURRENTLY
    MAINTENANCE_WORK_MEM=2GB uv run python misc/build_binary_index.py
"""

import argparse
import os

import psycopg
from dotenv import load_dotenv

from hn_search.db_config import get_db_config

load_dotenv()

DIM = 768


def find_tables(cur) -> list[str]:
    cur.execute(
        """
        SELECT tablename FROM pg_tables
        WHERE schemaname = 'public'
          AND (tablename = 'hn_documents' OR tablename ~ '^hn_documents_[0-9]{4}_[0-9]{2}$')
        ORDER BY tablename
        """
    )
    return [r[0] for r in cur.fetchall()]


def embedding_type(cur, table: str) -> str | None:
    cur.execute(
        """
        SELECT atttypid::regtype::text
        FROM pg_attribute
        WHERE attrelid = %s::regclass AND attname = 'embedding' AND NOT attisdropped
        """,
        (table,),
    )
    row = cur.fetchone()
    return row[0] if row else None


def float_indexes(cur, table: str) -> list[str]:
    """Old HNSW/IVFFlat indexes built on the raw float vector (cosine/L2/ip ops)."""
    cur.execute(
        """
        SELECT indexname FROM pg_indexes
        WHERE schemaname = 'public' AND tablename = %s
          AND (indexdef LIKE '%%_cosine_ops%%'
               OR indexdef LIKE '%%_l2_ops%%'
               OR indexdef LIKE '%%_ip_ops%%')
        """,
        (table,),
    )
    return [r[0] for r in cur.fetchall()]


def pretty_size(cur, relation: str) -> str:
    cur.execute(
        "SELECT pg_size_pretty(pg_total_relation_size(%s::regclass))", (relation,)
    )
    return cur.fetchone()[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--concurrent",
        action="store_true",
        help="Use CREATE INDEX CONCURRENTLY (no write lock; slower, can't run in a txn)",
    )
    args = ap.parse_args()

    mwm = os.getenv("MAINTENANCE_WORK_MEM", "1GB")

    cfg = get_db_config()
    print(f"Connecting to {cfg['host']}:{cfg['port']}/{cfg['dbname']}")
    # autocommit so CREATE INDEX CONCURRENTLY works and each DDL stands alone.
    conn = psycopg.connect(**cfg, autocommit=True)
    cur = conn.cursor()
    cur.execute(f"SET maintenance_work_mem = '{mwm}'")
    try:
        cur.execute("SET max_parallel_maintenance_workers = 4")
    except Exception:
        pass

    tables = find_tables(cur)
    if not tables:
        print("No hn_documents* tables found.")
        return
    print(f"Found {len(tables)} table(s): {', '.join(tables)}\n")

    concurrently = "CONCURRENTLY " if args.concurrent else ""

    for table in tables:
        coltype = embedding_type(cur, table)
        print(f"▶ {table}  (embedding: {coltype})")

        # 1. vector -> halfvec (skip if already halfvec or absent)
        if coltype == "vector":
            print(f"    converting embedding -> halfvec({DIM}) (table rewrite)...")
            cur.execute(
                f"ALTER TABLE {table} ALTER COLUMN embedding "
                f"TYPE halfvec({DIM}) USING embedding::halfvec({DIM})"
            )

        # 2. drop old float indexes (the RAM hogs we're replacing)
        for idx in float_indexes(cur, table):
            print(f"    dropping old float index {idx}")
            cur.execute(f"DROP INDEX {concurrently}IF EXISTS {idx}")

        # 3. create the binary HNSW index
        idx_name = f"{table}_bin_idx"
        print(f"    creating binary HNSW index {idx_name} ...")
        cur.execute(
            f"CREATE INDEX {concurrently}IF NOT EXISTS {idx_name} ON {table} "
            f"USING hnsw ((binary_quantize(embedding)::bit({DIM})) bit_hamming_ops)"
        )

        print(
            f"    ✅ {table}: total size now {pretty_size(cur, table)}, "
            f"binary index {pretty_size(cur, idx_name)}\n"
        )

    conn.close()
    print("✅ All tables migrated to halfvec + binary HNSW index.")


if __name__ == "__main__":
    main()
