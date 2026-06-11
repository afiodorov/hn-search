#!/usr/bin/env python
"""Make `hn_documents` a partitioned parent and attach the monthly tables.

Creates `hn_documents` as PARTITION BY RANGE (timestamp) (if it doesn't already
exist) and ATTACHes every standalone `hn_documents_YYYY_MM` table as a monthly
range partition. After this, the web path queries the single parent table and
Postgres fans out across each partition's binary HNSW index (a MergeAppend) — no
application-side fan-out needed.

Idempotent: tables already attached are skipped, so re-run it whenever new months
are added (e.g. after the incremental fetcher creates a new monthly table).
Connects via DATABASE_URL / PG* env vars.

Usage:
    uv run python misc/attach_partitions.py
"""

import re

import psycopg
from dotenv import load_dotenv

from hn_search.db_config import get_db_config

load_dotenv()

DIM = 768
MONTH_RE = re.compile(r"^hn_documents_(\d{4})_(\d{2})$")


def get_conn():
    conn = psycopg.connect(**get_db_config(), autocommit=True)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    return conn


def relkind(cur, name: str):
    cur.execute(
        "SELECT relkind FROM pg_class WHERE relname = %s "
        "AND relnamespace = 'public'::regnamespace",
        (name,),
    )
    row = cur.fetchone()
    return row[0] if row else None


def month_bounds(year: int, month: int) -> tuple[str, str]:
    lo = f"{year:04d}-{month:02d}-01"
    ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
    return lo, f"{ny:04d}-{nm:02d}-01"


def main():
    conn = get_conn()
    cur = conn.cursor()

    kind = relkind(cur, "hn_documents")
    if kind == "r":
        raise SystemExit(
            "❌ `hn_documents` already exists as a regular table. Drop or rename it "
            "first — a partitioned parent can't share its name."
        )
    if kind is None:
        print(
            "Creating partitioned parent `hn_documents` (PARTITION BY RANGE timestamp)"
        )
        cur.execute(
            f"""
            CREATE TABLE hn_documents (
                id TEXT,
                clean_text TEXT NOT NULL,
                author TEXT,
                timestamp TEXT,
                type TEXT,
                embedding halfvec({DIM})
            ) PARTITION BY RANGE (timestamp)
            """
        )
    else:
        print("Parent `hn_documents` already partitioned ✓")

    # standalone monthly tables not yet attached as partitions
    cur.execute(
        """
        SELECT c.relname
        FROM pg_class c
        WHERE c.relnamespace = 'public'::regnamespace
          AND c.relkind = 'r'
          AND c.relispartition = false
          AND c.relname ~ '^hn_documents_[0-9]{4}_[0-9]{2}$'
        ORDER BY c.relname
        """
    )
    tables = [r[0] for r in cur.fetchall()]
    if not tables:
        print("No unattached hn_documents_YYYY_MM tables found.")
        # still report current partition count
    attached = 0
    for t in tables:
        m = MONTH_RE.match(t)
        lo, hi = month_bounds(int(m.group(1)), int(m.group(2)))
        print(f"  attaching {t}  FOR VALUES FROM ('{lo}') TO ('{hi}') ...")
        cur.execute(
            f"ALTER TABLE hn_documents ATTACH PARTITION {t} "
            f"FOR VALUES FROM ('{lo}') TO ('{hi}')"
        )
        attached += 1

    cur.execute(
        "SELECT count(*) FROM pg_inherits WHERE inhparent = 'hn_documents'::regclass"
    )
    total = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM hn_documents")
    rows = cur.fetchone()[0]
    conn.close()
    print(
        f"\n✅ Attached {attached} new partition(s). "
        f"hn_documents now has {total} partitions, {rows:,} rows total."
    )


if __name__ == "__main__":
    main()
