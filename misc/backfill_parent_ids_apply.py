#!/usr/bin/env python
"""One-off (step 2 of 2): apply a parent_map.sqlite (built by
backfill_parent_ids_fetch.py) to the real docs.sqlite's parent_id column.

Lightweight and memory-cheap — no BigQuery client, no pandas, just an ATTACHed
SQLite database and a single indexed UPDATE...FROM. Safe to run on the same
resource-constrained box that runs the live search service. Metadata-only — no
reembedding, no changes to codes.bin/rerank_f16.bin. Safe to re-run.

**Touches production data.** The Rust service must be restarted after this to
pick up the change — it holds its own open connection and won't see the
update otherwise.

Usage:
    uv run python misc/backfill_parent_ids_apply.py --db /var/lib/hnsearch/current/docs.sqlite --map parent_map.sqlite
"""

import argparse
import sqlite3
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="path to docs.sqlite")
    parser.add_argument("--map", required=True, help="path to parent_map.sqlite")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode=WAL")
    # Idempotent — same index the Rust service creates on startup.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hn_id ON doc(hn_id)")
    conn.execute("ATTACH DATABASE ? AS pm", (args.map,))

    print("Applying UPDATE ... FROM (be patient)...")
    t0 = time.time()
    cur = conn.execute(
        "UPDATE doc SET parent_id = pm.parent_map.parent_id "
        "FROM pm.parent_map WHERE doc.hn_id = pm.parent_map.hn_id"
    )
    conn.commit()
    print(f"Updated {cur.rowcount:,} rows in {time.time() - t0:.0f}s")

    conn.execute("DETACH DATABASE pm")

    total = conn.execute("SELECT COUNT(*) FROM doc").fetchone()[0]
    with_parent = conn.execute(
        "SELECT COUNT(*) FROM doc WHERE parent_id IS NOT NULL"
    ).fetchone()[0]
    print(f"✅ Done. {with_parent:,}/{total:,} rows now have parent_id set.")
    print("Restart the Rust service to pick up the change: systemctl restart hnsearch")
    conn.close()


if __name__ == "__main__":
    main()
