#!/usr/bin/env python
"""One-off (step 1 of 2): fetch the id -> parent mapping from BigQuery into a
small local SQLite file, meant to be rsynced to wherever docs.sqlite actually
lives and applied with backfill_parent_ids_apply.py.

Run this on a machine with plenty of RAM (e.g. your laptop), not on the
resource-constrained VPS that also runs the live search service — a naive
`to_dataframe()` over the full ~12M-row result was confirmed to OOM-kill a
7.6GB box shared with the Rust service and Airflow. Streams via the paginated
row iterator regardless, so memory stays bounded to one page either way.

Usage:
    uv run --extra dev python misc/backfill_parent_ids_fetch.py --out parent_map.sqlite
    rsync -avP parent_map.sqlite root@your-vps:/root/hn-search/
"""

import argparse
import sqlite3
import time
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="parent_map.sqlite")
    parser.add_argument(
        "--project", default=None, help="GCP project for BigQuery billing"
    )
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.unlink(missing_ok=True)

    client = bigquery.Client(project=args.project)
    print(f"💰 Using GCP project: {client.project}")

    # Same filters as fetch_and_embed_new_comments.py/fetch_historical.py, so this
    # matches exactly what's actually in docs.sqlite — without them this pulls
    # every comment ever posted (including dead/deleted/textless ones our own
    # indexing pipeline already excludes), plus the corpus only covers 2023+
    # (fetch_historical.py's --start default) while HN itself goes back to 2006,
    # so skipping the date bound alone pulls ~1.5-2x more rows than needed.
    query = """
    SELECT CAST(id AS STRING) AS hn_id, CAST(parent AS STRING) AS parent_id
    FROM `bigquery-public-data.hacker_news.full`
    WHERE type = 'comment' AND parent IS NOT NULL
      AND dead IS NOT TRUE AND deleted IS NOT TRUE AND text IS NOT NULL
      AND timestamp >= TIMESTAMP('2023-01-01')
    """
    print("Running BigQuery query for id -> parent mapping (~12.7M rows expected)...")
    t0 = time.time()
    query_job = client.query(query)

    conn = sqlite3.connect(out_path)
    conn.execute("CREATE TABLE parent_map (hn_id TEXT PRIMARY KEY, parent_id TEXT)")

    buf = []
    total = 0
    for row in query_job.result(page_size=args.chunk_size):
        buf.append((row.hn_id, row.parent_id))
        if len(buf) >= args.chunk_size:
            conn.executemany(
                "INSERT INTO parent_map (hn_id, parent_id) VALUES (?, ?)", buf
            )
            conn.commit()
            total += len(buf)
            print(f"  {total:,} fetched")
            buf.clear()
    if buf:
        conn.executemany("INSERT INTO parent_map (hn_id, parent_id) VALUES (?, ?)", buf)
        conn.commit()
        total += len(buf)
    conn.close()

    print(f"✅ Fetched {total:,} rows in {time.time() - t0:.0f}s -> {out_path}")
    print(f"   ({out_path.stat().st_size / 1e6:.0f} MB)")
    print(
        "Next: rsync this file to the box, then run backfill_parent_ids_apply.py there."
    )


if __name__ == "__main__":
    main()
