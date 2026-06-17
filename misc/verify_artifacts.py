#!/usr/bin/env python
"""Integrity check (Gate A) for the search artifacts. Safe to run mid-build (read-only).

Verifies the invariants that guarantee no partial / misaligned data:

  1. row-count agreement — codes.bin, rerank_f16.bin and docs.sqlite all imply the
     same N (and meta.json matches, if the build has finished).
  2. rowid contiguity    — rowids are exactly 1..N with no gaps or duplicate hn_ids.
  3. code↔vector alignment — for random rows, re-quantizing the stored f16 vector
     reproduces the stored 96-byte code, proving row i is the same comment in both
     files (catches any cross-file drift).
  4. (optional) --check-pg — N and a few rows match the live Postgres.

Usage:
    uv run python misc/verify_artifacts.py --artifacts rust-search/artifacts
    uv run python misc/verify_artifacts.py --artifacts rust-search/artifacts --check-pg
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

DIM = 768
CODE_BYTES = 96
F16_BYTES = DIM * 2


def fail(msg: str):
    print(f"❌ {msg}")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=Path, default=Path("rust-search/artifacts"))
    ap.add_argument("--samples", type=int, default=2000, help="random rows for alignment check")
    ap.add_argument("--check-pg", action="store_true")
    args = ap.parse_args()

    a = args.artifacts
    codes_rows = (a / "codes.bin").stat().st_size // CODE_BYTES
    f16_rows = (a / "rerank_f16.bin").stat().st_size // F16_BYTES
    codes_rem = (a / "codes.bin").stat().st_size % CODE_BYTES
    f16_rem = (a / "rerank_f16.bin").stat().st_size % F16_BYTES

    conn = sqlite3.connect(f"file:{a / 'docs.sqlite'}?mode=ro", uri=True)
    sql_rows = conn.execute("SELECT COUNT(*) FROM doc").fetchone()[0]

    print(f"codes.bin:      {codes_rows:,} rows (remainder {codes_rem}B)")
    print(f"rerank_f16.bin: {f16_rows:,} rows (remainder {f16_rem}B)")
    print(f"docs.sqlite:    {sql_rows:,} rows")

    # 1. no torn trailing record; SQLite (committed last) is the source of truth.
    # The .bin files may be a fraction of a batch AHEAD while a build is in progress;
    # they must never be BEHIND (that would mean lost vectors for committed rows).
    if codes_rem or f16_rem:
        fail("a .bin file size is not a whole number of rows (torn write)")
    if codes_rows < sql_rows or f16_rows < sql_rows:
        fail("a .bin file is BEHIND docs.sqlite — vectors missing for committed rows")
    n = sql_rows
    ahead = max(codes_rows, f16_rows) - n
    if ahead:
        print(f"… build in progress: .bin files {ahead:,} rows ahead of SQLite "
              f"(in-flight batch; the next resume trims to {n:,})")
    else:
        print("✓ all three files agree exactly (at rest)")

    # 2. contiguity + uniqueness
    mn, mx, distinct = conn.execute(
        "SELECT MIN(rowid), MAX(rowid), COUNT(DISTINCT hn_id) FROM doc"
    ).fetchone()
    if not (mn == 1 and mx == n):
        fail(f"rowids not contiguous 1..{n} (min={mn} max={mx})")
    if distinct != n:
        fail(f"duplicate hn_ids: {n - distinct} dupes")
    print("✓ row counts agree, rowids contiguous 1..N, hn_ids unique")

    # 3. code <-> vector alignment on random rows
    codes = np.memmap(a / "codes.bin", dtype=np.uint8, mode="r", shape=(n, CODE_BYTES))
    f16 = np.memmap(a / "rerank_f16.bin", dtype="<f2", mode="r", shape=(n, DIM))
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(args.samples, n), replace=False)
    vecs = np.asarray(f16[idx], dtype=np.float32)
    recomputed = np.packbits(vecs > 0, axis=1, bitorder="big")
    if not np.array_equal(recomputed, np.asarray(codes[idx])):
        fail("codes.bin does not match re-quantized rerank_f16.bin — files are misaligned")
    print(f"✓ code↔vector alignment verified on {len(idx):,} random rows")

    meta_path = a / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        status = "matches" if meta["count"] == n else f"STALE (meta={meta['count']:,}, build still running)"
        print(f"meta.json count: {meta['count']:,} — {status}")

    if args.check_pg:
        from dotenv import load_dotenv

        load_dotenv()
        import psycopg

        from hn_search.db_config import get_db_config

        pg = psycopg.connect(**get_db_config())
        pg_n = pg.execute("SELECT COUNT(*) FROM hn_documents").fetchone()[0]
        print(f"pg hn_documents: {pg_n:,} rows ({'complete' if pg_n == n else f'{n/pg_n:.1%} dumped'})")
        # spot-check a few hn_ids exist in pg with matching text
        sample = conn.execute("SELECT hn_id, clean_text FROM doc ORDER BY RANDOM() LIMIT 5").fetchall()
        mism = 0
        for hn_id, text in sample:
            row = pg.execute("SELECT clean_text FROM hn_documents WHERE id = %s", (hn_id,)).fetchone()
            if row is None or row[0] != text:
                mism += 1
        print(f"✓ {len(sample) - mism}/{len(sample)} sampled rows match pg text" if not mism
              else f"❌ {mism}/{len(sample)} sampled rows differ from pg")
        pg.close()

    conn.close()
    print("\n✅ artifacts integrity OK")


if __name__ == "__main__":
    main()
