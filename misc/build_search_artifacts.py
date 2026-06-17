#!/usr/bin/env python
"""Build the flat artifact files for the Rust brute-force search service.

Streams ``hn_documents`` from the live Postgres (the one-time bootstrap source) and
writes four **row-aligned** files into the output dir (row ``i`` is the same comment
in all of them):

  codes.bin       N x 96 bytes    sign-bit binary quantization. bit i = 1 iff
                                  embedding[i] > 0, packed MSB-first (np.packbits,
                                  bitorder='big'), matching pgvector binary_quantize.
  rerank_f16.bin  N x 768 x 2 B   little-endian IEEE-754 half of each embedding.
  docs.sqlite     doc(rowid INTEGER PRIMARY KEY, hn_id, clean_text, author,
                                  timestamp, type); rowid = .bin row index + 1.
  meta.json       {count, dim, code_bytes, built_at, source}

Rows are streamed ``ORDER BY id::bigint``, so the build is **resumable and
idempotent**: on restart it reconciles the three files to a consistent row count and
continues from ``id > last_written_id``. Re-running a finished build is a no-op.

Usage:
    uv run python misc/build_search_artifacts.py --out artifacts/        # full dump
    uv run python misc/build_search_artifacts.py --out artifacts/ --limit 1000
    uv run python misc/build_search_artifacts.py --out artifacts/ --restart   # fresh
    uv run python misc/build_search_artifacts.py --source parquet --glob 'data/embedded/*.parquet'
"""

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DIM = 768
CODE_BYTES = DIM // 8  # 96
F16_BYTES = DIM * 2  # 1536


def to_vec(emb) -> np.ndarray:
    """Coerce a pgvector Vector/HalfVector (or list) to a float32 ndarray."""
    if hasattr(emb, "to_numpy"):
        return emb.to_numpy().astype(np.float32, copy=False)
    return np.asarray(emb, dtype=np.float32)


def quantize_block(vecs: np.ndarray) -> np.ndarray:
    """(M, 768) float -> (M, 96) uint8 sign-bit codes, MSB-first per byte."""
    bits = (vecs > 0).astype(np.uint8)
    return np.packbits(bits, axis=1, bitorder="big")


def f16_block(vecs: np.ndarray) -> bytes:
    """(M, 768) float -> raw little-endian float16 bytes, row-major."""
    return np.ascontiguousarray(vecs, dtype="<f2").tobytes()


class ArtifactWriter:
    """Append-only, resumable writer for the four artifact files.

    Writes vectors to the .bin files *before* committing the matching SQLite rows, so
    after a crash the .bin files are >= SQLite; on open we reconcile all three down to
    the common committed count.
    """

    def __init__(self, out: Path):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(out / "docs.sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS doc (rowid INTEGER PRIMARY KEY, hn_id TEXT, "
            "clean_text TEXT, author TEXT, timestamp TEXT, type TEXT)"
        )
        self.count, self.last_id = self._reconcile()
        self.codes_f = open(out / "codes.bin", "r+b" if (out / "codes.bin").exists() else "wb")
        self.f16_f = open(out / "rerank_f16.bin", "r+b" if (out / "rerank_f16.bin").exists() else "wb")
        self.codes_f.seek(self.count * CODE_BYTES)
        self.f16_f.seek(self.count * F16_BYTES)

    def _reconcile(self) -> tuple[int, int | None]:
        """Trim the three files to a common committed row count; return (count, last_id)."""
        codes_rows = _rows(self.out / "codes.bin", CODE_BYTES)
        f16_rows = _rows(self.out / "rerank_f16.bin", F16_BYTES)
        sql_rows = self.db.execute("SELECT COUNT(*) FROM doc").fetchone()[0]
        count = min(codes_rows, f16_rows, sql_rows)

        _truncate(self.out / "codes.bin", count * CODE_BYTES)
        _truncate(self.out / "rerank_f16.bin", count * F16_BYTES)
        if sql_rows > count:
            self.db.execute("DELETE FROM doc WHERE rowid > ?", (count,))
            self.db.commit()

        last_id = None
        if count > 0:
            row = self.db.execute(
                "SELECT hn_id FROM doc WHERE rowid = ?", (count,)
            ).fetchone()
            last_id = row[0] if row else None  # text id (rows stream in id-text order)
            print(f"↻ resuming from {count:,} rows (last id={last_id})", file=sys.stderr)
        return count, last_id

    def write_batch(self, rows: list[tuple], vecs: np.ndarray):
        """rows: list of (hn_id, clean_text, author, timestamp, type). vecs: (M,768)."""
        self.codes_f.write(np.ascontiguousarray(quantize_block(vecs), dtype=np.uint8).tobytes())
        self.f16_f.write(f16_block(vecs))
        self.codes_f.flush()
        self.f16_f.flush()
        records = [(self.count + i + 1, *r) for i, r in enumerate(rows)]
        self.db.executemany(
            "INSERT INTO doc (rowid, hn_id, clean_text, author, timestamp, type) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            records,
        )
        self.db.commit()
        self.count += len(rows)
        if rows:
            self.last_id = rows[-1][0]  # text id

    def finish(self, source: str):
        self.codes_f.close()
        self.f16_f.close()
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_doc_hn_id ON doc (hn_id)")
        self.db.commit()
        self.db.close()
        meta = {
            "count": self.count,
            "dim": DIM,
            "code_bytes": CODE_BYTES,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
        }
        (self.out / "meta.json").write_text(json.dumps(meta, indent=2))
        return meta


def _rows(path: Path, row_bytes: int) -> int:
    return path.stat().st_size // row_bytes if path.exists() else 0


def _truncate(path: Path, size: int):
    if path.exists() and path.stat().st_size != size:
        with open(path, "r+b") as f:
            f.truncate(size)


class Progress:
    """Throughput + ETA reporter."""

    def __init__(self, total: int | None, done: int):
        self.total = total
        self.start = time.time()
        self.start_done = done
        self.last = 0.0
        self.retries = 0  # consecutive reconnect attempts since last committed batch

    def ok(self):
        """Real progress was committed — reset the reconnect budget."""
        self.retries = 0

    def update(self, done: int):
        now = time.time()
        if now - self.last < 2 and (self.total is None or done < self.total):
            return
        self.last = now
        elapsed = now - self.start
        rate = (done - self.start_done) / elapsed if elapsed > 0 else 0
        msg = f"  {done:,} rows | {rate:,.0f} rows/s"
        if self.total:
            pct = 100 * done / self.total
            eta = (self.total - done) / rate if rate > 0 else 0
            msg += f" | {pct:.1f}% | ETA {eta/60:.0f}m"
        print(msg, file=sys.stderr, flush=True)


def _stream_once(writer, conn, total, batch, limit, progress) -> int:
    """Stream rows from a fresh connection, resuming from writer.last_id. Returns
    rows skipped (NULL embedding). Resets the caller's retry counter via progress."""
    # ORDER BY id (text) streams via Merge Append over the per-partition PK indexes —
    # no global Sort, so nothing spills to the server's pgsql_tmp. (ORDER BY id::bigint
    # can't use the text index → full 18 GB sort → server DiskFull / hang.) Order is
    # purely positional for the artifact; resume is a text-id keyset.
    params: list = []
    where = ""
    if writer.last_id is not None:
        where = "WHERE id > %s"
        params.append(writer.last_id)
    sql = (
        f"SELECT id, clean_text, author, timestamp, type, embedding "
        f"FROM hn_documents {where} ORDER BY id"
    )
    if limit:
        sql += f" LIMIT {int(limit - writer.count)}"

    skipped = 0
    with conn.cursor(name="artifact_dump") as cur:
        cur.itersize = batch
        cur.execute(sql, params)
        rows, vecs = [], []
        for hn_id, clean_text, author, ts, typ, emb in cur:
            if emb is None:
                skipped += 1
                continue
            rows.append((str(hn_id), clean_text, author, str(ts), typ))
            vecs.append(to_vec(emb))
            if len(rows) >= batch:
                writer.write_batch(rows, np.vstack(vecs))
                rows, vecs = [], []
                progress.update(writer.count)
                progress.ok()  # committed progress → reset retry budget
        if rows:
            writer.write_batch(rows, np.vstack(vecs))
            progress.update(writer.count)
    return skipped


def dump_pg(writer: ArtifactWriter, limit: int | None, batch: int, max_retries: int = 8):
    import time

    import psycopg
    from pgvector.psycopg import register_vector

    from hn_search.db_config import get_db_config

    with psycopg.connect(**get_db_config()) as c0:
        total = c0.execute("SELECT COUNT(*) FROM hn_documents").fetchone()[0]
    if limit:
        total = min(total, limit)
    print(f"target: {total:,} rows (have {writer.count:,})", file=sys.stderr)
    if writer.count >= total:
        print("✓ already complete", file=sys.stderr)
        return
    progress = Progress(total, writer.count)

    # Auto-reconnect: a dropped connection just resumes from writer.last_id (which
    # advances per committed batch). The retry budget resets whenever real progress is
    # made, so transient blips over a long dump don't accumulate toward the cap.
    skipped = 0
    while writer.count < total:
        try:
            conn = psycopg.connect(**get_db_config())
            register_vector(conn)
            try:
                skipped += _stream_once(writer, conn, total, batch, limit, progress)
            finally:
                conn.close()
            break  # cursor drained without error → done
        except (psycopg.OperationalError, psycopg.errors.AdminShutdown) as e:
            progress.retries += 1
            if progress.retries > max_retries:
                print(f"✗ giving up after {max_retries} reconnects; re-run to resume", file=sys.stderr)
                raise
            wait = min(30, 2**progress.retries)
            print(
                f"⚠️  connection lost at {writer.count:,} rows ({type(e).__name__}); "
                f"reconnecting in {wait}s (attempt {progress.retries}/{max_retries})",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)
    if skipped:
        print(f"⚠️  skipped {skipped} rows with NULL embedding", file=sys.stderr)


def dump_parquet(writer: ArtifactWriter, glob: str, limit: int | None, batch: int):
    import pandas as pd

    files = sorted(Path().glob(glob))
    if not files:
        sys.exit(f"no parquet files matched {glob!r}")
    seen_max = str(writer.last_id) if writer.last_id is not None else None
    progress = Progress(limit, writer.count)
    for fp in files:
        df = pd.read_parquet(fp)
        df = df.assign(_id_text=df["id"].astype(str)).sort_values("_id_text")
        if seen_max is not None:
            df = df[df["_id_text"] > seen_max]
        for start in range(0, len(df), batch):
            chunk = df.iloc[start : start + batch]
            if limit is not None and writer.count >= limit:
                progress.update(writer.count)
                return
            if limit is not None:
                chunk = chunk.iloc[: limit - writer.count]
            rows = [
                (str(r.id), r.clean_text, str(r.author), str(r.timestamp), str(r.type))
                for r in chunk.itertuples()
            ]
            vecs = np.vstack([to_vec(e) for e in chunk["embedding"]])
            writer.write_batch(rows, vecs)
            progress.update(writer.count)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts", type=Path)
    ap.add_argument("--source", choices=["pg", "parquet"], default="pg")
    ap.add_argument("--glob", default="data/embedded/*.parquet")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=20000)
    ap.add_argument("--restart", action="store_true", help="Delete existing artifacts and start fresh")
    args = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv()

    if args.restart:
        for name in ("codes.bin", "rerank_f16.bin", "docs.sqlite", "meta.json"):
            (args.out / name).unlink(missing_ok=True)

    print(f"Building artifacts → {args.out} (source={args.source})", file=sys.stderr)
    writer = ArtifactWriter(args.out)
    if args.source == "pg":
        dump_pg(writer, args.limit, args.batch)
    else:
        dump_parquet(writer, args.glob, args.limit, args.batch)
    meta = writer.finish(args.source)
    print(f"✅ wrote {meta['count']:,} rows to {args.out}/", file=sys.stderr)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
