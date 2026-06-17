#!/usr/bin/env python
"""Build the flat artifact files for the Rust brute-force search service from the
embedded parquet shards (data/embedded/*.parquet, produced by
misc/generate_embeddings_gpu.py). For a full rebuild from scratch.

Writes four **row-aligned** files into the output dir (row ``i`` is the same comment
in all of them):

  codes.bin       N x 96 bytes    sign-bit binary quantization. bit i = 1 iff
                                  embedding[i] > 0, packed MSB-first (np.packbits,
                                  bitorder='big').
  rerank_f16.bin  N x 768 x 2 B   little-endian IEEE-754 half of each embedding.
  docs.sqlite     doc(rowid INTEGER PRIMARY KEY, hn_id, clean_text, author,
                                  timestamp, type); rowid = .bin row index + 1.
  meta.json       {count, dim, code_bytes, built_at, source}

Resumable and idempotent: on restart it reconciles the three files to a consistent
row count and continues. (Daily growth goes through the service's /append, not here.)

Usage:
    uv run python misc/build_search_artifacts.py --out artifacts/
    uv run python misc/build_search_artifacts.py --out artifacts/ --limit 1000
    uv run python misc/build_search_artifacts.py --out artifacts/ --restart   # fresh
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
    """Coerce a parquet embedding cell (list/ndarray) to a float32 ndarray."""
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
    ap.add_argument("--glob", default="data/embedded/*.parquet")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=20000)
    ap.add_argument("--restart", action="store_true", help="Delete existing artifacts and start fresh")
    args = ap.parse_args()

    if args.restart:
        for name in ("codes.bin", "rerank_f16.bin", "docs.sqlite", "meta.json"):
            (args.out / name).unlink(missing_ok=True)

    print(f"Building artifacts → {args.out} from {args.glob}", file=sys.stderr)
    writer = ArtifactWriter(args.out)
    dump_parquet(writer, args.glob, args.limit, args.batch)
    meta = writer.finish("parquet")
    print(f"✅ wrote {meta['count']:,} rows to {args.out}/", file=sys.stderr)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
