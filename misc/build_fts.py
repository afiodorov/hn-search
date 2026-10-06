#!/usr/bin/env python
"""Build the full-text index (`fts.sqlite`) off-box, for shipping to the search box.

The search box has 8 GB of RAM and 4 cores and is busy serving; building an
FTS5 index over 13M comments there would evict the vector index from page cache
for the length of the build. So the text is read once, sequentially, at idle
I/O priority, streamed here over ssh, and indexed on this machine. Nothing is
written on the box until the finished file is shipped.

The index is contentless and keyed by `doc.rowid` (see rust-search/src/fts.rs),
so it is valid for the `docs.sqlite` it was read from, and for any later
version of it that only appended rows: the service indexes rows past the
file's `fts_meta.max_rowid` itself on startup. After a full artifact rebuild
(new rowids), build it again from the new docs.sqlite.

    uv run python misc/build_fts.py --remote hnsearch --out data/fts/fts.sqlite
    uv run python misc/build_fts.py --docs artifacts/docs.sqlite --out artifacts/fts.sqlite
    uv run python misc/build_fts.py --remote hnsearch --limit 50000 --out /tmp/x.sqlite

Ship (atomic rename on the box, then restart so the service opens it):

    rsync -P data/fts/fts.sqlite hnsearch:/var/lib/hnsearch/current/fts.sqlite.tmp
    ssh hnsearch 'cd /var/lib/hnsearch/current && chown hnsearch: fts.sqlite.tmp &&
                  mv fts.sqlite.tmp fts.sqlite && systemctl restart hnsearch'
"""

import argparse
import sqlite3
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import BinaryIO, Iterator

# Must match rust-search/src/fts.rs (CREATE, CREATE_META).
CREATE = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5("
    "clean_text, content='', columnsize=0, detail=full, tokenize='unicode61')"
)
CREATE_META = (
    "CREATE TABLE IF NOT EXISTS fts_meta (id INTEGER PRIMARY KEY CHECK (id = 1), "
    "max_rowid INTEGER NOT NULL)"
)

# Runs on the box with its system python3. Frames: rowid (i64), byte length
# (u32), UTF-8 text. Read-only, idle I/O class, lowest CPU priority.
_EXPORTER = r"""
import sqlite3, struct, sys
limit = int(sys.argv[1])
c = sqlite3.connect("file:/var/lib/hnsearch/current/docs.sqlite?mode=ro", uri=True)
out = sys.stdout.buffer
q = "SELECT rowid, clean_text FROM doc ORDER BY rowid" + (" LIMIT ?" if limit else "")
for rowid, text in c.execute(q, (limit,) if limit else ()):
    b = (text or "").encode("utf-8", "replace")
    out.write(struct.pack("<qI", rowid, len(b)))
    out.write(b)
"""

_HEADER = struct.Struct("<qI")


def read_frames(stream: BinaryIO) -> Iterator[tuple[int, str]]:
    while True:
        head = stream.read(_HEADER.size)
        if not head:
            return
        rowid, n = _HEADER.unpack(head)
        yield rowid, stream.read(n).decode("utf-8")


def remote_rows(
    host: str, limit: int
) -> tuple[Iterator[tuple[int, str]], subprocess.Popen]:
    proc = subprocess.Popen(
        [
            "ssh",
            "-C",
            host,
            f"nice -n 19 ionice -c3 python3 -c {_quote(_EXPORTER)} {limit}",
        ],
        stdout=subprocess.PIPE,
        bufsize=1 << 20,
    )
    assert proc.stdout is not None
    return read_frames(proc.stdout), proc


def _quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def local_rows(path: str, limit: int) -> Iterator[tuple[int, str]]:
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    q = "SELECT rowid, clean_text FROM doc ORDER BY rowid" + (
        " LIMIT ?" if limit else ""
    )
    for rowid, text in c.execute(q, (limit,) if limit else ()):
        yield rowid, text or ""


def build(rows: Iterator[tuple[int, str]], out: Path, batch: int) -> int:
    tmp = out.with_suffix(".building")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(tmp)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA cache_size=-4000000")  # ~4 GB
    db.execute(CREATE)
    db.execute(CREATE_META)
    db.execute("INSERT INTO fts_meta (id, max_rowid) VALUES (1, 0)")

    t0 = time.time()
    n = 0
    last = 0
    buf: list[tuple[int, str]] = []

    def flush():
        db.executemany("INSERT INTO fts (rowid, clean_text) VALUES (?, ?)", buf)
        db.execute("UPDATE fts_meta SET max_rowid = ? WHERE id = 1", (last,))
        db.commit()
        buf.clear()

    for rowid, text in rows:
        if rowid <= last:
            sys.exit(f"rows out of order at rowid {rowid} (after {last})")
        buf.append((rowid, text))
        last = rowid
        n += 1
        if len(buf) >= batch:
            flush()
            rate = n / (time.time() - t0)
            print(f"  {n:,} rows | {rate:,.0f} rows/s", file=sys.stderr, flush=True)
    if buf:
        flush()

    print(
        f"indexed {n:,} rows in {time.time() - t0:.0f}s; optimizing…", file=sys.stderr
    )
    t1 = time.time()
    # Merge all segments into one: smaller file, and every query reads one
    # b-tree per term instead of many.
    db.execute("INSERT INTO fts (fts) VALUES ('optimize')")
    db.commit()
    db.execute("VACUUM")
    db.close()
    tmp.rename(out)
    print(
        f"optimized in {time.time() - t1:.0f}s → {out} "
        f"({out.stat().st_size / 1e9:.2f} GB, max_rowid {last})",
        file=sys.stderr,
    )
    return n


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--remote", help="ssh host whose docs.sqlite to read")
    src.add_argument("--docs", help="a local docs.sqlite")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (testing)")
    ap.add_argument("--batch", type=int, default=100_000)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    proc = None
    if args.remote:
        rows, proc = remote_rows(args.remote, args.limit)
    else:
        rows = local_rows(args.docs, args.limit)
    build(rows, args.out, args.batch)
    if proc is not None and proc.wait() != 0:
        sys.exit(f"remote export exited with {proc.returncode}")


if __name__ == "__main__":
    main()
