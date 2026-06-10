#!/usr/bin/env python
"""Embed monthly raw parquet shards on a GPU (the vast.ai step).

Reads ``data/raw/month_*.parquet`` (from misc/fetch_historical.py), cleans HTML,
embeds clean_text with all-mpnet-base-v2, and writes
``data/embedded/month_*.parquet`` with an added float32 ``embedding`` column.

Resumable: a month whose output already exists is skipped, so you can interrupt
and restart, or sync finished files off the box while the rest keep computing.

Run on a CUDA box (torch + sentence-transformers are in the `dev` extra):
    uv run --extra dev python misc/generate_embeddings_gpu.py
    uv run --extra dev python misc/generate_embeddings_gpu.py --in data/raw --out data/embedded
"""

import argparse
from glob import glob
from pathlib import Path

import html2text
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def strip_html(text: str) -> str:
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.ignore_emphasis = True
    return h.handle(text).strip()


def pick_device() -> str:
    if torch.cuda.is_available():
        print(
            f"🚀 CUDA GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})"
        )
        return "cuda"
    if torch.backends.mps.is_available():
        print("🚀 MPS (Apple Silicon)")
        return "mps"
    print("⚠️  No GPU — using CPU (slow)")
    return "cpu"


def embed_file(model, in_file: Path, out_file: Path, encode_batch_size: int):
    df = pd.read_parquet(in_file)
    df = df[df["text"].notna() & (df["text"] != "")].copy()
    df["clean_text"] = df["text"].astype(str).apply(strip_html)
    df = df[df["clean_text"].str.len() > 0]
    if len(df) == 0:
        print(f"  {in_file.name}: no valid rows after cleaning, skipping")
        return 0

    embeddings = model.encode(
        df["clean_text"].tolist(),
        batch_size=encode_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.append_column(
        "embedding",
        pa.array([e.tolist() for e in embeddings], type=pa.list_(pa.float32())),
    )
    tmp = out_file.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="snappy")
    tmp.rename(out_file)
    return len(df)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in", dest="in_dir", default="data/raw", help="dir with month_*.parquet"
    )
    ap.add_argument("--out", dest="out_dir", default="data/embedded", help="output dir")
    ap.add_argument("--glob", default="month_*.parquet", help="input filename glob")
    ap.add_argument("--batch-size", type=int, default=128, help="encode batch size")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    in_files = sorted(glob(str(Path(args.in_dir) / args.glob)))
    if not in_files:
        raise FileNotFoundError(f"No files matching {args.glob} in {args.in_dir}")

    model = SentenceTransformer(MODEL_NAME, device=pick_device())
    print(f"Found {len(in_files)} month file(s)\n")

    total = 0
    for idx, in_path in enumerate(in_files, 1):
        in_file = Path(in_path)
        out_file = out_dir / in_file.name
        if out_file.exists():
            print(f"[{idx}/{len(in_files)}] ✓ {out_file.name} exists, skipping")
            continue
        print(f"[{idx}/{len(in_files)}] embedding {in_file.name} ...")
        n = embed_file(model, in_file, out_file, args.batch_size)
        total += n
        print(f"  ✅ {out_file.name}: {n:,} rows  (running total {total:,})")

    print(f"\n✅ Embedded {total:,} comments into {out_dir}/")


if __name__ == "__main__":
    main()
