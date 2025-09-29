from glob import glob
from pathlib import Path

import html2text
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer


def strip_html(text: str) -> str:
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.ignore_emphasis = True
    return h.handle(text).strip()


def generate_embeddings(data_dir: str = "data", output_dir: str = "embeddings"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2", device=device
    )

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    parquet_files = sorted(glob(str(data_path / "since_2023_*")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    print(f"Found {len(parquet_files)} parquet files")

    total_processed = 0
    batch_size = 5_000
    encode_batch_size = 128

    for file_idx, parquet_file in enumerate(parquet_files, 1):
        file_name = Path(parquet_file).name
        print(f"\n[{file_idx}/{len(parquet_files)}] Processing {file_name}")

        df = pd.read_parquet(parquet_file)
        df = df[df["text"].notna() & (df["text"] != "")]

        print(f"  Loaded {len(df)} rows")

        all_embeddings = []
        all_indices = []

        for batch_start in range(0, len(df), batch_size):
            batch_df = df.iloc[batch_start : batch_start + batch_size].copy()

            batch_df["clean_text"] = batch_df["text"].astype(str).apply(strip_html)
            batch_df = batch_df[batch_df["clean_text"].str.len() > 0]

            if len(batch_df) == 0:
                continue

            documents = batch_df["clean_text"].tolist()

            print(f"  Encoding batch of {len(documents)} documents...")
            embeddings = model.encode(
                documents,
                batch_size=encode_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            all_embeddings.extend(embeddings)
            all_indices.extend(batch_df.index.tolist())

            total_processed += len(documents)
            print(f"  Total processed: {total_processed}")

        if all_embeddings:
            result_df = df.loc[all_indices].copy()
            result_df["clean_text"] = result_df["text"].astype(str).apply(strip_html)

            embeddings_array = np.array([emb for emb in all_embeddings])

            table = pa.Table.from_pandas(result_df)
            embeddings_list = pa.array(
                [emb.tolist() for emb in embeddings_array], type=pa.list_(pa.float32())
            )
            table = table.append_column("embedding", embeddings_list)

            output_file = output_path / f"{file_name}.parquet"
            pq.write_table(table, output_file, compression="snappy")
            print(f"  Saved to {output_file}")

    print(f"\n✅ Successfully processed {total_processed} documents")


if __name__ == "__main__":
    generate_embeddings()
