#!/usr/bin/env python
"""Fetch new HN comments from BigQuery, embed with the ONNX encoder, and append them
to the Rust search service (/append).

Asks the service for its high-water mark (/max_id), pulls only newer comments,
embeds them on CPU (no torch/GPU), and POSTs them — new rows are searchable
immediately via the service's tail. Idempotent and resumable.

Run from wherever the ADMIN token lives (your laptop). Requires:
    HN_SEARCH_URL, HN_SEARCH_ADMIN_TOKEN     — the service + write token
    GOOGLE_CLOUD_PROJECT (+ creds)           — BigQuery billing

Usage:
    uv run --extra dev python misc/fetch_and_embed_new_comments.py
    uv run --extra dev python misc/fetch_and_embed_new_comments.py [--skip-fetch] [--skip-embed] [--skip-append] [--reset]
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import html2text
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

from hn_search.common import get_model

load_dotenv()

STATE_FILE = "data/raw/fetch_state.json"


def load_state() -> dict:
    state_path = Path(STATE_FILE)
    if state_path.exists():
        with open(state_path, "r") as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    state_path = Path(STATE_FILE)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"💾 State saved to {state_path}")


def strip_html(text: str) -> str:
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.ignore_emphasis = True
    return h.handle(text).strip()


def _search_service():
    """(url, headers) for the Rust search service from env.

    Uses the ADMIN token: /append and /max_id are admin-only. This token should live
    only where updates run (your laptop), never on the public web app.
    """
    url = os.getenv("HN_SEARCH_URL", "").rstrip("/")
    if not url:
        raise RuntimeError("HN_SEARCH_URL must be set")
    token = os.getenv("HN_SEARCH_ADMIN_TOKEN") or os.getenv("HN_SEARCH_TOKEN", "")
    if not token:
        raise RuntimeError("HN_SEARCH_ADMIN_TOKEN must be set")
    return url, {"Authorization": f"Bearer {token}"}


def get_max_id_rust() -> int:
    """Resume point: the service's highest ingested hn_id."""
    import httpx

    url, headers = _search_service()
    r = httpx.get(f"{url}/max_id", headers=headers, timeout=30)
    r.raise_for_status()
    max_id = r.json()["max_id"]
    print(f"Max ID from rust service: {max_id}")
    return max_id


def fetch_from_bigquery(min_id, output_dir="data/raw", state=None, project=None) -> Optional[Path]:
    """Fetch new comments from BigQuery starting after min_id."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if state and state.get("raw_file"):
        raw_file = Path(state["raw_file"])
        if raw_file.exists():
            print(f"✅ Found existing raw file: {raw_file}")
            return raw_file

    print(f"\n📥 Fetching comments from BigQuery with id > {min_id}")
    project = project or os.getenv("GOOGLE_CLOUD_PROJECT")

    # Service-account JSON in env (for cloud) or Application Default Credentials.
    creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if creds_json:
        import tempfile

        from google.oauth2 import service_account

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            f.write(creds_json)
            creds_file = f.name
        credentials = service_account.Credentials.from_service_account_file(creds_file)
        client = bigquery.Client(project=project, credentials=credentials)
        os.unlink(creds_file)
    else:
        client = bigquery.Client(project=project)

    print(f"💰 Using GCP project: {client.project}")

    query = f"""
    SELECT id, `by` AS author, `type`, text, timestamp
    FROM `bigquery-public-data.hacker_news.full`
    WHERE dead IS NOT TRUE AND deleted IS NOT TRUE AND text IS NOT NULL
      AND type = 'comment' AND id > {min_id}
    ORDER BY id
    """
    print("Running BigQuery query...")
    df = client.query(query).to_dataframe()
    print(f"Fetched {len(df):,} new comments")
    if len(df) == 0:
        print("No new comments to process")
        return None

    filepath = output_path / f"new_comments_from_{min_id}.parquet"
    temp_filepath = filepath.with_suffix(".parquet.tmp")
    df.to_parquet(temp_filepath, index=False)
    temp_filepath.rename(filepath)
    print(f"💾 Saved to {filepath}")
    return filepath


def generate_embeddings(parquet_file, state=None) -> Tuple[Optional[Path], Optional[pd.DataFrame]]:
    """Embed new comments using the ONNX encoder (no torch/GPU required)."""
    output_file = Path(str(parquet_file).replace(".parquet", "_embedded.parquet"))
    if output_file.exists():
        print(f"✅ Found existing embedded file: {output_file}")
        return output_file, pd.read_parquet(output_file)

    print(f"\n🔄 Loading {parquet_file}")
    df = pd.read_parquet(parquet_file)
    df = df[df["text"].notna() & (df["text"] != "")]
    df["clean_text"] = df["text"].astype(str).apply(strip_html)
    df = df[df["clean_text"].str.len() > 0]
    if len(df) == 0:
        print("No valid documents after cleaning")
        return None, None

    print(f"Processing {len(df):,} documents with ONNX encoder...")
    model = get_model()
    documents = df["clean_text"].tolist()
    batch_size = 64
    all_embeddings = []
    temp_output = output_file.with_suffix(".parquet.tmp")
    for start in range(0, len(documents), batch_size):
        end = min(start + batch_size, len(documents))
        all_embeddings.extend(model.encode(documents[start:end]))
        if end % (batch_size * 16) == 0 or end == len(documents):
            partial_df = df.iloc[:end].copy()
            partial_df["embedding"] = [e.tolist() for e in all_embeddings]
            partial_df.to_parquet(temp_output, index=False)
            print(f"  💾 {end}/{len(documents)} embedded")

    df["embedding"] = [e.tolist() for e in all_embeddings]
    temp_output.rename(output_file)
    print(f"\n✅ Saved all embeddings to {output_file}")
    return output_file, df


def append_to_rust(df, batch_size=500):
    """POST embedded rows to the Rust service /append (dedup is server-side)."""
    import httpx

    url, headers = _search_service()
    print(f"\n🔄 Appending {len(df):,} documents to rust service at {url}")
    total_appended = 0
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start : start + batch_size]
        rows = [
            {
                "hn_id": str(row["id"]),
                "clean_text": row["clean_text"],
                "author": str(row["author"]),
                "timestamp": str(row["timestamp"]),
                "type": str(row["type"]),
                "embedding": [float(x) for x in row["embedding"]],
            }
            for _, row in chunk.iterrows()
        ]
        resp = httpx.post(f"{url}/append", json={"rows": rows}, headers=headers, timeout=120)
        resp.raise_for_status()
        j = resp.json()
        total_appended += j["appended"]
        print(f"  appended {j['appended']} skipped {j['skipped']} (max_id={j['max_id']})")
    print(f"✅ Appended {total_appended:,} new rows to rust service")


def main():
    parser = argparse.ArgumentParser(description="Fetch, embed, and append new HN comments")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-embed", action="store_true")
    parser.add_argument("--skip-append", action="store_true")
    parser.add_argument("--reset", action="store_true", help="Reset local fetch state")
    parser.add_argument("--project", type=str, help="GCP project ID for BigQuery billing")
    args = parser.parse_args()

    print("=" * 80)
    print("HN incremental update → Rust search service")
    print("=" * 80)

    state = {} if args.reset else load_state()

    # Resume point comes from the service, not local state.
    max_id = get_max_id_rust()

    parquet_file = None
    if not args.skip_fetch:
        parquet_file = fetch_from_bigquery(max_id, state=state, project=args.project)
        if not parquet_file:
            return
        state["raw_file"] = str(parquet_file)
        save_state(state)
    elif state.get("raw_file"):
        parquet_file = Path(state["raw_file"])
    else:
        print("❌ No raw file in state - cannot skip fetch")
        return

    df = None
    if not args.skip_embed:
        _, df = generate_embeddings(parquet_file, state=state)
        if df is None:
            print("❌ Embedding generation produced no rows")
            return
    elif state.get("embedded_file"):
        df = pd.read_parquet(state["embedded_file"])
    else:
        print("❌ No embedded file in state - cannot skip embed")
        return

    if not args.skip_append:
        append_to_rust(df)
        state["completed_at"] = datetime.now().isoformat()
        save_state(state)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
