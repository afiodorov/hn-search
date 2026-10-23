#!/usr/bin/env python
"""Fetch new HN comments, embed with the ONNX encoder, and append them to the Rust
search service (/append).

Asks the service for its high-water mark (/max_id), pulls only newer comments,
embeds them on CPU (no torch/GPU), and POSTs them — new rows are searchable
immediately via the service's tail. Idempotent and resumable.

Comments come from HN's official API by default, up to the newest one that is past
HN's 2-hour edit window. `--source bigquery` uses the public BigQuery dataset
instead, which is cheaper in bulk but has gone days without refreshing.

The high-water mark is the only state, and it must only ever move forward over a
contiguous id range: the service stores rows in id order and finds date windows by
binary search, so an id appended out of order breaks them. A missed or failed run
is therefore harmless (the next one starts from the same max_id), but a run must
never skip an id it failed to fetch — it stops there instead.

Run from wherever the ADMIN token lives (your laptop). Requires:
    HN_SEARCH_URL, HN_SEARCH_ADMIN_TOKEN     — the service + write token
    GOOGLE_CLOUD_PROJECT (+ creds)           — BigQuery billing, --source bigquery only

Usage:
    uv run --extra dev python misc/fetch_and_embed_new_comments.py
    uv run --extra dev python misc/fetch_and_embed_new_comments.py [--skip-fetch] [--skip-embed] [--skip-append] [--reset]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, cast

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


def fetch_from_bigquery(
    min_id, output_dir="data/raw", state=None, project=None
) -> Optional[Path]:
    """Fetch new comments from BigQuery starting after min_id."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Reuse is keyed on min_id, which the filename encodes — a retry (even with
    # --reset, which wipes the state file) skips the BigQuery re-scan, while a run
    # whose min_id has advanced past an appended chunk correctly fetches fresh.
    # The write below is rename-atomic, so an existing final file is always complete.
    filepath = output_path / f"new_comments_from_{min_id}.parquet"
    if filepath.exists():
        print(f"✅ Found existing raw file: {filepath}")
        return filepath

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
    SELECT id, `by` AS author, `type`, text, timestamp, parent
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

    temp_filepath = filepath.with_suffix(".parquet.tmp")
    df.to_parquet(temp_filepath, index=False)
    temp_filepath.rename(filepath)
    print(f"💾 Saved to {filepath}")
    return filepath


HN_API = "https://hacker-news.firebaseio.com/v0"
# HN lets authors edit for 2 hours; fetching only items at least this old means
# we index what they settled on, not a draft.
EDIT_WINDOW_S = 2 * 3600


class FetchStopped(Exception):
    """An item couldn't be fetched, so the run stopped just before it.

    Appends must stay in id order (the service binary-searches rows by time), so
    skipping past a failed id would strand it below the next run's max_id
    forever. Everything before it is still appended; the next run (or Airflow's
    retry) resumes from there.
    """


async def _fetch_items(ids, concurrency: int, attempts: int = 4) -> dict:
    """{id: item JSON or None}; raises FetchStopped at the first id that kept
    failing, carrying what was fetched before it."""
    import asyncio

    import httpx

    sem = asyncio.Semaphore(concurrency)
    out: dict = {}

    async with httpx.AsyncClient(timeout=20) as client:

        async def one(i):
            async with sem:
                for attempt in range(attempts):
                    try:
                        r = await client.get(f"{HN_API}/item/{i}.json")
                        r.raise_for_status()
                        out[i] = r.json()
                        return
                    except (httpx.HTTPError, ValueError):
                        await asyncio.sleep(2**attempt)

        await asyncio.gather(*(one(i) for i in ids))

    for i in ids:
        if i not in out:
            raise FetchStopped(i, {k: v for k, v in out.items() if k < i})
    return out


def fetch_from_hn_api(
    min_id, output_dir="data/raw", concurrency=20, batch=2000
) -> Tuple[Optional[Path], bool]:
    """Fetch comments with id > min_id from the official HN API, stopping at the
    first item younger than the edit window.

    Returns (parquet path or None, stopped_early). The rows are a contiguous id
    range, so appending them in order keeps the service's rows time-ordered.
    """
    import asyncio

    import httpx

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / f"new_comments_from_{min_id}.parquet"
    if filepath.exists():
        print(f"✅ Found existing raw file: {filepath}")
        return filepath, False

    max_item = httpx.get(f"{HN_API}/maxitem.json", timeout=30).json()
    cutoff = time.time() - EDIT_WINDOW_S
    print(
        f"\n📥 Fetching items {min_id + 1}..{max_item} from the HN API "
        f"(posted before {datetime.fromtimestamp(cutoff, timezone.utc):%F %T} UTC)"
    )

    rows, stopped, done = [], False, False
    for start in range(min_id + 1, max_item + 1, batch):
        ids = list(range(start, min(start + batch, max_item + 1)))
        try:
            items = asyncio.run(_fetch_items(ids, concurrency))
        except FetchStopped as e:
            failed_id, items = e.args
            ids = [i for i in ids if i < failed_id]
            print(f"⚠️  item {failed_id} kept failing; stopping just before it")
            stopped = done = True
        for i in ids:
            item = items[i]
            if item is None:  # never existed / not visible; nothing to index
                continue
            if item.get("time", 0) > cutoff:
                done = True
                break
            if (
                item.get("type") == "comment"
                and not item.get("dead")
                and not item.get("deleted")
                and item.get("text")
            ):
                rows.append(
                    {
                        "id": i,
                        "author": item.get("by"),
                        "type": "comment",
                        "text": item["text"],
                        "timestamp": pd.Timestamp(item["time"], unit="s", tz="UTC"),
                        "parent": item.get("parent"),
                    }
                )
        print(f"  {ids[-1] if ids else start}: {len(rows):,} comments so far")
        if done:
            break

    print(f"Fetched {len(rows):,} new comments")
    if not rows:
        print("No new comments to process")
        return None, stopped
    df = pd.DataFrame(rows)
    temp_filepath = filepath.with_suffix(".parquet.tmp")
    df.to_parquet(temp_filepath, index=False)
    temp_filepath.rename(filepath)
    print(f"💾 Saved to {filepath}")
    return filepath, stopped


def embed_and_append(parquet_file, chunk_size=1000) -> int:
    """Embed and append in chunks, so a crash/timeout only loses one chunk's work.

    Each chunk is embedded then immediately POSTed to /append, which advances the
    service's max_id. A retry after a failure calls get_max_id_rust() and re-fetches
    from that new high-water mark, so at most `chunk_size` rows of embedding are ever
    redone instead of the whole backlog.
    """
    print(f"\n🔄 Loading {parquet_file}")
    df = pd.read_parquet(parquet_file)
    # pandas-stubs resolves boolean-mask indexing ambiguously here; the runtime
    # value is always a DataFrame.
    df = cast(pd.DataFrame, df[df["text"].notna() & (df["text"] != "")])
    df["clean_text"] = df["text"].astype(str).apply(strip_html)
    df = cast(pd.DataFrame, df[df["clean_text"].str.len() > 0])
    total = len(df)
    if total == 0:
        print("No valid documents after cleaning")
        return 0

    print(
        f"Processing {total:,} documents with ONNX encoder, {chunk_size}-row chunks..."
    )
    model = get_model()
    # ONNX attention/FFN activations scale with batch_size * MAX_SEQ_LENGTH; on the
    # memory-capped Hetzner box (LocalExecutor runs this inside airflow.service's
    # cgroup) a large batch spikes RAM and thrashes swap. 16 keeps peak ~hundreds of MB.
    embed_batch_size = int(os.getenv("HN_EMBED_BATCH_SIZE", "16"))
    total_appended = 0
    for start in range(0, total, chunk_size):
        chunk = df.iloc[start : start + chunk_size].copy()
        documents = chunk["clean_text"].tolist()
        embeddings = []
        for b in range(0, len(documents), embed_batch_size):
            embeddings.extend(model.encode(documents[b : b + embed_batch_size]))
        chunk["embedding"] = [e.tolist() for e in embeddings]
        total_appended += append_to_rust(chunk, batch_size=chunk_size)
        print(f"  💾 {min(start + chunk_size, total)}/{total} embedded+appended")
    return total_appended


def generate_embeddings(
    parquet_file, state=None
) -> Tuple[Optional[Path], Optional[pd.DataFrame]]:
    """Embed new comments using the ONNX encoder (no torch/GPU required)."""
    output_file = Path(str(parquet_file).replace(".parquet", "_embedded.parquet"))
    if output_file.exists():
        print(f"✅ Found existing embedded file: {output_file}")
        return output_file, pd.read_parquet(output_file)

    print(f"\n🔄 Loading {parquet_file}")
    df = pd.read_parquet(parquet_file)
    # pandas-stubs resolves boolean-mask indexing ambiguously here; the runtime
    # value is always a DataFrame.
    df = cast(pd.DataFrame, df[df["text"].notna() & (df["text"] != "")])
    df["clean_text"] = df["text"].astype(str).apply(strip_html)
    df = cast(pd.DataFrame, df[df["clean_text"].str.len() > 0])
    if len(df) == 0:
        print("No valid documents after cleaning")
        return None, None

    print(f"Processing {len(df):,} documents with ONNX encoder...")
    model = get_model()
    documents = df["clean_text"].tolist()
    # ONNX attention/FFN activations scale with batch_size * MAX_SEQ_LENGTH; on the
    # memory-capped Hetzner box (LocalExecutor runs this inside airflow.service's
    # cgroup) a large batch spikes RAM and thrashes swap. 16 keeps peak ~hundreds of MB.
    batch_size = int(os.getenv("HN_EMBED_BATCH_SIZE", "16"))
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


def append_to_rust(df, batch_size=1000) -> int:
    """POST embedded rows to the Rust service /append (dedup is server-side).

    Each row carries a 768-d f32 embedding (~10 KB of JSON), so a batch of 1000 is
    ~10 MB. The service and Caddy both cap request bodies at 64 MB; keep batches
    well under that.
    """
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
                "parent_id": str(int(row["parent"]))
                if "parent" in row and pd.notna(row["parent"])
                else None,
            }
            for _, row in chunk.iterrows()
        ]
        resp = httpx.post(
            f"{url}/append", json={"rows": rows}, headers=headers, timeout=120
        )
        resp.raise_for_status()
        j = resp.json()
        total_appended += j["appended"]
        print(
            f"  appended {j['appended']} skipped {j['skipped']} (max_id={j['max_id']})"
        )
    print(f"✅ Appended {total_appended:,} new rows to rust service")
    return total_appended


def cleanup_artifacts(data_dir="data/raw"):
    """Delete consumed fetch/embed parquet artifacts after a successful append.

    Each daily run leaves a ~60 MB ``*_embedded.parquet`` (plus the small raw fetch and
    any ``.parquet.tmp`` checkpoint); once those rows are in the Rust service the files
    are dead weight and were accumulating until the box ran low on disk. Call this ONLY
    after ``append_to_rust`` succeeds — a failed run must keep its files so a retry can
    resume cheaply by filename (see ``fetch_from_bigquery`` / ``generate_embeddings``).
    ``fetch_state.json`` is intentionally left in place.
    """
    d = Path(data_dir)
    stale = list(d.glob("new_comments_from_*.parquet")) + list(
        d.glob("new_comments_from_*.parquet.tmp")
    )
    removed, freed = 0, 0
    for f in stale:
        try:
            freed += f.stat().st_size
            f.unlink()
            removed += 1
        except OSError as e:
            print(f"  ⚠️  could not remove {f}: {e}")
    if removed:
        print(f"🧹 Cleaned {removed} artifact file(s), freed {freed / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch, embed, and append new HN comments"
    )
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-embed", action="store_true")
    parser.add_argument("--skip-append", action="store_true")
    parser.add_argument("--reset", action="store_true", help="Reset local fetch state")
    parser.add_argument(
        "--project", type=str, help="GCP project ID for BigQuery billing"
    )
    parser.add_argument(
        "--source",
        choices=["api", "bigquery"],
        default="api",
        help="HN's live API (default), or the BigQuery dataset, which lags by days",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("HN incremental update → Rust search service")
    print("=" * 80)

    state = {} if args.reset else load_state()

    # Resume point comes from the service, not local state.
    max_id = get_max_id_rust()

    parquet_file = None
    stopped = False
    if not args.skip_fetch:
        if args.source == "api":
            parquet_file, stopped = fetch_from_hn_api(max_id)
        else:
            parquet_file = fetch_from_bigquery(
                max_id, state=state, project=args.project
            )
        if not parquet_file:
            sys.exit(1 if stopped else 0)
        state["raw_file"] = str(parquet_file)
        save_state(state)
    elif state.get("raw_file"):
        parquet_file = Path(state["raw_file"])
    else:
        print("❌ No raw file in state - cannot skip fetch")
        return

    if not args.skip_embed and not args.skip_append:
        # Normal path: embed+append in chunks so a mid-run failure only costs one
        # chunk of re-embedding, not the whole backlog (state comes from the
        # service's max_id, not a local checkpoint file).
        chunk_size = int(os.getenv("HN_APPEND_CHUNK_SIZE", "1000"))
        embed_and_append(parquet_file, chunk_size=chunk_size)
        state["completed_at"] = datetime.now().isoformat()
        save_state(state)
        cleanup_artifacts()
        if stopped:
            # What was fetched is in; fail so Airflow retries the rest.
            sys.exit("❌ Stopped early on an item that wouldn't fetch")
        print("\n✅ Done.")
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
        cleanup_artifacts()

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
