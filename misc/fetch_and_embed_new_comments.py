#!/usr/bin/env python
"""
Script to fetch new HN comments from BigQuery, generate embeddings using MPS (Mac GPU),
and upsert them into partitioned PostgreSQL tables.

Idempotent and resumable - can be interrupted and restarted at any point.

Usage:
    uv run python misc/fetch_and_embed_new_comments.py [--resume] [--skip-fetch] [--skip-embed] [--skip-upsert]
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import html2text
import numpy as np
import pandas as pd
import psycopg
import torch
from dotenv import load_dotenv
from google.cloud import bigquery
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from hn_search.db_config import get_db_config

load_dotenv()

STATE_FILE = "data/raw/fetch_state.json"


def load_state() -> dict:
    """Load state from file"""
    state_path = Path(STATE_FILE)
    if state_path.exists():
        with open(state_path, "r") as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    """Save state to file"""
    state_path = Path(STATE_FILE)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"💾 State saved to {state_path}")


def strip_html(text: str) -> str:
    """Clean HTML from text"""
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.ignore_emphasis = True
    return h.handle(text).strip()


def get_connection():
    """Get a fresh database connection with vector registration"""
    db_config = get_db_config()
    conn = psycopg.connect(**db_config)
    register_vector(conn)
    return conn


def find_latest_nonempty_partition():
    """Find the latest non-empty partitioned table"""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'public'
            AND tablename LIKE 'hn_documents_____%%'
            ORDER BY tablename DESC
        """)
        tables = [row[0] for row in cur.fetchall()]

        # Check each table for rows
        for table in tables:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            if count > 0:
                conn.close()
                print(f"Found latest non-empty partition: {table} ({count:,} rows)")
                return table

    conn.close()
    return None


def get_max_id_from_partition(table_name):
    """Get the maximum id from the specified partition"""
    conn = get_connection()
    with conn.cursor() as cur:
        # IDs are stored as text, so we need to cast to bigint for proper comparison
        cur.execute(f"SELECT MAX(CAST(id AS BIGINT)) FROM {table_name}")
        max_id = cur.fetchone()[0]
    conn.close()
    print(f"Max ID in {table_name}: {max_id}")
    return max_id


def fetch_from_bigquery(
    min_id, output_dir="data/raw", state=None, project=None
) -> Optional[Path]:
    """Fetch new comments from BigQuery starting after min_id"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if we already have a raw file from a previous run
    if state and state.get("raw_file"):
        raw_file = Path(state["raw_file"])
        if raw_file.exists():
            print(f"✅ Found existing raw file: {raw_file}")
            return raw_file

    print(f"\n📥 Fetching comments from BigQuery with id > {min_id}")

    # Initialize client - uses Application Default Credentials by default
    # Specify project to control billing
    project = project or os.getenv("GOOGLE_CLOUD_PROJECT")

    # Check for credentials JSON in environment (for Railway/cloud deployments)
    creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if creds_json:
        import tempfile
        from google.oauth2 import service_account

        # Write credentials to a temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            f.write(creds_json)
            creds_file = f.name

        credentials = service_account.Credentials.from_service_account_file(creds_file)
        client = bigquery.Client(project=project, credentials=credentials)

        # Clean up temp file
        os.unlink(creds_file)
    else:
        client = bigquery.Client(project=project)

    print(f"💰 Using GCP project: {client.project}")
    print(f"   Account: Run 'gcloud auth list' to see active account")
    print(f"   Tip: Set billing alerts at https://console.cloud.google.com/billing/")

    query = f"""
    SELECT
      id,
      `by` AS author,
      `type`,
      text,
      timestamp
    FROM
      `bigquery-public-data.hacker_news.full`
    WHERE
      dead IS NOT TRUE
      AND deleted IS NOT TRUE
      AND text IS NOT NULL
      AND type = 'comment'
      AND id > {min_id}
    ORDER BY id
    """

    print("Running BigQuery query...")
    query_job = client.query(query)
    df = query_job.to_dataframe()

    print(f"Fetched {len(df):,} new comments")

    if len(df) == 0:
        print("No new comments to process")
        return None

    # Use consistent filename for resumability
    filename = f"new_comments_from_{min_id}.parquet"
    filepath = output_path / filename

    # Save atomically with temp file
    temp_filepath = filepath.with_suffix(".parquet.tmp")
    df.to_parquet(temp_filepath, index=False)
    temp_filepath.rename(filepath)

    print(f"💾 Saved to {filepath}")

    return filepath


def generate_embeddings_mps(
    parquet_file, state=None
) -> Tuple[Optional[Path], Optional[pd.DataFrame]]:
    """Generate embeddings using MPS (Mac GPU) - saves progress incrementally"""
    output_file = Path(str(parquet_file).replace(".parquet", "_embedded.parquet"))

    # Check if embeddings already exist
    if output_file.exists():
        print(f"✅ Found existing embedded file: {output_file}")
        df = pd.read_parquet(output_file)
        return output_file, df

    # Check for MPS availability
    if torch.backends.mps.is_available():
        device = "mps"
        print("🚀 Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("⚠️  Using CPU (no GPU available)")

    model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2", device=device
    )

    print(f"\n🔄 Loading {parquet_file}")
    df = pd.read_parquet(parquet_file)
    df = df[df["text"].notna() & (df["text"] != "")]

    print(f"Processing {len(df):,} documents")

    # Clean text
    df["clean_text"] = df["text"].astype(str).apply(strip_html)
    df = df[df["clean_text"].str.len() > 0]

    if len(df) == 0:
        print("No valid documents after cleaning")
        return None, None

    # Generate embeddings in chunks and save incrementally
    chunk_size = 1000  # Save every 1000 documents
    encode_batch_size = 128

    documents = df["clean_text"].tolist()
    all_embeddings = []

    temp_output = output_file.with_suffix(".parquet.tmp")

    for chunk_start in range(0, len(documents), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(documents))
        chunk_docs = documents[chunk_start:chunk_end]

        print(
            f"\n📦 Processing chunk {chunk_start // chunk_size + 1}/{(len(documents) - 1) // chunk_size + 1}"
        )

        chunk_embeddings = []
        for batch_start in range(0, len(chunk_docs), encode_batch_size):
            batch = chunk_docs[batch_start : batch_start + encode_batch_size]
            print(
                f"  Encoding batch {batch_start // encode_batch_size + 1}/{(len(chunk_docs) - 1) // encode_batch_size + 1}"
            )

            embeddings = model.encode(
                batch,
                batch_size=encode_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            chunk_embeddings.extend(embeddings)

        all_embeddings.extend(chunk_embeddings)

        # Save intermediate progress
        partial_df = df.iloc[:chunk_end].copy()
        partial_df["embedding"] = [emb.tolist() for emb in all_embeddings]
        partial_df.to_parquet(temp_output, index=False)
        print(f"  💾 Saved progress: {chunk_end}/{len(documents)} documents")

    # Final save - rename temp to final
    df["embedding"] = [emb.tolist() for emb in all_embeddings]
    temp_output.rename(output_file)
    print(f"\n✅ Saved all embeddings to {output_file}")

    return output_file, df


def get_partition_name_for_timestamp(timestamp):
    """Get partition table name for a given timestamp"""
    # Convert timestamp to datetime if it's a string
    if isinstance(timestamp, str):
        dt = pd.to_datetime(timestamp)
    else:
        dt = timestamp

    return f"hn_documents_{dt.year:04d}_{dt.month:02d}"


def upsert_to_db(embedded_parquet_file, df, state=None):
    """Upsert embeddings into partitioned tables with batch processing"""
    print(f"\n🔄 Upserting {len(df):,} documents to database")

    conn = get_connection()

    # Group by partition
    df["partition"] = df["timestamp"].apply(get_partition_name_for_timestamp)

    # Track processed IDs if resuming
    processed_ids = set(state.get("upserted_ids", [])) if state else set()

    total_inserted = 0
    batch_size = 500  # Insert in batches for better performance

    try:
        for partition_name, partition_df in df.groupby("partition"):
            print(
                f"\n  📋 Processing partition {partition_name} ({len(partition_df):,} rows)"
            )

            # Get existing IDs from the database to avoid duplicates
            with conn.cursor() as cur:
                partition_ids = partition_df["id"].astype(str).tolist()
                cur.execute(
                    f"SELECT id FROM {partition_name} WHERE id = ANY(%s)",
                    (partition_ids,),
                )
                existing_ids = set(row[0] for row in cur.fetchall())
                print(f"  Found {len(existing_ids):,} existing IDs in {partition_name}")

            # Filter out already processed rows and existing IDs
            partition_df = partition_df[
                ~partition_df["id"].astype(str).isin(processed_ids | existing_ids)
            ]

            if len(partition_df) == 0:
                print(f"  ⏭️  All rows already exist in {partition_name}")
                continue

            print(f"  Inserting {len(partition_df):,} new rows")

            records = [
                (
                    str(row["id"]),
                    row["clean_text"],
                    str(row["author"]),
                    str(row["timestamp"]),
                    str(row["type"]),
                    row["embedding"],
                )
                for _, row in partition_df.iterrows()
            ]

            # Process in batches
            for batch_start in range(0, len(records), batch_size):
                batch = records[batch_start : batch_start + batch_size]

                with conn.cursor() as cur:
                    cur.executemany(
                        f"""
                        INSERT INTO {partition_name} (id, clean_text, author, timestamp, type, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        batch,
                    )

                conn.commit()

                # Track progress
                batch_ids = [r[0] for r in batch]
                processed_ids.update(batch_ids)

                total_inserted += len(batch)
                print(
                    f"    Progress: {batch_start + len(batch)}/{len(records)} | Total: {total_inserted:,}"
                )

                # Save state periodically
                if state is not None:
                    state["upserted_ids"] = list(processed_ids)
                    save_state(state)

            print(f"  ✅ Inserted {len(records):,} rows into {partition_name}")

    finally:
        conn.close()

    print(f"\n✅ Total inserted: {total_inserted:,} documents")


def main():
    """Main execution flow with resumability"""
    parser = argparse.ArgumentParser(
        description="Fetch, embed, and upsert HN comments (idempotent & resumable)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from previous state"
    )
    parser.add_argument(
        "--skip-fetch", action="store_true", help="Skip BigQuery fetch step"
    )
    parser.add_argument(
        "--skip-embed", action="store_true", help="Skip embedding generation step"
    )
    parser.add_argument(
        "--skip-upsert", action="store_true", help="Skip database upsert step"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Reset state and start fresh"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="GCP project ID for BigQuery billing (defaults to GOOGLE_CLOUD_PROJECT env var or gcloud default)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("HN Comments Fetcher, Embedder, and Upserter (Idempotent)")
    print("=" * 80)

    # Load or initialize state
    if args.reset:
        print("🔄 Resetting state...")
        state = {}
    else:
        state = load_state()
        if state:
            print(f"📂 Loaded state from {STATE_FILE}")
            print(f"   Raw file: {state.get('raw_file', 'None')}")
            print(f"   Embedded file: {state.get('embedded_file', 'None')}")
            print(f"   Upserted IDs: {len(state.get('upserted_ids', []))} documents")

    # Step 1: Find latest non-empty partition
    latest_partition = find_latest_nonempty_partition()
    if not latest_partition:
        print("❌ No partitioned tables found!")
        return

    # Step 2: Get max ID from that partition
    max_id = get_max_id_from_partition(latest_partition)
    if max_id is None:
        print("❌ Could not determine max ID")
        return

    state["max_id"] = max_id
    save_state(state)

    # Step 3: Fetch new comments from BigQuery
    parquet_file = None
    if not args.skip_fetch:
        parquet_file = fetch_from_bigquery(max_id, state=state, project=args.project)
        if not parquet_file:
            print("No new data to process")
            return
        state["raw_file"] = str(parquet_file)
        save_state(state)
    else:
        print("⏭️  Skipping fetch step")
        if state.get("raw_file"):
            parquet_file = Path(state["raw_file"])
        else:
            print("❌ No raw file in state - cannot skip fetch")
            return

    # Step 4: Generate embeddings
    embedded_file = None
    df = None
    if not args.skip_embed:
        embedded_file, df = generate_embeddings_mps(parquet_file, state=state)
        if not embedded_file:
            print("❌ Embedding generation failed")
            return
        state["embedded_file"] = str(embedded_file)
        save_state(state)
    else:
        print("⏭️  Skipping embed step")
        if state.get("embedded_file"):
            embedded_file = Path(state["embedded_file"])
            df = pd.read_parquet(embedded_file)
        else:
            print("❌ No embedded file in state - cannot skip embed")
            return

    # Step 5: Upsert to database
    if not args.skip_upsert:
        upsert_to_db(embedded_file, df, state=state)
        state["completed"] = True
        state["completed_at"] = datetime.now().isoformat()
        save_state(state)
    else:
        print("⏭️  Skipping upsert step")

    print("\n" + "=" * 80)
    print("✅ All done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
