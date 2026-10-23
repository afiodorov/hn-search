"""Hourly HN incremental update: fetch new comments from HN's official API (only
those past the 2-hour edit window), embed (ONNX), append to the local Rust search
service. Runs the repo script via uv on the box.

Each run starts from the service's max_id, not from its own logical date, so runs
aren't hour slices: a missed or failed hour just makes the next run bigger, and
Airflow retries are always safe. max_active_runs=1 matters: two runs appending at
once could land ids out of order, which breaks the service's date windows.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

REPO = "/root/hn-search"
UV = "/root/.local/bin/uv"

default_args = {
    "owner": "hnsearch",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="hn_incremental_update",
    description="Fetch (HN API)+embed+append new HN comments to the Rust search service",
    schedule="10 * * * *",  # hourly, at :10
    start_date=datetime(2026, 6, 19),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["hn-search"],
) as dag:
    update = BashOperator(
        task_id="fetch_embed_append",
        bash_command=(
            f"cd {REPO} && "
            f"PYTHONUNBUFFERED=1 {UV} run --extra dev python -u "
            f"misc/fetch_and_embed_new_comments.py --reset"
        ),
        # The script embeds+appends in 1000-row chunks, so each chunk advances the
        # service max_id and a timeout/retry only redoes the last partial chunk
        # (before 2026-07-27 it appended once at the very end, so a timeout threw
        # away the whole run and every retry restarted from row 0 -- the backlog
        # could never finish). Embedding runs ~5 rows/s on this box, so a large
        # catch-up backlog needs hours; 8h lets one attempt drain ~100k rows, and
        # retries pick up where the previous attempt left off.
        execution_timeout=timedelta(hours=8),
        # Shares a box-wide cgroup with imdb-sql's build_parquet; both peak in
        # the multi-GB range and can't run concurrently without swap-thrashing the
        # scheduler itself (2026-07-26 incident). heavy_memory_pool (1 slot, shared
        # with imdb_dataset_update) serializes them.
        pool="heavy_memory_pool",
    )
