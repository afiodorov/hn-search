#!/usr/bin/env python
"""Fetch HN comments from BigQuery, one parquet per month.

Writes ``data/raw/month_YYYY_MM.parquet`` shards (step 1 of the full rebuild:
fetch -> embed -> build artifacts). Resumable: existing month files are skipped.
Torch-free, so it runs anywhere (locally or on the GPU box).

Usage:
    uv run --extra dev python misc/fetch_historical.py                 # 2023-01 .. current month
    uv run --extra dev python misc/fetch_historical.py --start 2023-01 --end 2026-06
    uv run --extra dev python misc/fetch_historical.py --project my-gcp-project
"""

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()


def month_range(start: str, end: str):
    """Yield (year, month) tuples from start to end inclusive. Args are 'YYYY-MM'."""
    sy, sm = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


def make_client(project: str | None):
    project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
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
        return client
    return bigquery.Client(project=project)


def fetch_month(client, year: int, month: int, out_dir: Path) -> Path | None:
    out_file = out_dir / f"month_{year:04d}_{month:02d}.parquet"
    if out_file.exists():
        print(f"  ✓ {out_file.name} exists, skipping")
        return out_file

    nxt_y, nxt_m = (year + 1, 1) if month == 12 else (year, month + 1)
    query = f"""
    SELECT id, `by` AS author, `type`, text, timestamp
    FROM `bigquery-public-data.hacker_news.full`
    WHERE dead IS NOT TRUE
      AND deleted IS NOT TRUE
      AND text IS NOT NULL
      AND type = 'comment'
      AND timestamp >= TIMESTAMP('{year:04d}-{month:02d}-01')
      AND timestamp <  TIMESTAMP('{nxt_y:04d}-{nxt_m:02d}-01')
    ORDER BY id
    """
    df = client.query(query).to_dataframe()
    print(f"  fetched {len(df):,} comments for {year:04d}-{month:02d}")
    if len(df) == 0:
        return None
    tmp = out_file.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(out_file)
    print(f"  💾 saved {out_file.name}")
    return out_file


def main():
    now = datetime.now(timezone.utc)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2023-01", help="first month, YYYY-MM")
    ap.add_argument(
        "--end", default=f"{now.year:04d}-{now.month:02d}", help="last month, YYYY-MM"
    )
    ap.add_argument("--out", default="data/raw", help="output directory")
    ap.add_argument("--project", help="GCP billing project (else GOOGLE_CLOUD_PROJECT)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = make_client(args.project)
    print(f"💰 BigQuery project: {client.project}")
    print(f"Fetching {args.start} .. {args.end} into {out_dir}/\n")

    total = 0
    for year, month in month_range(args.start, args.end):
        print(f"[{year:04d}-{month:02d}]")
        f = fetch_month(client, year, month, out_dir)
        if f:
            total += 1
    print(f"\n✅ Done. {total} month file(s) present in {out_dir}/")


if __name__ == "__main__":
    main()
