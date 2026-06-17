#!/usr/bin/env python
"""Railway cron job: fetch new HN comments, embed, and ship them downstream.

Wraps fetch_and_embed_new_comments.py with structured logging suitable for
Railway's log viewer. Exits non-zero on failure so Railway marks the run failed.

The destination is set by HN_UPDATE_TARGET (default "pg"):
  pg    — upsert into Postgres (legacy).
  rust  — POST to the Rust search service /append (the cut-over target). Requires
          HN_SEARCH_URL and HN_SEARCH_TOKEN.

Requires env vars:
  GOOGLE_CLOUD_PROJECT                — GCP project for BigQuery billing
  GOOGLE_APPLICATION_CREDENTIALS_JSON — service account JSON (paste full contents)
  DATABASE_URL                        — only for HN_UPDATE_TARGET=pg
  HN_SEARCH_URL / HN_SEARCH_TOKEN     — only for HN_UPDATE_TARGET=rust
"""

import os
import subprocess
import sys
import time
from datetime import datetime, timezone

target = os.getenv("HN_UPDATE_TARGET", "pg")
start = datetime.now(timezone.utc)
print(f"[cron] hn-search incremental update (target={target}) starting at {start.isoformat()}")

result = subprocess.run(
    [sys.executable, "misc/fetch_and_embed_new_comments.py", "--reset", "--target", target],
    text=True,
)

elapsed = time.time() - start.timestamp()
if result.returncode == 0:
    print(f"[cron] ✅ completed in {elapsed:.0f}s")
else:
    print(f"[cron] ❌ failed (exit {result.returncode}) after {elapsed:.0f}s")
    sys.exit(result.returncode)
