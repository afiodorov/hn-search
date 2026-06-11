#!/usr/bin/env python
"""Railway cron job: fetch new HN comments, embed, upsert, attach partitions.

Wraps fetch_and_embed_new_comments.py with structured logging suitable for
Railway's log viewer. Exits non-zero on failure so Railway marks the run failed.

Requires env vars:
  DATABASE_URL                        — set to ${{Postgres.DATABASE_URL}}
  GOOGLE_CLOUD_PROJECT                — GCP project for BigQuery billing
  GOOGLE_APPLICATION_CREDENTIALS_JSON — service account JSON (paste full contents)
"""

import subprocess
import sys
import time
from datetime import datetime, timezone

start = datetime.now(timezone.utc)
print(f"[cron] hn-search incremental update starting at {start.isoformat()}")

result = subprocess.run(
    [sys.executable, "misc/fetch_and_embed_new_comments.py", "--reset"],
    text=True,
)

elapsed = time.time() - start.timestamp()
if result.returncode == 0:
    print(f"[cron] ✅ completed in {elapsed:.0f}s")
else:
    print(f"[cron] ❌ failed (exit {result.returncode}) after {elapsed:.0f}s")
    sys.exit(result.returncode)
