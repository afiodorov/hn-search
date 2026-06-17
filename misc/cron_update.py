#!/usr/bin/env python
"""Wrapper to run the incremental update with structured start/finish logging.
Exits non-zero on failure. Run from where the ADMIN token lives (your laptop).

Requires env: HN_SEARCH_URL, HN_SEARCH_ADMIN_TOKEN, GOOGLE_CLOUD_PROJECT (+ creds).
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
