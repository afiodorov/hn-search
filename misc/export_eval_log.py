#!/usr/bin/env python
"""Export the durable eval log (Redis list `eval:log`) to a local JSONL file.

`hn_search.job_manager.JobManager.log_eval_record` appends one JSON record per
completed query — {query, source_ids, answer, ts} — to `eval:log` on every real
search, capped at the most recent 20,000 (see job_manager.py). This script dumps
that list to a versioned local file so the eval set survives a Redis flush and is
diffable/reviewable in the repo. Safe to re-run: it overwrites the output file with
whatever is currently in Redis (append-only on the Redis side, not on disk).

Usage:
    uv run python misc/export_eval_log.py
    uv run python misc/export_eval_log.py --redis-url redis://... --out evals/production_queries.jsonl
"""

import argparse
import json
from pathlib import Path

import redis
from dotenv import load_dotenv

from hn_search.cache_config import REDIS_URL

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis-url", default=REDIS_URL)
    parser.add_argument("--out", default="evals/production_queries.jsonl")
    args = parser.parse_args()

    client = redis.from_url(args.redis_url)
    raw_records = client.lrange("eval:log", 0, -1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for raw in raw_records:
            f.write(raw.decode("utf-8") + "\n")

    print(f"Wrote {len(raw_records)} eval records to {out_path}")


if __name__ == "__main__":
    main()
