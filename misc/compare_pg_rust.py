#!/usr/bin/env python
"""Head-to-head: the pgvector production path vs the Rust service, same queries.

For each query it embeds once, fetches pg `_search` top-k and Rust `/search` top-k,
and prints the id overlap plus any disagreements with each side's (exact-cosine)
distance. Both paths use the same vectors and the same exact rerank, so differences
come only from the *shortlist*: pg's approximate HNSW walk vs Rust's exact brute-force
scan. When a `rust-only` hit has a *lower* distance than the `pg-only` hit it
displaced, Rust found a true-nearer neighbor that pg's HNSW missed.

Usage:
    uv run python misc/compare_pg_rust.py --url http://localhost:8011
"""

import argparse

import numpy as np

QUERIES = [
    "rust vs go performance",
    "concerns about artificial intelligence safety",
    "why is housing so expensive",
    "how to scale a startup",
    "burnout in software engineering",
    "postgres vs mysql for web apps",
    "best programming language for beginners",
    "self hosting vs cloud costs",
    "how does kubernetes work",
    "tips for technical interviews",
]


def rust_search(url, token, q, k):
    import httpx

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = httpx.post(f"{url}/search", json={"embedding": q.tolist(), "k": k}, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8011")
    ap.add_argument("--token", default="")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--queries-file", type=str, help="One query per line (overrides defaults)")
    args = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv()

    from hn_search.common import get_model
    from hn_search.rag.nodes import _search, get_connection_pool

    queries = QUERIES
    if args.queries_file:
        queries = [ln.strip() for ln in open(args.queries_file) if ln.strip()]

    model = get_model()
    pool = get_connection_pool()

    overlaps = []
    for query in queries:
        q = model.encode([query])[0].astype(np.float32)
        pg = _search(pool, q, args.k)
        rust = rust_search(args.url, args.token, q, args.k)
        pg_d = {r[0]: r[5] for r in pg}
        rust_d = {h["id"]: h["distance"] for h in rust}

        overlap = len(set(pg_d) & set(rust_d)) / args.k
        same_order = list(pg_d) == list(rust_d)
        overlaps.append(overlap)

        print(f"\nQ: {query!r}")
        print(f"   overlap {len(set(pg_d) & set(rust_d))}/{args.k}  same_order={same_order}")
        pg_only = set(pg_d) - set(rust_d)
        rust_only = set(rust_d) - set(pg_d)
        if pg_only or rust_only:
            print(f"   pg-only:   {[f'{i}(d={pg_d[i]:.4f})' for i in pg_only]}")
            print(f"   rust-only: {[f'{i}(d={rust_d[i]:.4f})' for i in rust_only]}")
            best_pg_only = min((pg_d[i] for i in pg_only), default=None)
            best_rust_only = min((rust_d[i] for i in rust_only), default=None)
            if best_pg_only is not None and best_rust_only is not None:
                verdict = "rust nearer (pg HNSW missed it)" if best_rust_only < best_pg_only else "pg nearer"
                print(f"   → {verdict}")

    print(f"\nmean top-{args.k} overlap: {np.mean(overlaps):.3f} over {len(queries)} queries")


if __name__ == "__main__":
    main()
