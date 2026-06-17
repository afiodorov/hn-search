#!/usr/bin/env python
"""Gate B: parity check for the Rust search service against exact cosine.

Loads the same f16 vectors the service serves (rerank_f16.bin) and the row ids
(docs.sqlite), embeds a set of query strings with the ONNX encoder, and for each
query compares:

  * strict recall@10      — id overlap of Rust top-10 vs exact-cosine top-10
  * tie-tolerant recall   — Rust hit counts if its true distance ≤ the 10th-best
                            exact distance (+eps); tolerates swaps among near-ties
  * distance parity       — Rust-reported distance vs recomputed exact distance

Also runs a self-retrieval sanity check: querying with a stored vector must return
that same row at distance ≈ 0.

Usage:
    uv run python misc/eval_rust_parity.py --artifacts artifacts/ --url http://localhost:8011
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

DIM = 768
DEFAULT_QUERIES = [
    "best programming language for beginners",
    "how to scale a startup",
    "is remote work productive",
    "rust vs go performance",
    "concerns about artificial intelligence safety",
    "why is housing so expensive",
    "favorite text editor vim or emacs",
    "how to learn machine learning",
    "postgres vs mysql for web apps",
    "burnout in software engineering",
    "cryptocurrency is a scam",
    "tips for technical interviews",
    "self hosting vs cloud costs",
    "functional programming benefits",
    "how does kubernetes work",
    "climate change solutions",
    "starting a side project",
    "best laptop for developers",
    "imposter syndrome advice",
    "open source business models",
]


def load_corpus(artifacts: Path):
    """Memmap the f16 vectors (no 37 GB materialization) + load ids in row order."""
    meta = json.loads((artifacts / "meta.json").read_text())
    n = meta["count"]
    mm = np.memmap(artifacts / "rerank_f16.bin", dtype="<f2", mode="r", shape=(n, DIM))
    conn = sqlite3.connect(artifacts / "docs.sqlite")
    ids = [r[0] for r in conn.execute("SELECT hn_id FROM doc ORDER BY rowid").fetchall()]
    conn.close()
    assert n == len(ids), f"{n} vecs vs {len(ids)} ids"
    return mm, ids


def exact_topk(mm, q, k, chunk=500_000):
    """Exact cosine distance to every row, chunked over the memmap. Returns (order, dist)."""
    n = mm.shape[0]
    qn = np.linalg.norm(q) or 1e-12
    dist = np.empty(n, dtype=np.float32)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        c = np.asarray(mm[a:b], dtype=np.float32)
        cn = np.linalg.norm(c, axis=1)
        cn[cn == 0] = 1e-12
        dist[a:b] = 1.0 - (c @ q) / (cn * qn)
    order = np.argpartition(dist, k)[:k]
    return order[np.argsort(dist[order], kind="stable")], dist


def rust_search(url, token, q, k):
    import httpx

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = httpx.post(f"{url}/search", json={"embedding": q.tolist(), "k": k}, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--url", default="http://localhost:8011")
    ap.add_argument("--token", default="")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--eps", type=float, default=2e-3)
    ap.add_argument("--min-recall", type=float, default=0.99)
    ap.add_argument("--n-queries", type=int, default=len(DEFAULT_QUERIES),
                    help="How many of the default queries to run (each is a full corpus scan)")
    ap.add_argument("--chunk", type=int, default=500_000)
    args = ap.parse_args()

    mm, ids = load_corpus(args.artifacts)
    id_to_idx = {hn: i for i, hn in enumerate(ids)}  # built once
    print(f"corpus: {len(ids):,} vectors", file=sys.stderr)

    from hn_search.common import get_model

    model = get_model()

    # --- self-retrieval sanity: stored vector must return itself at distance ~0 ---
    probe = np.asarray(mm[0], dtype=np.float32)
    probe = probe / (np.linalg.norm(probe) or 1e-12)
    hits = rust_search(args.url, args.token, probe, args.k)
    ok_self = hits and hits[0]["id"] == ids[0] and hits[0]["distance"] < 1e-3
    print(f"self-retrieval: top1={hits[0]['id']} dist={hits[0]['distance']:.2e} "
          f"expected={ids[0]} -> {'OK' if ok_self else 'FAIL'}")

    queries = DEFAULT_QUERIES[: args.n_queries]
    strict, tie, dmax = [], [], 0.0
    for query in queries:
        q = model.encode([query])[0].astype(np.float32)
        gt_order, dist = exact_topk(mm, q, args.k, args.chunk)
        gt_ids = [ids[i] for i in gt_order]
        threshold = dist[gt_order[-1]]

        hits = rust_search(args.url, args.token, q, args.k)
        rust_ids = [h["id"] for h in hits]

        strict.append(len(set(gt_ids) & set(rust_ids)) / args.k)
        good = sum(1 for rid in rust_ids if dist[id_to_idx[rid]] <= threshold + args.eps)
        tie.append(good / args.k)
        for h in hits:
            dmax = max(dmax, abs(dist[id_to_idx[h["id"]]] - h["distance"]))
        print(f"  · {query[:40]:40s} strict={strict[-1]:.1f} tie={tie[-1]:.1f}", file=sys.stderr)

    msr, mtr = float(np.mean(strict)), float(np.mean(tie))
    print(f"\nqueries: {len(queries)}")
    print(f"strict recall@{args.k}:        {msr:.3f}")
    print(f"tie-tolerant recall@{args.k}:  {mtr:.3f}")
    print(f"max |distance| delta:        {dmax:.2e}")

    passed = ok_self and mtr >= args.min_recall and dmax < 0.05
    print(f"\nGate B: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
