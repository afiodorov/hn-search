"""Search backend selection: pgvector (default), the Rust service, or shadow.

`HN_SEARCH_BACKEND` picks the path used by the RAG retrieve step:
  pg      — Postgres/pgvector only (unchanged legacy path).
  rust    — the standalone Rust brute-force service over HTTP.
  shadow  — run both, log top-k overlap + latency, return the pg results (safe
            default for cut-over validation on live traffic).

The Rust `/search` returns rows already ordered by ascending cosine distance, in the
same column order as the pg query (`id, clean_text, author, timestamp, type, distance`),
so callers can treat both identically via `rows_to_results`.
"""

import os
import time

from hn_search.logging_config import get_logger

logger = get_logger(__name__)

BACKEND = os.getenv("HN_SEARCH_BACKEND", "pg").lower()
RUST_URL = os.getenv("HN_SEARCH_URL", "").rstrip("/")
RUST_TOKEN = os.getenv("HN_SEARCH_TOKEN", "")
RUST_TIMEOUT = float(os.getenv("HN_SEARCH_TIMEOUT", "10"))


def _to_list(embedding) -> list[float]:
    return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)


# Persistent keep-alive client: reuse one TCP+TLS connection across queries so the
# handshake (costly over the cross-datacenter hop) is paid once, not per request.
_client = None


def _get_client():
    global _client
    if _client is None:
        import httpx

        headers = {"Authorization": f"Bearer {RUST_TOKEN}"} if RUST_TOKEN else {}
        _client = httpx.Client(
            base_url=RUST_URL,
            headers=headers,
            timeout=RUST_TIMEOUT,
            limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=60),
        )
    return _client


def search_rust(query_embedding, n_results: int) -> list[tuple]:
    """POST the query vector to the Rust service; return pg-shaped rows."""
    if not RUST_URL:
        raise RuntimeError("HN_SEARCH_URL is not set for the rust search backend")
    resp = _get_client().post(
        "/search", json={"embedding": _to_list(query_embedding), "k": n_results}
    )
    resp.raise_for_status()
    return [
        (h["id"], h["clean_text"], h["author"], h["timestamp"], h["type"], h["distance"])
        for h in resp.json()
    ]


def log_shadow(pg_rows: list[tuple], rust_rows: list[tuple], pg_ms: float, rust_ms: float) -> None:
    """Compare two result sets (id is column 0) and log overlap + latency."""
    pg_ids = [r[0] for r in pg_rows]
    rust_ids = [r[0] for r in rust_rows]
    k = max(len(pg_ids), 1)
    overlap = len(set(pg_ids) & set(rust_ids)) / k
    same_order = pg_ids == rust_ids
    logger.info(
        "🔀 shadow: top-%d overlap=%.2f same_order=%s pg=%.0fms rust=%.0fms",
        len(pg_ids),
        overlap,
        same_order,
        pg_ms,
        rust_ms,
    )


def dispatch_search(pg_search, pool, query_embedding, n_results: int) -> list[tuple]:
    """Run the configured backend. `pg_search` is the legacy `_search(pool, ...)`."""
    if BACKEND == "rust":
        return search_rust(query_embedding, n_results)

    if BACKEND == "shadow":
        t0 = time.perf_counter()
        pg_rows = pg_search(pool, query_embedding, n_results)
        pg_ms = (time.perf_counter() - t0) * 1000
        try:
            t1 = time.perf_counter()
            rust_rows = search_rust(query_embedding, n_results)
            rust_ms = (time.perf_counter() - t1) * 1000
            log_shadow(pg_rows, rust_rows, pg_ms, rust_ms)
        except Exception as e:
            logger.warning("shadow rust search failed: %s", e)
        return pg_rows

    return pg_search(pool, query_embedding, n_results)
