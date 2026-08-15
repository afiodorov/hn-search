"""Vector search via the standalone Rust service (HTTP).

The Rust `/search` returns rows already ordered by ascending cosine distance, in the
column order callers expect (`id, clean_text, author, timestamp, type, distance`), so
results flow straight through `rows_to_results`.
"""

import functools
import os

from hn_search.logging_config import get_logger

logger = get_logger(__name__)

RUST_URL = os.getenv("HN_SEARCH_URL", "").rstrip("/")
RUST_TOKEN = os.getenv("HN_SEARCH_TOKEN", "")
RUST_TIMEOUT = float(os.getenv("HN_SEARCH_TIMEOUT", "10"))


def _to_list(embedding) -> list[float]:
    return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)


@functools.cache
def _get_client():
    """Persistent keep-alive client: reuse one TCP+TLS connection across queries
    so the handshake (costly over the cross-datacenter hop) is paid once, not
    per request."""
    import httpx

    headers = {"Authorization": f"Bearer {RUST_TOKEN}"} if RUST_TOKEN else {}
    return httpx.Client(
        base_url=RUST_URL,
        headers=headers,
        timeout=RUST_TIMEOUT,
        limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=60),
    )


def _rows(hits: list[dict]) -> list[tuple]:
    return [
        (
            h["id"],
            h["clean_text"],
            h["author"],
            h["timestamp"],
            h["type"],
            h["distance"],
        )
        for h in hits
    ]


def search(
    query_embedding,
    n_results: int,
    time_after: str | None = None,
    time_before: str | None = None,
) -> list[tuple]:
    """POST the query vector to the Rust service; return rows as pg-shaped tuples.

    time_after/time_before (ISO8601) optionally filter by timestamp; omitting both
    is identical to a plain search.
    """
    if not RUST_URL:
        raise RuntimeError("HN_SEARCH_URL is not set for the rust search backend")
    body = {"embedding": _to_list(query_embedding), "k": n_results}
    if time_after:
        body["time_after"] = time_after
    if time_before:
        body["time_before"] = time_before
    resp = _get_client().post("/search", json=body)
    resp.raise_for_status()
    return _rows(resp.json())


def similar(hn_id: str, n_results: int) -> list[tuple]:
    """POST to the Rust /similar endpoint: reuses hn_id's own stored embedding as
    the query, returning related comments (excluding hn_id itself). Raises if
    hn_id isn't a known comment (404 from the service)."""
    if not RUST_URL:
        raise RuntimeError("HN_SEARCH_URL is not set for the rust search backend")
    resp = _get_client().post("/similar", json={"hn_id": hn_id, "k": n_results})
    resp.raise_for_status()
    return _rows(resp.json())


def get_docs(hn_ids: list[str]) -> dict[str, dict]:
    """Batch-fetch docs by hn_id (e.g. resolving parent_ids to their own
    text/author/timestamp). Missing ids are simply absent from the returned
    dict, not an error. Empty input short-circuits without a request."""
    if not hn_ids:
        return {}
    if not RUST_URL:
        raise RuntimeError("HN_SEARCH_URL is not set for the rust search backend")
    resp = _get_client().post("/docs", json={"hn_ids": hn_ids})
    resp.raise_for_status()
    return {d["id"]: d for d in resp.json()}
