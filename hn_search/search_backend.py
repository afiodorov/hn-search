"""Vector search via the standalone Rust service (HTTP).

The Rust `/search` returns rows already ordered by ascending cosine distance, in the
column order callers expect (`id, clean_text, author, timestamp, type, distance`), so
results flow straight through `rows_to_results`.
"""

import os

from hn_search.logging_config import get_logger

logger = get_logger(__name__)

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


def search(query_embedding, n_results: int) -> list[tuple]:
    """POST the query vector to the Rust service; return rows as pg-shaped tuples."""
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
