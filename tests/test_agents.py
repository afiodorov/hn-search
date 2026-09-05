"""The agent-facing surface: /mcp, the JSON routes and llms.txt.

Drives the FastAPI app in-process through Starlette's TestClient, which runs
the lifespan (and so the MCP session manager) for it. The Rust service and the
ONNX encoder are stubbed at the `search_backend` seam, so no network, no model
download, no Redis. What is under test is the wiring: every route answers, the
MCP tools are listed and callable, and errors come back readable.

Run with `uv run --group test pytest`.
"""

import httpx
import pytest
from fastapi.testclient import TestClient

from hn_search import search_backend
from hn_search.api import agents
from hn_search.api.app import app

# Stateless JSON-mode MCP over plain HTTP: every POST is a complete JSON-RPC
# exchange, no session id to carry.
MCP_HEADERS = {
    "content-type": "application/json",
    "accept": "application/json, text/event-stream",
}

ROW = ("43000001", "Rust is fine.", "alice", "2025-01-02T03:04:05", "comment", 0.31)


class FakeEncoder:
    def encode(self, texts):
        return [[0.0] * 768 for _ in texts]


@pytest.fixture(autouse=True)
def stub_backend(monkeypatch):
    calls: dict[str, list] = {"search": [], "similar": [], "docs": []}

    def fake_search(embedding, n_results, time_after=None, time_before=None):
        calls["search"].append((n_results, time_after, time_before))
        return [ROW]

    def fake_similar(hn_id, n_results):
        calls["similar"].append((hn_id, n_results))
        if hn_id == "404":
            # What httpx raises for the Rust service's 404; its text names the
            # service URL, which must not reach the caller.
            req = httpx.Request("POST", "https://backend.internal/similar")
            resp = httpx.Response(404, request=req)
            raise httpx.HTTPStatusError(
                "Client error '404 Not Found' for url", request=req, response=resp
            )
        if hn_id == "500":
            raise RuntimeError("boom at https://backend.internal")
        return [ROW]

    def fake_docs(hn_ids):
        calls["docs"].append(list(hn_ids))
        return {
            "43000001": {
                "id": "43000001",
                "clean_text": "Rust is fine.",
                "author": "alice",
                "timestamp": "2025-01-02T03:04:05",
                "type": "comment",
                "parent_id": "43000000",
            }
        }

    monkeypatch.setattr(search_backend, "search", fake_search)
    monkeypatch.setattr(search_backend, "similar", fake_similar)
    monkeypatch.setattr(search_backend, "get_docs", fake_docs)
    monkeypatch.setattr(search_backend, "stats", lambda: {"count": 12, "max_id": 9})
    # The tools module imported these names directly, so patch them there too.
    monkeypatch.setattr("hn_search.rag.tools.search", fake_search)
    monkeypatch.setattr("hn_search.rag.tools.similar", fake_similar)
    monkeypatch.setattr("hn_search.rag.tools.get_model", lambda: FakeEncoder())
    monkeypatch.setattr("hn_search.rag.tools.get_cached_vector_search", lambda *a: None)
    monkeypatch.setattr("hn_search.rag.tools.cache_vector_search", lambda *a: None)
    return calls


@pytest.fixture(scope="module")
def client():
    with TestClient(app, base_url="https://hn.example") as c:
        yield c


def _call(client, name, **arguments):
    r = client.post(
        agents.MCP_PATH,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
        headers=MCP_HEADERS,
    )
    assert r.status_code == 200, r.text
    return r.json()["result"]


def test_llms_txt_names_its_own_host(client):
    r = client.get("/llms.txt")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "https://hn.example/mcp" in r.text
    assert "cosine distance" in r.text


def test_llms_txt_is_https_behind_a_proxy(client):
    # Uvicorn will not trust X-Forwarded-Proto from a non-loopback proxy, so
    # the request arrives as http; the file must still advertise https. An
    # absolute URL overrides the client's base, and the one client is reused
    # because the MCP session manager may only be started once per process.
    r = client.get("http://hn.fiodorov.es/llms.txt")
    assert "https://hn.fiodorov.es/mcp" in r.text


def test_find_route(client, stub_backend):
    r = client.get(
        "/api/find", params={"q": "rust", "k": 5, "time_after": "2025-01-01"}
    )
    assert r.status_code == 200, r.text
    hit = r.json()[0]
    assert hit["id"] == "43000001"
    assert hit["url"] == "https://news.ycombinator.com/item?id=43000001"
    assert hit["distance"] == 0.31
    assert stub_backend["search"] == [(5, "2025-01-01", None)]


def test_find_route_rejects_empty_query(client):
    r = client.get("/api/find", params={"q": "  "})
    assert r.status_code == 400


def test_find_caps_k(client, stub_backend):
    client.get("/api/find", params={"q": "rust", "k": 500})
    assert stub_backend["search"][-1][0] == agents.MAX_K


def test_similar_route(client, stub_backend):
    r = client.get("/api/similar", params={"id": "43000000", "k": 3})
    assert r.status_code == 200
    assert r.json()[0]["author"] == "alice"
    assert stub_backend["similar"] == [("43000000", 3)]


def test_similar_route_rejects_non_numeric_id(client):
    r = client.get("/api/similar", params={"id": "abc"})
    assert r.status_code == 400


def test_comments_route_accepts_both_list_styles(client, stub_backend):
    r = client.get("/api/comments", params={"ids": "43000001,999"})
    assert r.status_code == 200
    body = r.json()
    assert [d["id"] for d in body] == ["43000001"]  # the unknown id is omitted
    assert body[0]["parent_id"] == "43000000"
    assert body[0]["text"] == "Rust is fine."

    r = client.get("/api/comments", params=[("ids", "43000001"), ("ids", "999")])
    assert [d["id"] for d in r.json()] == ["43000001"]
    assert stub_backend["docs"][-1] == ["43000001", "999"]


def test_comments_route_rejects_empty(client):
    assert client.get("/api/comments").status_code == 400


def test_unknown_id_is_a_404_without_the_backend_url(client):
    r = client.get("/api/similar", params={"id": "404"})
    assert r.status_code == 404
    assert "no such id" in r.json()["detail"]
    assert "backend.internal" not in r.text


def test_backend_failure_is_a_502_without_the_backend_url(client):
    r = client.get("/api/similar", params={"id": "500"})
    assert r.status_code == 502
    assert "RuntimeError" in r.json()["detail"]
    assert "backend.internal" not in r.text


def test_mcp_does_not_shadow_the_static_mount(client):
    r = client.get("/mcpx")
    assert r.status_code in (200, 404)
    assert "jsonrpc" not in r.text


def test_mcp_lists_the_tools(client):
    r = client.post(
        agents.MCP_PATH,
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers=MCP_HEADERS,
    )
    assert r.status_code == 200, r.text
    names = {t["name"] for t in r.json()["result"]["tools"]}
    assert names == {"search", "similar", "comments", "stats", "ask"}


def test_mcp_search_returns_structured_hits(client):
    result = _call(client, "search", query="rust", k=2)
    assert result["isError"] is False, result
    hits = result["structuredContent"]["result"]
    assert hits[0]["id"] == "43000001"
    assert hits[0]["url"].endswith("43000001")


def test_mcp_stats(client):
    result = _call(client, "stats")
    assert result["structuredContent"] == {"count": 12, "max_id": 9}


def test_mcp_error_is_readable(client):
    result = _call(client, "similar", hn_id="404")
    assert result["isError"] is True
    assert "404" in result["content"][0]["text"]
    assert "backend.internal" not in result["content"][0]["text"]

    result = _call(client, "search", query="")
    assert result["isError"] is True
    assert "empty" in result["content"][0]["text"]
