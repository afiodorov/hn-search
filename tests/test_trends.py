"""The trends page: terms side by side per month, and its link preview.

The keyword index is stubbed at `trends._keyword` (what `/api/keyword` returns
per term), Redis is off, and the SPA shell is a minimal index.html.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from hn_search import permalinks, trends
from hn_search.api import app as app_module

INDEX = (
    "<!doctype html><html><head><title>HN Search</title></head>"
    '<body><div id="root"></div></body></html>'
)

MONTHS = [("2023-01", 1000), ("2023-02", 2000)]
COUNTS = {"rust": [10, 40], "go": [100, 100], "frobnicate": [0, 3]}


def fake_keyword(term):
    counts = COUNTS.get(term.lower(), [0, 0])
    return {
        "term": term,
        "count": sum(counts),
        "rows": 3000,
        "first": [{"id": "1", "timestamp": "2023-01-02", "url": "u"}]
        if sum(counts)
        else [],
        "months": [
            {"month": m, "rows": rows, "count": c}
            for (m, rows), c in zip(MONTHS, counts)
        ],
    }


@pytest.fixture(autouse=True)
def stubs(monkeypatch, tmp_path):
    monkeypatch.setattr(trends, "_keyword", fake_keyword)
    monkeypatch.setattr(trends, "redis_client", None)
    (tmp_path / "index.html").write_text(INDEX)
    monkeypatch.setattr(app_module, "_static_dir", tmp_path)
    permalinks.index_html.cache_clear()
    yield
    permalinks.index_html.cache_clear()


@pytest.fixture(scope="module")
def client():
    return TestClient(app_module.app)


def test_terms_are_trimmed_deduplicated_and_capped():
    assert trends.parse_terms(" rust ,Go,, rust , RUST, a  b ") == [
        "rust",
        "Go",
        "a b",
    ]
    assert len(trends.parse_terms(",".join(f"t{i}" for i in range(9)))) == 5
    assert trends.parse_terms("x" * 100) == ["x" * 40]


def test_series_line_up_month_by_month(client):
    body = client.get("/api/trends", params={"terms": "rust, go"}).json()
    assert [t["term"] for t in body["terms"]] == ["rust", "go"]
    assert body["terms"][0]["count"] == 50
    assert body["months"] == [
        {"month": "2023-01", "rows": 1000, "counts": [10, 100]},
        {"month": "2023-02", "rows": 2000, "counts": [40, 100]},
    ]
    assert client.get("/api/trends", params={"terms": " , "}).status_code == 400


def test_the_page_previews_only_terms_hn_actually_uses(client):
    page = client.get("/trends", params={"terms": "rust,frobnicate,go"}).text
    # "frobnicate" has 3 mentions: below the floor, so it is not named.
    assert "<title>rust vs go on Hacker News — HN Search</title>" in page
    described = [
        line for line in page.splitlines() if "title" in line or "description" in line
    ]
    assert described and not any("frobnicate" in line for line in described)
    assert "/trends.png?terms=rust%2Cfrobnicate%2Cgo" in page


def test_rare_terms_alone_get_the_generic_preview(client):
    page = client.get("/trends", params={"terms": "frobnicate"}).text
    assert "<title>Hacker News trends — HN Search</title>" in page
    assert "/og.png" in page and "trends.png" not in page


def test_the_preview_is_a_chart_png(client):
    res = client.get("/trends.png", params={"terms": "rust,go"})
    assert res.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(res.content)).size == (1200, 630)


def test_gridlines_land_on_round_numbers():
    assert trends.nice_step(505) == 200
    assert trends.nice_step(210) == 100
    assert trends.nice_step(3.7) == 1
    assert trends.nice_step(0) == 1
