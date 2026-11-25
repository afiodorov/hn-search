"""Permanent answer pages: saved snapshots, /a/<id>, and their link previews.

Redis is a dict-backed stand-in with the commands `permalinks.py` uses, and the
SPA shell is a minimal index.html in a temp dir, so this needs neither a Redis
nor a frontend build.
"""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from hn_search import permalinks
from hn_search.api import app as app_module
from hn_search.api import search
from hn_search.job_manager import JobManager

INDEX = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="description" content="Default description." />
    <title>HN Search</title>
  </head>
  <body><div id="root"></div></body>
</html>"""

ANSWER = (
    "Most commenters prefer **Rust** for this, as user alice puts it "
    '"the borrow checker pays for itself" [[1]](https://news.ycombinator.com/'
    "item?id=1). Others say [Go](https://go.dev) is simpler [[2]](https://x)."
)
SOURCES = [{"id": "1", "author": "alice", "text": "...", "url": "u1"}]


class FakeRedis:
    def __init__(self):
        self.kv: dict[str, bytes] = {}
        self.sets: dict[str, set] = {}
        self.ttl: dict[str, int] = {}
        self.zsets: dict[str, dict] = {}

    def set(self, key, value, ex=None):
        self.kv[key] = value if isinstance(value, bytes) else value.encode()
        if ex is not None:
            self.ttl[key] = ex

    def expire(self, key, seconds):
        self.ttl[key] = seconds

    def get(self, key):
        return self.kv.get(key)

    def sadd(self, key, member):
        self.sets.setdefault(key, set()).add(member.encode())

    def smembers(self, key):
        return set(self.sets.get(key, ()))

    def delete(self, *keys):
        for key in keys:
            self.kv.pop(key, None)
            self.sets.pop(key, None)

    def zadd(self, key, mapping):
        self.zsets.setdefault(key, {}).update(mapping)

    def zrem(self, key, *members):
        for m in members:
            self.zsets.get(key, {}).pop(m, None)

    def zrevrange(self, key, start, end, withscores=False):
        rows = sorted(self.zsets.get(key, {}).items(), key=lambda r: -r[1])
        rows = rows[start : None if end == -1 else end + 1]
        return [(m.encode(), sc) for m, sc in rows]

    def exists(self, key):
        return int(key in self.kv)

    def pipeline(self, transaction=True):
        fake = self

        class Pipe:
            calls: list = []

            def exists(self, key):
                self.calls.append(fake.exists(key))

            def execute(self):
                return self.calls

        Pipe.calls = []
        return Pipe()

    def scan_iter(self, match, count=None):
        return []


@pytest.fixture
def redis(monkeypatch, tmp_path):
    fake = FakeRedis()
    monkeypatch.setattr(permalinks, "redis_client", fake)
    # The app's own caches (the sitemap's) must not reach a real Redis either.
    monkeypatch.setattr(app_module, "redis_client", None)
    (tmp_path / "index.html").write_text(INDEX)
    monkeypatch.setattr(app_module, "_static_dir", tmp_path)
    permalinks.index_html.cache_clear()
    yield fake
    permalinks.index_html.cache_clear()


@pytest.fixture(scope="module")
def client():
    # No `with`: the lifespan would start the MCP session manager, which can
    # only run once per process and test_agents.py already runs it.
    return TestClient(app_module.app)


def _save(query: str) -> str:
    share_id = permalinks.save(query, ANSWER, SOURCES)
    assert share_id is not None
    return share_id


def test_a_saved_answer_round_trips(redis, client):
    share_id = _save(" rust vs go ")
    assert len(share_id) == 8

    body = client.get(f"/api/answers/{share_id}").json()
    assert body["query"] == "rust vs go"
    assert body["answer"] == ANSWER
    assert body["sources"] == SOURCES
    assert client.get("/api/answers/AAAAAAAA").status_code == 404


def test_saved_answers_expire_unless_someone_opens_them(redis):
    share_id = _save("rust vs go")
    key = f"share:{share_id}"
    assert redis.ttl[key] == permalinks.SHARE_TTL
    assert redis.ttl["share:q:" + permalinks.query_hash("rust vs go")] > 0

    redis.ttl[key] = 5  # nearly expired...
    permalinks.load(share_id)  # ...until somebody opens the link
    assert redis.ttl[key] == permalinks.SHARE_TTL


def test_the_page_carries_the_answers_preview_tags(redis, client):
    share_id = permalinks.save("<script>rust</script> vs go", ANSWER, SOURCES)
    page = client.get(f"/a/{share_id}", headers={"host": "hn.example.com"}).text

    # The built head's own title and description are replaced, not doubled.
    assert page.count("<title>") == 1
    assert "Default description." not in page
    # What the visitor typed is escaped wherever it lands.
    assert "<script>rust" not in page
    assert "&lt;script&gt;rust&lt;/script&gt; vs go — HN Search" in page
    assert f'og:url" content="https://hn.example.com/a/{share_id}"' in page
    assert f'og:image" content="https://hn.example.com/a/{share_id}/og.png"' in page
    # The description is the answer as prose: no citation links, no markdown.
    assert (
        'og:description" content="Most commenters prefer Rust for this, as user '
        "alice puts it &quot;the borrow checker pays for itself&quot;. Others say "
        'Go is simpler."' in page
    )
    # A dead link is still the app (which says the answer is gone), as a 404.
    missing = client.get("/a/AAAAAAAA")
    assert missing.status_code == 404
    assert '<div id="root">' in missing.text


def test_the_preview_image_is_a_cacheable_png(redis, client):
    share_id = permalinks.save("rust vs go", ANSWER, SOURCES)
    res = client.get(f"/a/{share_id}/og.png")
    assert res.headers["content-type"] == "image/png"
    assert "max-age=86400" in res.headers["cache-control"]
    assert Image.open(io.BytesIO(res.content)).size == (1200, 630)


def test_a_non_latin_question_still_draws(redis, client):
    share_id = permalinks.save("图片无损压缩技术", ANSWER, SOURCES)
    assert client.get(f"/a/{share_id}/og.png").status_code == 200


def test_the_home_page_titles_a_shared_question(redis, client):
    page = client.get("/", params={"q": "rust vs go"}).text
    assert "<title>rust vs go — HN Search</title>" in page
    # Unvetted text never goes onto an image: the site image, not a per-query one.
    assert '/og.png" />' in page and "og.png?" not in page
    assert "<title>HN Search</title>" in client.get("/").text


def test_deleting_the_query_deletes_its_pages(redis):
    keep = _save("rust vs go")
    doomed = [_save("bad query") for _ in range(2)]

    JobManager(redis).delete_recent_query("bad query")

    assert all(permalinks.load(i) is None for i in doomed)
    assert permalinks.load(keep) is not None


def test_every_reader_of_a_job_gets_the_same_link(redis, monkeypatch):
    """The id is saved with the job's result, so a later visitor served from
    the cache gets the link the first one did, and a refusal gets none."""
    results = {}

    def search_stream(query):
        yield {"type": "sources", "sources": SOURCES}
        yield {"type": "answer", "text": ANSWER, "refused": query == "refuse me"}

    def store_result(job_id, result):
        results[job_id] = result

    monkeypatch.setattr(search, "search_stream", search_stream)
    monkeypatch.setattr(search.job_manager, "try_claim_job", lambda q: (True, q))
    monkeypatch.setattr(search.job_manager, "store_result", store_result)
    monkeypatch.setattr(search.job_manager, "track_recent_query", lambda q: None)
    monkeypatch.setattr(search.job_manager, "log_eval_record", lambda *a: None)

    events = [e.data for e in search.sse_search("rust vs go") if e.event == "answer"]
    share_id = results["rust vs go"]["share_id"]
    snapshot = permalinks.load(share_id)
    assert snapshot is not None and snapshot["answer"] == ANSWER
    assert f'"share_id": "{share_id}"' in str(events[0])

    replayed = list(search._result_events(results["rust vs go"]))
    assert f'"share_id": "{share_id}"' in str(replayed[1].data)

    list(search.sse_search("refuse me"))
    assert results["refuse me"]["share_id"] is None


def test_the_sitemap_lists_the_newest_answer_per_question(redis, client):
    monkey_time = iter(range(1_700_000_000, 1_700_000_100))
    original = permalinks.time.time
    permalinks.time.time = lambda: next(monkey_time)
    try:
        stale = _save("rust vs go")
        fresh = _save("rust vs go")  # the same question, asked again
        other = _save("sqlite in production")
        gone = _save("expired question")
    finally:
        permalinks.time.time = original
    del redis.kv[f"share:{gone}"]  # its TTL ran out

    assert [i for i, _ in permalinks.indexed(10)] == [other, fresh]
    assert gone not in redis.zsets["share:index"]  # pruned on the way

    body = client.get("/sitemap.xml", headers={"host": "hn.example.com"}).text
    assert f"<loc>https://hn.example.com/a/{fresh}</loc><lastmod>2023-11-14" in body
    assert stale not in body and gone not in body
    assert "<loc>https://hn.example.com/trends?terms=rust%2Cgo%2Czig</loc>" in body

    JobManager(redis).delete_recent_query("sqlite in production")
    assert [i for i, _ in permalinks.indexed(10)] == [fresh]


def test_robots_keep_crawlers_off_the_paid_pipeline(client):
    body = client.get("/robots.txt", headers={"host": "hn.example.com"}).text
    rules = body.splitlines()
    # /api/search runs DeepSeek; a ?q= page calls it as soon as it renders.
    assert "Disallow: /api/" in rules and "Disallow: /*?q=" in rules
    # What the saved-answer and trends pages render from stays fetchable.
    assert "Allow: /api/answers/" in rules and "Allow: /api/trends" in rules
    assert "Sitemap: https://hn.example.com/sitemap.xml" in rules
