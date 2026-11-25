"""The per-IP limit on new questions (`ratelimit.py`).

Redis is an in-memory stand-in with the sorted-set commands the limiter uses.
What is under test: who a request is counted as (forged headers must not
help), the sliding hour, that only new pipeline runs are counted, and that a
refused claim is given back.
"""

import json

import pytest

from hn_search import ratelimit
from hn_search.api import search

CF_EDGE = "172.71.10.20"  # inside Cloudflare's 172.64.0.0/13


class FakeRedis:
    def __init__(self):
        self.z: dict[str, dict[str, float]] = {}

    def zremrangebyscore(self, key, lo, hi):
        self.z[key] = {
            m: s for m, s in self.z.get(key, {}).items() if not lo <= s <= hi
        }

    def zrange(self, key, start, end, withscores=False):
        return sorted(self.z.get(key, {}).items(), key=lambda r: r[1])

    def zadd(self, key, mapping):
        self.z.setdefault(key, {}).update(mapping)

    def expire(self, key, seconds):
        pass


@pytest.fixture
def redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(ratelimit, "redis_client", fake)
    return fake


def test_the_visitor_is_the_hop_our_proxy_added():
    # Railway appends the peer it saw; anything before it is client-supplied.
    headers = {"X-Forwarded-For": "6.6.6.6, 203.0.113.7"}
    assert ratelimit.client_ip(headers, "10.0.0.1") == "203.0.113.7"
    assert ratelimit.client_ip({}, "10.0.0.1") == "10.0.0.1"


def test_cloudflares_visitor_header_counts_only_from_cloudflare():
    via_cf = {
        "x-forwarded-for": f"6.6.6.6, {CF_EDGE}",
        "cf-connecting-ip": "198.51.100.9",
    }
    assert ratelimit.client_ip(via_cf, None) == "198.51.100.9"

    # Straight to the railway.app address: the header is forged, not trusted.
    direct = {"x-forwarded-for": "203.0.113.7", "cf-connecting-ip": "198.51.100.9"}
    assert ratelimit.client_ip(direct, None) == "203.0.113.7"


def test_twenty_an_hour_then_wait_for_the_oldest_to_age_out(redis):
    t0 = 1_000_000.0
    for i in range(ratelimit.LIMIT):
        verdict = ratelimit.take("1.2.3.4", now=t0 + i)
        assert verdict.allowed and verdict.remaining == ratelimit.LIMIT - 1 - i

    over = ratelimit.take("1.2.3.4", now=t0 + 600)
    assert not over.allowed
    assert over.retry_after == ratelimit.WINDOW - 600 + 1
    assert "Try again in 51 minutes" in over.message()  # 3001 s, rounded up
    assert ratelimit.take("5.6.7.8", now=t0 + 600).allowed  # someone else

    # An hour after the first question, one slot frees up — just one.
    assert ratelimit.take("1.2.3.4", now=t0 + ratelimit.WINDOW + 0.5).allowed
    assert not ratelimit.take("1.2.3.4", now=t0 + ratelimit.WINDOW + 0.6).allowed


def _events(stream):
    return [(e.event, json.loads(e.data)) for e in stream]


def test_only_new_runs_are_counted(redis, monkeypatch):
    released = []
    ran = []

    def search_stream(query):
        ran.append(query)
        yield {"type": "sources", "sources": []}
        yield {"type": "answer", "text": "An answer.", "refused": False}

    monkeypatch.setattr(search, "search_stream", search_stream)
    monkeypatch.setattr(search.job_manager, "release_job", released.append)
    monkeypatch.setattr(search.job_manager, "store_result", lambda *a: None)
    monkeypatch.setattr(search.job_manager, "track_recent_query", lambda q: None)
    monkeypatch.setattr(search.job_manager, "log_eval_record", lambda *a: None)
    monkeypatch.setattr(search.permalinks, "save", lambda *a: None)

    for i in range(ratelimit.LIMIT):
        ratelimit.take("1.2.3.4")

    # A cached answer costs nothing, so it is served at the limit.
    monkeypatch.setattr(search.job_manager, "try_claim_job", lambda q: (False, "j"))
    monkeypatch.setattr(
        search.job_manager, "get_result", lambda j: {"answer": "Cached.", "sources": []}
    )
    monkeypatch.setattr(search.job_manager, "get_progress_events", lambda j: [])
    events = _events(search.sse_search("cached question", "1.2.3.4"))
    assert (
        "answer",
        {"type": "answer", "text": "Cached.", "refused": False, "share_id": None},
    ) in events

    # A new run is refused, and its claim given back for the next asker.
    monkeypatch.setattr(search.job_manager, "try_claim_job", lambda q: (True, "j2"))
    events = _events(search.sse_search("new question", "1.2.3.4"))
    assert events[0][0] == "error" and "Try again in" in events[0][1]["message"]
    assert released == ["j2"] and ran == []

    # Someone else asking it runs it.
    events = _events(search.sse_search("new question", "5.6.7.8"))
    assert ran == ["new question"]
