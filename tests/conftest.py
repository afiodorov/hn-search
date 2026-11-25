import pytest

from hn_search import permalinks, ratelimit


@pytest.fixture(autouse=True)
def no_saved_answers_in_real_redis(monkeypatch):
    """Any search a test runs saves its answer's permalink (`permalinks.py`),
    to whatever Redis the machine has. Tests that care swap in a fake."""
    monkeypatch.setattr(permalinks, "redis_client", None)


@pytest.fixture(autouse=True)
def no_rate_limits_in_real_redis(monkeypatch):
    """Every question a test asks would otherwise be counted, per IP, in
    whatever Redis the machine has. test_ratelimit.py swaps in a fake."""
    monkeypatch.setattr(ratelimit, "redis_client", None)
