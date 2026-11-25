import pytest

from hn_search import permalinks


@pytest.fixture(autouse=True)
def no_saved_answers_in_real_redis(monkeypatch):
    """Any search a test runs saves its answer's permalink (`permalinks.py`),
    to whatever Redis the machine has. Tests that care swap in a fake."""
    monkeypatch.setattr(permalinks, "redis_client", None)
