"""The updater's HN API fetch: which ids it keeps, and where it must stop.

The service stores rows in id order and finds date windows by binary search, and
each run resumes from its max_id. So a fetch has to hand back a contiguous id
range: stopping is always safe, skipping past an id is not.
"""

import sys
import time
from pathlib import Path

import httpx
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "misc"))
import fetch_and_embed_new_comments as updater  # noqa: E402  # pyright: ignore[reportMissingImports]

OLD = time.time() - 6 * 3600
NEW = time.time() - 600  # still inside the edit window


def comment(i, t=OLD, **kw):
    return {
        "id": i,
        "type": "comment",
        "by": "pg",
        "text": f"c{i}",
        "time": t,
        "parent": 1,
        **kw,
    }


@pytest.fixture
def api(monkeypatch):
    """Serve `items` as the API; ids in `failing` never fetch."""
    items: dict = {}
    failing: set = set()

    class MaxItem:
        def json(self):
            return max(items)

    async def fetch_items(ids, concurrency, attempts=4):
        for i in ids:
            if i in failing:
                raise updater.FetchStopped(i, {k: items.get(k) for k in ids if k < i})
        return {i: items.get(i) for i in ids}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: MaxItem())
    monkeypatch.setattr(updater, "_fetch_items", fetch_items)
    return items, failing


def fetched_ids(path):
    return pd.read_parquet(path)["id"].tolist() if path else []


def test_keeps_live_comments_and_stops_at_the_edit_window(api, tmp_path):
    items, _ = api
    items.update(
        {
            11: comment(11),
            12: {"id": 12, "type": "story", "time": OLD},
            13: comment(13, dead=True),
            14: comment(14, deleted=True),
            15: None,
            16: comment(16),
            17: comment(17, t=NEW),  # too fresh: the run ends here...
            18: comment(18),  # ...even though this one looks old enough
        }
    )

    path, stopped = updater.fetch_from_hn_api(10, output_dir=tmp_path, batch=3)

    assert fetched_ids(path) == [11, 16]
    assert not stopped


def test_an_id_that_wont_fetch_is_never_skipped(api, tmp_path):
    items, failing = api
    items.update({i: comment(i) for i in range(11, 21)})
    failing.add(15)

    path, stopped = updater.fetch_from_hn_api(10, output_dir=tmp_path, batch=4)

    assert fetched_ids(path) == [11, 12, 13, 14]
    assert stopped
