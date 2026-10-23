"""A search outlives the request that started it.

Clicking away mid-search (another query, a cached answer, a closed tab) used to
stop the pipeline where it stood, leaving the job claimed but never finished —
and every later request for the same question waited on it, then failed.
"""

import threading

from hn_search.api import search


def test_a_search_finishes_after_its_client_leaves(monkeypatch):
    release = threading.Event()
    stored = threading.Event()
    results = {}

    def search_stream(query):
        yield {"type": "progress", "step": "guard", "status": "start", "ms": None}
        release.wait(5)
        yield {"type": "sources", "sources": []}
        yield {"type": "answer", "text": "An answer.", "refused": False}

    def store_result(job_id, result):
        results[job_id] = result
        stored.set()

    monkeypatch.setattr(search, "search_stream", search_stream)
    monkeypatch.setattr(search.job_manager, "try_claim_job", lambda q: (True, "j1"))
    monkeypatch.setattr(search.job_manager, "store_result", store_result)
    monkeypatch.setattr(search.job_manager, "track_recent_query", lambda q: None)
    monkeypatch.setattr(search.job_manager, "log_eval_record", lambda *a: None)

    stream = search.sse_search("rust vs go")
    next(stream)  # the first progress event reaches the client...
    stream.close()  # ...which then navigates away
    release.set()

    assert stored.wait(5)
    assert results["j1"]["answer"] == "An answer."
