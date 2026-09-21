"""Deleting a recent search forgets everything cached for it.

Deleting only the job result used to leave the search results and the answer
cached under the query, so its next run rebuilt exactly what was deleted. Redis
is an in-memory stand-in with just the commands the job manager uses.
"""

import fnmatch

from hn_search.cache_config import get_answer_cache_key, get_vector_cache_key
from hn_search.job_manager import JobManager


class _FakeRedis:
    def __init__(self, keys):
        self.store = dict.fromkeys(keys, b"x")
        self.zset = {}

    def zrem(self, name, member):
        self.zset.pop(member, None)

    def delete(self, *keys):
        for key in keys:
            self.store.pop(key, None)

    def scan_iter(self, match, count=None):
        return [k for k in list(self.store) if fnmatch.fnmatchcase(k, match)]


def test_deleting_a_query_forgets_its_searches_and_answers():
    query = "图片无损压缩技术"
    other = "rust vs go"
    doomed = [
        get_vector_cache_key(query, 10),
        get_vector_cache_key(query, 3),
        get_vector_cache_key(query, 10, "2025-01-01", None),
        get_answer_cache_key(query, "context one"),
        get_answer_cache_key(query, "context two"),
    ]
    kept = [
        get_vector_cache_key(other, 10),
        get_answer_cache_key(other, "context one"),
    ]
    redis = _FakeRedis(doomed + kept)
    manager = JobManager(redis)
    job_id = manager.get_job_id(query)
    redis.store[f"job:{job_id}:result"] = b"x"

    # The admin deletes the row as the list shows it; surrounding whitespace
    # is the same query, as it is for the job id.
    manager.delete_recent_query(f" {query} ")

    assert sorted(redis.store) == sorted(kept)
