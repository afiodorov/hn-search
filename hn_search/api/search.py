"""SSE search flow: job dedup + attach, streaming pipeline events to the client.

Port of the Gradio hn_search_rag generator: the first request for a query claims
the job and runs the pipeline (mirroring progress to Redis so attachers can
follow along); concurrent requests for the same query attach and poll, getting
mirrored progress then the full answer in one shot (tokens are SSE-only).
"""

import json
import time

from sse_starlette import ServerSentEvent

from hn_search.cache_config import redis_client
from hn_search.job_manager import JobManager
from hn_search.logging_config import get_logger
from hn_search.rag.pipeline import search_stream

logger = get_logger(__name__)

job_manager = JobManager(redis_client)


def _sse(event: dict) -> ServerSentEvent:
    return ServerSentEvent(event=event["type"], data=json.dumps(event))


def _done() -> ServerSentEvent:
    return ServerSentEvent(event="done", data="{}")


def sse_search(query: str):
    claimed, job_id = job_manager.try_claim_job(query)
    job_manager.track_recent_query(query)

    if claimed:
        yield from _process(query, job_id)
    elif job_manager.get_result(job_id):
        # Job already completed (claim refused, result stored): replay through
        # the pipeline. Its Redis caches make this near-instant and the client
        # gets real progress events instead of none.
        yield from _replay(query)
    else:
        yield from _attach(query, job_id)


def _replay(query: str):
    try:
        for event in search_stream(query):
            yield _sse(event)
            if event["type"] == "error":
                break
    except Exception as e:
        logger.exception(f"Replay failed for: {query}")
        yield _sse({"type": "error", "message": str(e)})
    yield _done()


def _process(query: str, job_id: str):
    answer = ""
    sources = []

    try:
        for event in search_stream(query):
            etype = event["type"]
            if etype == "progress":
                job_manager.append_progress_event(job_id, event)
            elif etype == "sources":
                sources = event["sources"]
            elif etype == "answer":
                answer = event["text"]
            elif etype == "error":
                job_manager.store_error(job_id, event["message"])
                yield _sse(event)
                yield _done()
                return
            yield _sse(event)

        job_manager.store_result(job_id, {"answer": answer, "sources": sources})
        yield _done()
    except Exception as e:
        logger.exception(f"Search failed for: {query}")
        job_manager.store_error(job_id, str(e))
        yield _sse({"type": "error", "message": str(e)})
        yield _done()


def _attach(query: str, job_id: str):
    """Follow a job another request is processing: mirror progress, then result."""
    logger.info(f"⏳ Attaching to in-flight job {job_id[:8]} for: {query}")
    sent = 0
    start_time = time.time()

    while time.time() - start_time < job_manager.max_poll_time:
        events = job_manager.get_progress_events(job_id)
        for event in events[sent:]:
            yield _sse(event)
        sent = len(events)

        result = job_manager.get_result(job_id)
        if result:
            yield _sse({"type": "sources", "sources": result.get("sources", [])})
            yield _sse({"type": "answer", "text": result.get("answer", "")})
            yield _done()
            return

        time.sleep(job_manager.poll_interval)

    # Timeout waiting on the other request - try to process ourselves
    logger.warning(f"⏱️ Timeout attached to job {job_id[:8]}, claiming it")
    claimed, job_id = job_manager.try_claim_job(query)
    if claimed:
        yield from _process(query, job_id)
    else:
        yield _sse({"type": "error", "message": "Unable to process query"})
        yield _done()
