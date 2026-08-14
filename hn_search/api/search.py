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
    else:
        result = job_manager.get_result(job_id)
        if result:
            # Job already completed: serve the stored result directly. This
            # used to re-run search_stream(query) on the assumption that its
            # own Redis caches would make that near-instant — true for the old
            # deterministic pipeline, but the agentic planner call is never
            # cached and isn't perfectly reproducible, so "replaying" it was
            # silently redoing real (slow) work instead of reusing the answer.
            yield from _replay(job_id, result)
        else:
            yield from _attach(query, job_id)


def _result_events(result: dict):
    yield _sse({"type": "sources", "sources": result.get("sources", [])})
    yield _sse({"type": "answer", "text": result.get("answer", "")})


def _replay(job_id: str, result: dict):
    """Serve an already-completed job's stored result. Replays its stored
    progress events too, when they haven't expired (5 min TTL), for the same
    step-by-step UI as a live run; falls back to just sources+answer after."""
    for event in job_manager.get_progress_events(job_id):
        yield _sse(event)
    yield from _result_events(result)
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
        job_manager.log_eval_record(query, sources, answer)
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
            yield from _result_events(result)
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
