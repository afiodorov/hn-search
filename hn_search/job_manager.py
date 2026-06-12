"""Job manager for deduplicating concurrent RAG queries using Redis."""

import hashlib
import json
import time
from typing import Any, Dict, Optional, Tuple

from hn_search.cache_config import RESULT_CACHE_TTL
from hn_search.logging_config import get_logger

logger = get_logger(__name__)


class JobManager:
    """Manages RAG query jobs to prevent duplicate processing."""

    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.job_timeout = 180  # 3 minutes (reduced from 10)
        self.result_ttl = RESULT_CACHE_TTL
        self.poll_interval = 0.5  # 500ms
        self.max_poll_time = 120  # 2 minutes (reduced from 5)

    def get_job_id(self, query: str) -> str:
        """Generate a unique job ID from query text."""
        return hashlib.md5(query.strip().encode()).hexdigest()

    def try_claim_job(self, query: str) -> Tuple[bool, str]:
        """
        Try to claim a job for processing.

        Returns:
            (claimed, job_id): claimed=True if this caller should process the job
        """
        if not self.redis:
            # No Redis - always claim (fallback to original behavior)
            return True, self.get_job_id(query)

        job_id = self.get_job_id(query)
        status_key = f"job:{job_id}:status"

        try:
            # Check current status
            status = self.redis.get(status_key)

            if status == b"completed":
                # Already completed - don't claim
                return False, job_id

            if status == b"failed":
                # Failed job - allow immediate retry by deleting it
                self.redis.delete(status_key)
                self.redis.delete(f"job:{job_id}:error")
                logger.info(f"🔄 Clearing failed job {job_id[:8]} for retry")

            # Try to claim with atomic SET NX (only if not exists)
            claimed = self.redis.set(
                status_key,
                "processing",
                nx=True,  # Only set if key doesn't exist
                ex=self.job_timeout,  # Auto-expire after timeout
            )

            return bool(claimed), job_id

        except Exception as e:
            logger.exception(f"⚠️ Job claim error: {e}")
            # Fallback: allow processing if Redis fails
            return True, job_id

    def store_result(self, job_id: str, result: Dict[str, Any]):
        """Store job result and mark as completed."""
        if not self.redis:
            return

        try:
            result_key = f"job:{job_id}:result"
            status_key = f"job:{job_id}:status"

            # Store result with TTL
            self.redis.setex(result_key, self.result_ttl, json.dumps(result))

            # Update status to completed
            self.redis.setex(status_key, self.result_ttl, "completed")

            logger.info(f"💾 Stored result for job {job_id[:8]}")

        except Exception as e:
            logger.exception(f"⚠️ Error storing result for job {job_id[:8]}: {e}")

    def store_error(self, job_id: str, error_message: str):
        """Store job error and mark as failed."""
        if not self.redis:
            return

        try:
            error_key = f"job:{job_id}:error"
            status_key = f"job:{job_id}:status"

            self.redis.setex(error_key, 3600, error_message)  # 1 hour TTL
            self.redis.setex(status_key, 3600, "failed")

            logger.info(f"💾 Stored error for job {job_id[:8]}")

        except Exception as e:
            logger.exception(f"⚠️ Error storing error for job {job_id[:8]}: {e}")

    def append_progress_event(self, job_id: str, event: Dict[str, Any]):
        """Append a progress event to the job's JSON event list.

        Single writer (the claimer), so read-modify-write is safe.
        """
        if not self.redis:
            return

        try:
            progress_key = f"job:{job_id}:progress"
            raw = self.redis.get(progress_key)
            events = json.loads(raw) if raw else []
            events.append(event)
            self.redis.setex(progress_key, 300, json.dumps(events))  # 5 min TTL
        except Exception:
            pass  # Silent fail for progress updates

    def get_progress_events(self, job_id: str) -> list:
        """Get the job's progress events (empty list if none)."""
        if not self.redis:
            return []

        try:
            progress_key = f"job:{job_id}:progress"
            raw = self.redis.get(progress_key)
            return json.loads(raw) if raw else []
        except Exception:
            return []

    def get_result(self, job_id: str) -> Optional[Dict]:
        """Get completed job result."""
        if not self.redis:
            return None

        try:
            result_key = f"job:{job_id}:result"
            result = self.redis.get(result_key)
            return json.loads(result) if result else None
        except Exception:
            return None

    def track_recent_query(self, query: str):
        """Track query in recent queries sorted set."""
        if not self.redis:
            logger.warning("⚠️ Redis not available, cannot track recent query")
            return

        try:
            # Use sorted set with timestamp as score
            timestamp = time.time()
            self.redis.zadd("recent_queries", {query: timestamp})

            # Keep only last 100 queries (trim older ones)
            self.redis.zremrangebyrank("recent_queries", 0, -101)

            logger.debug(f"✅ Tracked recent query: {query[:50]}...")

        except Exception as e:
            logger.exception(f"⚠️ Error tracking recent query: {e}")

    def get_recent_queries(self, limit: int = 10) -> list:
        """Get most recent queries with timestamps."""
        if not self.redis:
            return []

        try:
            # Get top queries in reverse order (most recent first)
            queries = self.redis.zrevrange(
                "recent_queries", 0, limit - 1, withscores=True
            )

            # Format results with human-readable timestamps
            from datetime import datetime

            result = []
            for query_bytes, timestamp in queries:
                query = query_bytes.decode("utf-8")
                dt = datetime.fromtimestamp(timestamp)
                time_ago = self._format_time_ago(timestamp)
                result.append(
                    {"query": query, "timestamp": dt.isoformat(), "time_ago": time_ago}
                )

            return result

        except Exception as e:
            logger.exception(f"⚠️ Error getting recent queries: {e}")
            return []

    def _format_time_ago(self, timestamp: float) -> str:
        """Format timestamp as human-readable 'time ago' string."""
        seconds_ago = time.time() - timestamp

        if seconds_ago < 60:
            return "just now"
        elif seconds_ago < 3600:
            minutes = int(seconds_ago / 60)
            return f"{minutes}m ago"
        elif seconds_ago < 86400:
            hours = int(seconds_ago / 3600)
            return f"{hours}h ago"
        else:
            days = int(seconds_ago / 86400)
            return f"{days}d ago"
