"""Job manager for deduplicating concurrent RAG queries using Redis."""

import hashlib
import json
import time
from typing import Any, Dict, Optional, Tuple

from hn_search.cache_config import redis_client


class JobManager:
    """Manages RAG query jobs to prevent duplicate processing."""

    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.job_timeout = 600  # 10 minutes
        self.result_ttl = 21600  # 6 hours
        self.poll_interval = 0.5  # 500ms
        self.max_poll_time = 300  # 5 minutes

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
            # Check if already completed
            status = self.redis.get(status_key)
            if status == b"completed":
                return False, job_id

            # Try to claim with atomic SET NX (only if not exists)
            claimed = self.redis.set(
                status_key,
                "processing",
                nx=True,  # Only set if key doesn't exist
                ex=self.job_timeout,  # Auto-expire after timeout
            )

            return bool(claimed), job_id

        except Exception as e:
            print(f"⚠️ Job claim error: {e}")
            # Fallback: allow processing if Redis fails
            return True, job_id

    def wait_for_job(
        self, job_id: str, timeout: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Wait for another process to complete the job.

        Returns:
            Result dict if job completes, None if timeout or error
        """
        if not self.redis:
            return None

        timeout = timeout or self.max_poll_time
        start_time = time.time()
        result_key = f"job:{job_id}:result"
        status_key = f"job:{job_id}:status"
        error_key = f"job:{job_id}:error"

        print(f"⏳ Waiting for job {job_id[:8]}... (another request is processing)")

        while time.time() - start_time < timeout:
            try:
                status = self.redis.get(status_key)

                if status == b"completed":
                    result = self.redis.get(result_key)
                    if result:
                        print(f"✅ Job {job_id[:8]} completed by another request")
                        return json.loads(result)

                elif status == b"failed":
                    error = self.redis.get(error_key)
                    error_msg = error.decode() if error else "Unknown error"
                    print(f"❌ Job {job_id[:8]} failed: {error_msg}")
                    return None

                elif status is None:
                    # Job disappeared (expired or deleted)
                    print(f"⚠️ Job {job_id[:8]} disappeared")
                    return None

                # Still processing - wait and retry
                time.sleep(self.poll_interval)

            except Exception as e:
                print(f"⚠️ Error polling job {job_id[:8]}: {e}")
                return None

        print(f"⏱️ Timeout waiting for job {job_id[:8]}")
        return None

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

            print(f"💾 Stored result for job {job_id[:8]}")

        except Exception as e:
            print(f"⚠️ Error storing result for job {job_id[:8]}: {e}")

    def store_error(self, job_id: str, error_message: str):
        """Store job error and mark as failed."""
        if not self.redis:
            return

        try:
            error_key = f"job:{job_id}:error"
            status_key = f"job:{job_id}:status"

            self.redis.setex(error_key, 3600, error_message)  # 1 hour TTL
            self.redis.setex(status_key, 3600, "failed")

            print(f"💾 Stored error for job {job_id[:8]}")

        except Exception as e:
            print(f"⚠️ Error storing error for job {job_id[:8]}: {e}")

    def update_progress(self, job_id: str, progress: str):
        """Update job progress for streaming updates."""
        if not self.redis:
            return

        try:
            progress_key = f"job:{job_id}:progress"
            self.redis.setex(progress_key, 300, progress)  # 5 min TTL
        except Exception:
            pass  # Silent fail for progress updates

    def get_progress(self, job_id: str) -> Optional[str]:
        """Get current job progress."""
        if not self.redis:
            return None

        try:
            progress_key = f"job:{job_id}:progress"
            progress = self.redis.get(progress_key)
            return progress.decode() if progress else None
        except Exception:
            return None

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
