"""Redis cache configuration for HN search."""

import hashlib
import json
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import redis
from dotenv import load_dotenv

from hn_search.logging_config import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__)


def sanitize_url(url: str) -> str:
    """Sanitize URL to hide credentials."""
    try:
        parsed = urlparse(url)
        if parsed.password:
            # Replace password with asterisks
            sanitized = parsed._replace(
                netloc=f"{parsed.username}:***@{parsed.hostname}:{parsed.port}"
                if parsed.port
                else f"{parsed.username}:***@{parsed.hostname}"
            )
            return sanitized.geturl()
        return url
    except Exception:
        return "redis://***"


# Redis configuration
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Job results, vector search, and answer caches must expire together: the SSE
# replay path assumes a completed job implies warm pipeline caches.
RESULT_CACHE_TTL = 43200  # 12 hours

# Initialize Redis client
try:
    redis_client = redis.from_url(REDIS_URL)
    # Test connection
    redis_client.ping()

    logger.info(f"✅ Redis cache initialized at {sanitize_url(REDIS_URL)}")
except Exception as e:
    logger.warning(f"⚠️ Redis cache not available: {e}")
    logger.warning("🔄 Running without cache")
    redis_client = None


# Vector search cache functions
def get_vector_cache_key(query: str, k: int = 10) -> str:
    """Generate a cache key for vector search queries."""
    return f"vector:{hashlib.md5(f'{query}:{k}'.encode()).hexdigest()}"


def get_cached_vector_search(query: str, k: int = 10) -> Optional[List[Dict[str, Any]]]:
    """Get cached vector search results."""
    if not redis_client:
        return None
    try:
        cache_key = get_vector_cache_key(query, k)
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass
    return None


def cache_vector_search(query: str, results: List[Dict[str, Any]], k: int = 10):
    """Cache vector search results."""
    if not redis_client:
        return
    try:
        cache_key = get_vector_cache_key(query, k)
        redis_client.setex(cache_key, RESULT_CACHE_TTL, json.dumps(results))
    except Exception:
        pass


# LangChain answer cache functions
def get_answer_cache_key(query: str, context: str) -> str:
    """Generate a cache key for LLM answers."""
    # Hash context to keep key manageable
    context_hash = hashlib.md5(context.encode()).hexdigest()
    return f"answer:{hashlib.md5(f'{query}:{context_hash}'.encode()).hexdigest()}"


def get_cached_answer(query: str, context: str) -> Optional[str]:
    """Get cached LLM answer."""
    if not redis_client:
        return None
    try:
        cache_key = get_answer_cache_key(query, context)
        cached = redis_client.get(cache_key)
        if cached:
            return cached.decode("utf-8")
    except Exception:
        pass
    return None


def cache_answer(query: str, context: str, answer: str):
    """Cache LLM answer."""
    if not redis_client:
        return
    try:
        cache_key = get_answer_cache_key(query, context)
        redis_client.setex(cache_key, RESULT_CACHE_TTL, answer)
    except Exception:
        pass
