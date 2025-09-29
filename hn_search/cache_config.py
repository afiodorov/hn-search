"""Redis cache configuration for HN search."""

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import redis
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain_community.cache import RedisCache

# Load environment variables
load_dotenv()

# Redis configuration
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Initialize Redis client
try:
    redis_client = redis.from_url(REDIS_URL)
    # Test connection
    redis_client.ping()

    # Set up LangChain Redis cache
    cache = RedisCache(redis_client)
    set_llm_cache(cache)

    print(f"✅ Redis cache initialized at {REDIS_URL}")
except Exception as e:
    print(f"⚠️ Redis cache not available: {e}")
    print("🔄 Running without cache")
    redis_client = None


# PostgreSQL query cache functions
def get_pg_cache_key(query: str, params: tuple = ()) -> str:
    """Generate a cache key for PostgreSQL queries."""
    # Combine query and params for unique key
    cache_str = f"{query}:{str(params)}"
    return f"pg:{hashlib.md5(cache_str.encode()).hexdigest()}"


def get_cached_pg_results(
    query: str, params: tuple = ()
) -> Optional[List[Dict[str, Any]]]:
    """Get cached PostgreSQL query results."""
    if not redis_client:
        return None
    try:
        cache_key = get_pg_cache_key(query, params)
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass
    return None


def cache_pg_results(
    query: str, results: List[Dict[str, Any]], params: tuple = (), ttl: int = 3600
):
    """Cache PostgreSQL query results with TTL (default 1 hour)."""
    if not redis_client:
        return
    try:
        cache_key = get_pg_cache_key(query, params)
        redis_client.setex(cache_key, ttl, json.dumps(results))
    except Exception:
        pass


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


def cache_vector_search(
    query: str, results: List[Dict[str, Any]], k: int = 10, ttl: int = 21600
):
    """Cache vector search results with TTL (default 6 hours)."""
    if not redis_client:
        return
    try:
        cache_key = get_vector_cache_key(query, k)
        redis_client.setex(cache_key, ttl, json.dumps(results))
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


def cache_answer(query: str, context: str, answer: str, ttl: int = 21600):
    """Cache LLM answer with TTL (default 6 hours)."""
    if not redis_client:
        return
    try:
        cache_key = get_answer_cache_key(query, context)
        redis_client.setex(cache_key, ttl, answer)
    except Exception:
        pass


# Clear cache utility
def clear_cache(pattern: str = "*"):
    """Clear cache entries matching pattern."""
    if not redis_client:
        return
    try:
        keys = redis_client.keys(pattern)
        if keys:
            redis_client.delete(*keys)
            print(f"🗑️ Cleared {len(keys)} cache entries")
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")
