"""Per-IP limit on questions that start a paid pipeline run.

A question costs DeepSeek calls (the guard, the planner's rounds, the answer),
and the site is public and unauthenticated, as is the MCP `ask` tool. So each
client IP gets LIMIT new runs per sliding WINDOW. Only new runs count: a
question whose answer is cached or already being worked on costs nothing and
is never limited, nor are saved answers, trends, search or the other tools.

The window is a Redis sorted set of run timestamps per IP. Check-then-add is
not atomic, so two requests racing at the very edge can both get in; for a
cost guard that is fine. Without Redis there is no limit (local dev).

The client IP: the socket peer is Railway's proxy (or Caddy on staging), so
the address comes from X-Forwarded-For — its *last* entry, the one the proxy
in front of us appended; earlier entries are whatever the client sent. In
production that last hop is a Cloudflare edge (hn.fiodorov.es is proxied by
Cloudflare), shared by many visitors, and the visitor is in CF-Connecting-IP.
That header is trusted only when the last hop really is Cloudflare: the
service also answers on its railway.app address, where anyone can send it.
"""

import ipaddress
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from hn_search.cache_config import redis_client
from hn_search.logging_config import get_logger

logger = get_logger(__name__)

LIMIT = 20
WINDOW = 3600  # seconds


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    remaining: int  # runs left in the window after this one
    retry_after: int  # seconds until the next run is allowed (0 if allowed)

    def message(self) -> str:
        minutes = max(1, -(-self.retry_after // 60))
        return (
            f"That's {LIMIT} new questions from you in the last hour, the most "
            f"this free site allows. Try again in {minutes} minute"
            f"{'s' if minutes != 1 else ''}. Saved answers, questions others "
            "have just asked, and Trends still work meanwhile."
        )


# https://www.cloudflare.com/ips-v4 and /ips-v6, as of 2026-11-25. Stable
# for years; a new range missing here only means its visitors share a quota.
CLOUDFLARE = [
    ipaddress.ip_network(n)
    for n in (
        "173.245.48.0/20 103.21.244.0/22 103.22.200.0/22 103.31.4.0/22 "
        "141.101.64.0/18 108.162.192.0/18 190.93.240.0/20 188.114.96.0/20 "
        "197.234.240.0/22 198.41.128.0/17 162.158.0.0/15 104.16.0.0/13 "
        "104.24.0.0/14 172.64.0.0/13 131.0.72.0/22 2400:cb00::/32 "
        "2606:4700::/32 2803:f800::/32 2405:b500::/32 2405:8100::/32 "
        "2a06:98c0::/29 2c0f:f248::/32"
    ).split()
]


def _is_cloudflare(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    return any(ip in net for net in CLOUDFLARE)


def client_ip(headers: Mapping[str, str] | None, peer: str | None) -> str:
    headers = {k.lower(): v for k, v in (headers or {}).items()}
    hops = [h.strip() for h in headers.get("x-forwarded-for", "").split(",")]
    hops = [h for h in hops if h]
    last = hops[-1] if hops else (peer or "unknown")
    visitor = headers.get("cf-connecting-ip", "").strip()
    if visitor and _is_cloudflare(last):
        return visitor
    return last


def _key(ip: str) -> str:
    return f"rl:ask:{ip}"


def peek(ip: str, redis=None, now: float | None = None) -> Verdict:
    """Where `ip` stands, without using a run."""
    redis = redis if redis is not None else redis_client
    if redis is None:
        return Verdict(True, LIMIT, 0)
    now = time.time() if now is None else now
    key = _key(ip)
    try:
        redis.zremrangebyscore(key, 0, now - WINDOW)
        used = cast(
            list[tuple[bytes, float]], redis.zrange(key, 0, -1, withscores=True)
        )
    except Exception as e:
        logger.warning(f"rate limit unavailable, allowing: {e}")
        return Verdict(True, LIMIT, 0)
    if len(used) < LIMIT:
        return Verdict(True, LIMIT - len(used), 0)
    oldest = min(score for _, score in used)
    return Verdict(False, 0, int(oldest + WINDOW - now) + 1)


def take(ip: str, redis=None, now: float | None = None) -> Verdict:
    """Use one run for `ip` if it has one left; the verdict says whether it
    did, and how many remain after."""
    redis = redis if redis is not None else redis_client
    now = time.time() if now is None else now
    verdict = peek(ip, redis, now)
    if not verdict.allowed or redis is None:
        return verdict
    key = _key(ip)
    try:
        redis.zadd(key, {f"{now}:{uuid.uuid4().hex[:8]}": now})
        redis.expire(key, WINDOW)
    except Exception as e:
        logger.warning(f"rate limit unavailable, allowing: {e}")
    return Verdict(True, verdict.remaining - 1, 0)
