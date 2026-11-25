"""Shareable answer pages: a saved snapshot per answer, and its link preview.

The job cache (`job_manager.py`) holds an answer for 12 hours under its query,
which is right for deduplicating work and wrong for a link someone pasted into
a thread: a `?q=` link reruns the whole pipeline once the cache lapses, and
may come back with a different answer than the one being discussed. So every
admitted answer is also saved under a short random id, and `/a/<id>` shows
exactly that answer. Saved answers expire after SHARE_TTL of nobody opening
them: every view renews the clock, so a link still in use keeps working and
the rest clean themselves up.

Stored zlib-compressed (answer + up to ~10 source comments is ~20 KB of JSON,
~6 KB compressed). Refusals are never saved: there is nothing worth sharing in
"I only answer questions about HN", and the guard's refusals often quote the
input it refused.

The page itself is the SPA's index.html with link-preview tags written into
its head (crawlers run no JavaScript), and an image drawn by Pillow — PNG,
because Twitter, Slack and Facebook ignore SVG previews.
"""

import functools
import html
import io
import json
import re
import secrets
import time
import zlib
from pathlib import Path
from typing import Any, cast

from hn_search.cache_config import query_hash, redis_client
from hn_search.logging_config import get_logger

logger = get_logger(__name__)

SHARE_TTL = 90 * 86400  # 90 days since the last view

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8}$")

SITE_NAME = "HN Search"
SITE_DESCRIPTION = (
    "Ask questions about Hacker News discussions and get AI-powered answers "
    "with sources."
)


def _key(share_id: str) -> str:
    return f"share:{share_id}"


def _query_key(query: str) -> str:
    # Every snapshot of a query, so deleting the query can forget them too.
    return f"share:q:{query_hash(query)}"


def save(query: str, answer: str, sources: list[dict], redis=None) -> str | None:
    """Save an answer under a new id and return the id, or None without Redis
    (or when Redis fails: a missing permalink must never fail the search)."""
    redis = redis if redis is not None else redis_client
    if redis is None:
        return None
    share_id = secrets.token_urlsafe(6)  # 6 bytes -> 8 url-safe characters
    snapshot = {
        "id": share_id,
        "query": query.strip(),
        "answer": answer,
        "sources": sources,
        "created_at": time.time(),
    }
    try:
        redis.set(
            _key(share_id),
            zlib.compress(json.dumps(snapshot).encode()),
            ex=SHARE_TTL,
        )
        # The id list outlives every id on it: renewed on each save, and a
        # stale id in it only costs forget_query a no-op delete.
        redis.sadd(_query_key(query), share_id)
        redis.expire(_query_key(query), SHARE_TTL)
    except Exception as e:
        logger.exception(f"⚠️ Could not save permalink: {e}")
        return None
    return share_id


def load(share_id: str, redis=None) -> dict[str, Any] | None:
    redis = redis if redis is not None else redis_client
    if redis is None or not _ID_RE.match(share_id):
        return None
    try:
        raw = cast(bytes | None, redis.get(_key(share_id)))
        if not raw:
            return None
        redis.expire(_key(share_id), SHARE_TTL)  # a view keeps it alive
        return json.loads(zlib.decompress(raw))
    except Exception as e:
        logger.exception(f"⚠️ Could not load permalink {share_id}: {e}")
        return None


def forget_query(query: str, redis) -> None:
    """Delete every saved snapshot of `query` (an admin deleting it from the
    recent list wants it gone, not merely unlisted)."""
    qkey = _query_key(query)
    ids = [i.decode() if isinstance(i, bytes) else i for i in redis.smembers(qkey)]
    redis.delete(qkey, *(_key(i) for i in ids))


# ---- Link previews ---------------------------------------------------------


def plain_text(markdown: str) -> str:
    """Markdown down to one line of prose, for a preview's description.

    Citation links (`[[1]](https://...)`) go entirely; other links keep their
    text; emphasis, headings, quote and list markers are dropped.
    """
    text = re.sub(r"\s*\[\[\d+\]\]\([^)]*\)", "", markdown)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"`{1,3}", "", text)
    text = re.sub(r"^\s{0,3}(#{1,6}|>|[-*+]|\d+\.)\s+", "", text, flags=re.M)
    text = re.sub(r"(\*\*|__|\*|_)(\S.*?\S|\S)\1", r"\2", text)
    return re.sub(r"\s+", " ", text).strip()


def summary(markdown: str, limit: int = 200) -> str:
    text = plain_text(markdown)
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:.") + "…"


def meta_tags(
    *, title: str, description: str, url: str, image: str, large: bool = True
) -> str:
    """The <head> tags crawlers read. Every value is escaped: titles here are
    whatever a visitor typed."""
    e = functools.partial(html.escape, quote=True)
    card = "summary_large_image" if large else "summary"
    return "\n    ".join(
        [
            f"<title>{e(title)}</title>",
            f'<meta name="description" content="{e(description)}" />',
            f'<link rel="canonical" href="{e(url)}" />',
            f'<meta property="og:site_name" content="{SITE_NAME}" />',
            '<meta property="og:type" content="article" />',
            f'<meta property="og:title" content="{e(title)}" />',
            f'<meta property="og:description" content="{e(description)}" />',
            f'<meta property="og:url" content="{e(url)}" />',
            f'<meta property="og:image" content="{e(image)}" />',
            '<meta property="og:image:width" content="1200" />',
            '<meta property="og:image:height" content="630" />',
            f'<meta name="twitter:card" content="{card}" />',
            f'<meta name="twitter:title" content="{e(title)}" />',
            f'<meta name="twitter:description" content="{e(description)}" />',
            f'<meta name="twitter:image" content="{e(image)}" />',
        ]
    )


# The built index.html's own title and description, replaced by meta_tags().
_HEAD_DEFAULTS = re.compile(
    r"\s*<title>.*?</title>|\s*<meta\s+name=\"description\"[^>]*>", re.S
)


def render_page(index_html: str, tags: str) -> str:
    head = _HEAD_DEFAULTS.sub("", index_html)
    return head.replace("</head>", f"  {tags}\n  </head>", 1)


# ---- Preview image ---------------------------------------------------------

W, H = 1200, 630
ACCENT = (255, 102, 0)
BG = (250, 249, 247)
TEXT = (28, 29, 32)
MUTED = (113, 118, 127)


@functools.cache
def _font(size: int):
    from PIL import ImageFont

    # Pillow's built-in scalable font (no font files in the image). It only
    # covers Latin, which is why non-Latin titles are left off the image.
    return ImageFont.load_default(size=size)


def _drawable(text: str) -> bool:
    return all(ord(c) < 0x250 or c in "–—‘’“”…•" for c in text)


def _wrap(draw, text: str, font, width: int, max_lines: int) -> list[str]:
    """Greedy word wrap by rendered width, ellipsised at max_lines."""
    lines: list[str] = []
    line = ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= width:
            line = trial
            continue
        if line:
            lines.append(line)
        line = word
        if len(lines) == max_lines:
            break
    if line and len(lines) < max_lines:
        lines.append(line)
    consumed = " ".join(lines)
    if len(consumed) < len(" ".join(text.split())):
        last = lines[-1]
        while last and draw.textlength(last + "…", font=font) > width:
            last = last.rsplit(" ", 1)[0] if " " in last else last[:-1]
        lines[-1] = last.rstrip(",;:.") + "…"
    return lines


def og_image(title: str, subtitle: str, footer: str) -> bytes:
    """A 1200x630 PNG: brand mark, the question, a line of the answer."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 12], fill=ACCENT)

    # The site's logo: an orange tile with a white Y, then the name.
    x, y = 72, 64
    d.rectangle([x, y, x + 56, y + 56], fill=ACCENT)
    d.text((x + 28, y + 28), "Y", font=_font(40), fill="white", anchor="mm")
    d.text((x + 76, y + 28), SITE_NAME, font=_font(34), fill=TEXT, anchor="lm")

    y = 172
    if title and _drawable(title):
        size = 58 if len(title) < 70 else 48
        title_font = _font(size)
        for line in _wrap(d, title, title_font, W - 144, 3):
            d.text((72, y), line, font=title_font, fill=TEXT)
            y += int(size * 1.22)
        y += 18
    if subtitle and _drawable(subtitle):
        body = _font(30)
        for line in _wrap(d, subtitle, body, W - 144, max(1, (520 - y) // 42)):
            d.text((72, y), line, font=body, fill=MUTED)
            y += 42

    d.text((72, H - 64), footer, font=_font(26), fill=ACCENT, anchor="lm")

    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


@functools.cache
def index_html(static_dir: Path) -> str | None:
    path = static_dir / "index.html"
    return path.read_text() if path.is_file() else None
