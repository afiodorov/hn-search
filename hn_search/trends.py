"""HN trends: how often a few terms are mentioned, month by month.

A thin layer over the full-text index behind `/keyword` (exact counts of
comments containing a whole word or phrase, per month, with each month's total
comment volume): up to MAX_TERMS terms side by side, cached for an hour (the
corpus grows hourly), and a PNG chart for link previews.

Rates, not counts, are what the chart plots: monthly volume swings by ±10%,
and the current month is partial, so "per 10,000 comments" is the honest
comparison across months.

What gets drawn on the preview image or put in a title is restricted to terms
with at least MIN_PREVIEW_COUNT mentions: those are words Hacker News itself
uses, rather than arbitrary text someone wants stamped onto this site's image.
"""

import hashlib
import io
import json
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

from hn_search.cache_config import redis_client
from hn_search.logging_config import get_logger

logger = get_logger(__name__)

MAX_TERMS = 5
MAX_TERM_LEN = 40
MIN_PREVIEW_COUNT = 20
CACHE_TTL = 3600
PER = 10_000  # rates are per this many comments

# Categorical slots 1-5 of the dataviz reference palette, light surface;
# validated as a set (adjacent CVD ΔE ≥ 11.8, normal-vision ΔE ≥ 20.5).
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac"]


def parse_terms(raw: str) -> list[str]:
    """Comma-separated terms, trimmed, deduplicated case-insensitively (the
    index is case-insensitive), at most MAX_TERMS of at most MAX_TERM_LEN."""
    terms: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        term = " ".join(part.split())[:MAX_TERM_LEN]
        if term and term.lower() not in seen:
            seen.add(term.lower())
            terms.append(term)
    return terms[:MAX_TERMS]


def _keyword(term: str) -> dict:
    # Imported here: agents pulls in the MCP server and the RAG graph.
    from hn_search.api import agents

    return agents.keyword(term, k=1)


def compute(terms: list[str]) -> dict[str, Any]:
    """{terms: [{term, count, first}], months: [{month, rows, counts}]}.
    `counts` is parallel to `terms`; `first` is the earliest matching comment
    (or None)."""
    with ThreadPoolExecutor(max_workers=len(terms) or 1) as pool:
        results = list(pool.map(_keyword, terms))

    months: dict[str, dict] = {}
    for i, res in enumerate(results):
        for m in res.get("months", []):
            row = months.setdefault(
                m["month"],
                {"month": m["month"], "rows": m["rows"], "counts": [0] * len(terms)},
            )
            row["counts"][i] = m["count"]
    return {
        "terms": [
            {
                "term": term,
                "count": res.get("count", 0),
                "first": (res.get("first") or [None])[0],
            }
            for term, res in zip(terms, results)
        ],
        "months": [months[k] for k in sorted(months)],
    }


def get(terms: list[str]) -> dict[str, Any]:
    norm = json.dumps([t.lower() for t in terms])
    key = "trends:" + hashlib.md5(norm.encode()).hexdigest()
    if redis_client is not None:
        try:
            cached = cast(bytes | None, redis_client.get(key))
            if cached:
                return json.loads(cached)
        except Exception:
            pass
    data = compute(terms)
    if redis_client is not None:
        try:
            redis_client.setex(key, CACHE_TTL, json.dumps(data))
        except Exception:
            pass
    return data


def previewable(data: dict[str, Any]) -> list[int]:
    """Indexes of the terms common enough to name in a title or image."""
    return [i for i, t in enumerate(data["terms"]) if t["count"] >= MIN_PREVIEW_COUNT]


def title(data: dict[str, Any]) -> str:
    names = [data["terms"][i]["term"] for i in previewable(data)]
    return " vs ".join(names) + " on Hacker News" if names else "Hacker News trends"


def description(data: dict[str, Any]) -> str:
    parts = [
        f"{data['terms'][i]['term']}: {data['terms'][i]['count']:,} comments"
        for i in previewable(data)
    ]
    span = ""
    if data["months"]:
        span = f" ({data['months'][0]['month']} to {data['months'][-1]['month']})"
    lead = "Mentions per month in Hacker News comments" + span
    return lead + (". " + "; ".join(parts) + "." if parts else ".")


# ---- Preview image ---------------------------------------------------------


def nice_step(top: float, ticks: int = 4) -> float:
    """A round gridline step (1, 2 or 5 × 10^n) giving about `ticks` lines up
    to `top`."""
    if top <= 0:
        return 1
    raw = top / ticks
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def chart_image(data: dict[str, Any]) -> bytes:
    """1200x630 PNG: the brand row, a line per term (per 10k comments), and a
    legend naming each line. Lines are 2px-equivalent at this scale."""
    from PIL import Image, ImageDraw

    from hn_search.permalinks import ACCENT, BG, MUTED, TEXT, W, H, _font

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 12], fill=ACCENT)
    x0 = 72
    d.rectangle([x0, 48, x0 + 48, 96], fill=ACCENT)
    d.text((x0 + 24, 72), "Y", font=_font(34), fill="white", anchor="mm")
    d.text((x0 + 66, 72), "HN Search · Trends", font=_font(30), fill=TEXT, anchor="lm")

    shown = previewable(data)
    months = data["months"]
    # The current month is partial; its rate is still fair, its count is not.
    rates = [
        [m["counts"][i] * PER / m["rows"] if m["rows"] else 0 for m in months]
        for i in shown
    ]

    # Legend: swatch + term, one row, under the brand.
    lx, ly = x0, 128
    for i in shown:
        colour = SERIES[i]  # colour follows the term, as on the page
        d.rounded_rectangle([lx, ly + 6, lx + 28, ly + 14], 4, fill=colour)
        label = data["terms"][i]["term"]
        d.text((lx + 38, ly + 10), label, font=_font(28), fill=TEXT, anchor="lm")
        lx += 38 + int(d.textlength(label, font=_font(28))) + 36

    # Plot area.
    px0, px1, py0, py1 = x0 + 64, W - 72, 196, H - 70
    step = nice_step(max((max(r) for r in rates if r), default=0))
    ticks = 1
    while step * ticks < max((max(r) for r in rates if r), default=0):
        ticks += 1
    top = step * ticks
    grid = _font(20)
    for k in range(ticks + 1):
        y = py1 - (py1 - py0) * k / ticks
        d.line([(px0, y), (px1, y)], fill=(228, 225, 220), width=1)
        d.text((px0 - 10, y), f"{step * k:g}", font=grid, fill=MUTED, anchor="rm")
    d.text(
        (px1, 156),
        f"mentions per {PER:,} comments, by month",
        font=grid,
        fill=MUTED,
        anchor="rm",
    )

    n = len(months)

    def xy(j: int, v: float) -> tuple[float, float]:
        x = px0 + (px1 - px0) * (j / (n - 1) if n > 1 else 0.5)
        return x, py1 - (py1 - py0) * v / top

    for j, m in enumerate(months):  # a year tick under each January
        if m["month"].endswith("-01"):
            x, _ = xy(j, 0)
            d.line([(x, py1), (x, py1 + 6)], fill=MUTED, width=1)
            d.text((x, py1 + 10), m["month"][:4], font=grid, fill=MUTED, anchor="mt")

    for i, series in zip(shown, rates):
        points = [xy(j, v) for j, v in enumerate(series)]
        if len(points) > 1:
            d.line(points, fill=SERIES[i], width=4, joint="curve")

    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()
