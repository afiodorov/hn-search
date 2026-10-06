#!/usr/bin/env python
"""Pairwise A/B eval of two pipeline versions over real production queries.

`eval_judge.py` asks "is the new answer still OK?". This asks "which of two
answers to the same question is better?", blind and in random order, which is
what you want before swapping one pipeline for another. Run each arm from the
code tree it should measure (e.g. a `git worktree` of main for the old arm),
then judge any two arm files against each other. Judging an arm against a
second run of itself gives the noise floor: a real difference has to beat it.

    uv run python misc/eval_compare.py run --out evals/runs/new.jsonl
    uv run python misc/eval_compare.py judge evals/runs/old.jsonl evals/runs/new.jsonl

`run` disables the Redis caches by pointing REDIS_URL at nothing unless you set
it yourself, so every query really goes through the pipeline. Needs the same env
as the app (DEEPSEEK_API_KEY, HN_SEARCH_URL, HN_SEARCH_TOKEN); `railway run -s
hn-search-web --` provides it.
"""

import argparse
import json
import os
import random
import re
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:1/0")

_CITATION = re.compile(
    r"\[\[(\d+)\]\]\(https://news\.ycombinator\.com/item\?id=(\d+)\)"
)


def load_queries(path: str, limit: int | None) -> list[str]:
    seen: dict[str, None] = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                seen.setdefault(json.loads(line)["query"].strip(), None)
    queries = list(seen)
    return queries[:limit] if limit else queries


def run_one(query: str) -> dict:
    from hn_search.rag.pipeline import search_stream

    record: dict = {"query": query, "steps": [], "sources": [], "answer": ""}
    t0 = time.perf_counter()
    try:
        for event in search_stream(query):
            kind = event["type"]
            if kind == "progress" and event["status"] == "done":
                record["steps"].append(event["label"])
            elif kind == "sources":
                record["sources"] = [
                    {
                        "id": s["id"],
                        "author": s["author"],
                        "timestamp": s["timestamp"],
                        "text": s["text"],
                    }
                    for s in event["sources"]
                ]
            elif kind == "answer":
                record["answer"] = event["text"]
                record["refused"] = event.get("refused", False)
            elif kind == "error":
                record["error"] = event["message"]
    except Exception as e:  # one bad query must not sink the run
        record["error"] = repr(e)
    record["ms"] = round((time.perf_counter() - t0) * 1000)
    return record


def cmd_run(args):
    queries = load_queries(args.queries, args.limit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"running {len(queries)} queries → {out}", file=sys.stderr)
    with ThreadPoolExecutor(args.workers) as pool, open(out, "w") as f:
        for i, rec in enumerate(pool.map(run_one, queries), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            flag = "ERR " if rec.get("error") else ""
            print(
                f"[{i}/{len(queries)}] {flag}{rec['ms']}ms {rec['query'][:70]!r}",
                file=sys.stderr,
            )


def citation_problems(rec: dict) -> int:
    """Citations whose number or id doesn't match the source list it was given."""
    ids = [s["id"] for s in rec["sources"]]
    bad = 0
    for num, hn_id in _CITATION.findall(rec["answer"]):
        n = int(num)
        if not (1 <= n <= len(ids)) or ids[n - 1] != hn_id:
            bad += 1
    return bad


JUDGE_PROMPT = """You are judging two answers from a search assistant over Hacker \
News comments. Each answer was written only from the HN comments listed under it, \
which are its sources. Today's date is {today}. The archive holds HN comments \
from 2023-01-01 onward.

Question: {query}

=== Answer 1 ===
{answer_1}

Sources for answer 1:
{sources_1}

=== Answer 2 ===
{answer_2}

Sources for answer 2:
{sources_2}

Judge which answer serves the person asking better. Consider, in this order:
1. Does it answer the question that was actually asked (including any time \
period, comparison, or specific comment it names)?
2. Is it grounded: are its claims and quotes supported by its own sources, with \
no invented facts?
3. Are its sources relevant to the question (and inside the time period, if one \
was asked for)?
4. Is it specific and useful rather than generic?
Length and style are not merits on their own. If both are about equally good, \
or both equally bad, say TIE.

Respond with strict JSON only:
{{"winner": "1" | "2" | "TIE", "reasoning": "<two sentences at most>"}}"""


SOURCES_PROMPT = """You are judging two sets of Hacker News comments retrieved by a \
search assistant for the same question. A writer will answer the question using \
only one of these sets. Today's date is {today}. The archive holds HN comments \
from 2023-01-01 onward.

Question: {query}

=== Set 1 ===
{sources_1}

=== Set 2 ===
{sources_2}

Which set would let a writer answer this question better? Consider relevance to \
what was actually asked (including any time period, comparison, or specific \
comment it names), coverage of the different sides or parts of the question, and \
how many comments are useful rather than off-topic. The number of comments is \
not a merit on its own. If they are about equally good, say TIE.

Respond with strict JSON only:
{{"winner": "1" | "2" | "TIE", "reasoning": "<two sentences at most>"}}"""


def _fmt_sources(rec: dict) -> str:
    if not rec["sources"]:
        return "(none)"
    return "\n".join(
        f"[{i}] {s['author']} ({s['timestamp'][:10]}): {s['text'][:350]}"
        for i, s in enumerate(rec["sources"], 1)
    )


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"winner": "TIE", "reasoning": f"unparseable judge output: {text[:200]}"}


def judge_pair(
    llm, today: str, a: dict, b: dict, seed: int, sources_only: bool = False
) -> dict:
    """Blind: the judge sees the arms as 1/2 in a seeded random order. With
    sources_only it compares what was retrieved, not the answers, which keeps
    answer length and style out of the verdict."""
    flip = random.Random(seed).random() < 0.5
    first, second = (b, a) if flip else (a, b)
    if sources_only:
        prompt = SOURCES_PROMPT.format(
            today=today,
            query=a["query"],
            sources_1=_fmt_sources(first),
            sources_2=_fmt_sources(second),
        )
    else:
        prompt = JUDGE_PROMPT.format(
            today=today,
            query=a["query"],
            answer_1=first["answer"] or "(empty)",
            sources_1=_fmt_sources(first),
            answer_2=second["answer"] or "(empty)",
            sources_2=_fmt_sources(second),
        )
    verdict = _parse_json(llm.invoke(prompt).content)
    w = str(verdict.get("winner", "TIE")).strip()
    if w == "1":
        winner = "B" if flip else "A"
    elif w == "2":
        winner = "A" if flip else "B"
    else:
        winner = "TIE"
    return {
        "winner": winner,
        "shown_first": "B" if flip else "A",
        "reasoning": verdict.get("reasoning", ""),
    }


def cmd_judge(args):
    from datetime import date

    from hn_search.rag.nodes import make_llm

    def load(p):
        with open(p) as f:
            return {r["query"]: r for r in map(json.loads, filter(str.strip, f))}

    arm_a, arm_b = load(args.a), load(args.b)
    queries = [q for q in arm_a if q in arm_b]
    llm = make_llm(temperature=0)
    today = date.today().isoformat()

    def one(q):
        a, b = arm_a[q], arm_b[q]
        if a.get("error") or b.get("error"):
            return {
                "query": q,
                "winner": "ERROR",
                "reasoning": a.get("error") or b.get("error"),
            }
        if a.get("refused") and b.get("refused"):
            return {"query": q, "winner": "TIE", "reasoning": "both refused"}
        verdict = judge_pair(
            llm,
            today,
            a,
            b,
            seed=zlib.crc32(q.encode()),
            sources_only=args.sources_only,
        )
        return {"query": q, **verdict}

    with ThreadPoolExecutor(args.workers) as pool:
        verdicts = list(pool.map(one, queries))

    def stats(arm):
        recs = [arm[q] for q in queries]
        ms = sorted(r["ms"] for r in recs)
        return {
            "errors": sum(1 for r in recs if r.get("error")),
            "refused": sum(1 for r in recs if r.get("refused")),
            "no_sources": sum(
                1 for r in recs if not r["sources"] and not r.get("refused")
            ),
            "bad_citations": sum(citation_problems(r) for r in recs),
            "p50_ms": ms[len(ms) // 2],
            "p90_ms": ms[int(len(ms) * 0.9)],
        }

    tally = {
        k: sum(1 for v in verdicts if v["winner"] == k)
        for k in ("A", "B", "TIE", "ERROR")
    }
    report = {
        "a": args.a,
        "b": args.b,
        "n": len(queries),
        "tally": tally,
        "stats_a": stats(arm_a),
        "stats_b": stats(arm_b),
        "verdicts": verdicts,
    }
    mode = "sources_" if args.sources_only else ""
    out = Path(
        args.out
        or f"evals/reports/compare_{mode}{Path(args.a).stem}_vs_{Path(args.b).stem}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"{len(queries)} queries: A={args.a} B={args.b}")
    print(
        f"  B better {tally['B']}, A better {tally['A']}, tie {tally['TIE']}, errors {tally['ERROR']}"
    )
    decided = [v for v in verdicts if v["winner"] in ("A", "B") and "shown_first" in v]
    first_won = sum(1 for v in decided if v["winner"] == v["shown_first"])
    print(f"  position check: the answer shown first won {first_won}/{len(decided)}")
    for name, s in (("A", report["stats_a"]), ("B", report["stats_b"])):
        print(f"  {name}: {s}")
    print(f"report → {out}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--queries", default="evals/production_queries.jsonl")
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int)
    r.add_argument("--workers", type=int, default=6)
    j = sub.add_parser("judge")
    j.add_argument("a")
    j.add_argument("b")
    j.add_argument("--out")
    j.add_argument("--workers", type=int, default=8)
    j.add_argument(
        "--sources-only",
        action="store_true",
        help="judge the retrieved comments, not the answers",
    )
    args = ap.parse_args()
    cmd_run(args) if args.cmd == "run" else cmd_judge(args)


if __name__ == "__main__":
    main()
