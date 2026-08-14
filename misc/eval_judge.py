#!/usr/bin/env python
"""LLM-as-judge regression check: replay logged eval queries against the current
pipeline and flag answers that drifted in substance from the recorded baseline.

Intended to be run by hand after a meaningful pipeline change (e.g. each stage of
the agentic RAG migration), not on every commit. Wording/phrasing/ordering
differences are expected and fine — only material changes (different
recommendations, missing key points, different/contradictory citations,
hallucinations) get flagged.

Note: `search_stream` uses the existing Redis caches (vector-search and answer),
so a query that hits a still-valid cache entry will just replay the same answer.
In practice a "big change" worth running this after also changes what those cache
keys look like, so this is usually not an issue — but for a fully clean run,
flush the relevant Redis first.

Usage:
    uv run python misc/eval_judge.py
    uv run python misc/eval_judge.py --limit 1           # cheap smoke test
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from hn_search.rag.nodes import make_llm
from hn_search.rag.pipeline import search_stream

JUDGE_PROMPT = """You are grading whether a new answer to a question is still a \
good answer, for regression-testing a Hacker News search/RAG system that is \
actively being changed (new retrieval tools, rewritten/combined searches, etc.).
The system is EXPECTED to retrieve a different set of HN comments over time as it
improves — a different specific book/blog/resource being cited than before is
normal and NOT a regression on its own.

Question: {query}

Answer A (baseline, an earlier accepted-good answer):
{baseline_answer}

Answer B (new):
{new_answer}

Wording, phrasing, ordering, formatting, and WHICH specific items/sources are \
cited are all expected to vary and are FINE — do not flag those, even if several \
recommendations differ. Flag ONLY if Answer B has an actual quality problem on \
its own merits: it doesn't address the question, contradicts itself, shows signs \
of hallucination (a claim/quote not plausibly grounded in real HN comments), is \
clearly lower quality or less useful than A (e.g. much thinner, generic, or \
off-topic), or is otherwise something you'd flag as broken if you saw it in \
isolation without needing A at all.

Respond with strict JSON and nothing else:
{{"verdict": "PASS" or "FLAG", "reasoning": "<one or two sentences>"}}"""


def run_query(query: str) -> tuple[str, list[str]]:
    """Drain search_stream for a query, returning (answer, source_ids)."""
    answer = ""
    source_ids: list[str] = []
    for event in search_stream(query):
        if event["type"] == "sources":
            source_ids = [s["id"] for s in event["sources"]]
        elif event["type"] == "answer":
            answer = event["text"]
        elif event["type"] == "error":
            raise RuntimeError(event["message"])
    return answer, source_ids


def judge(llm, query: str, baseline_answer: str, new_answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        query=query, baseline_answer=baseline_answer, new_answer=new_answer
    )
    text = llm.invoke(prompt).content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "verdict": "FLAG",
            "reasoning": f"unparseable judge response: {text[:200]}",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", default="evals/production_queries.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    records = []
    with open(args.eval_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.limit:
        records = records[: args.limit]

    llm = make_llm()
    results = []

    for r in records:
        query = r["query"]
        baseline_answer = r["answer"]
        baseline_sources = set(r.get("source_ids", []))

        try:
            new_answer, new_source_ids = run_query(query)
        except Exception as e:
            print(f"[ERROR] {query[:60]!r}: {e}")
            results.append({"query": query, "verdict": "ERROR", "reasoning": str(e)})
            continue

        verdict = judge(llm, query, baseline_answer, new_answer)
        overlap = len(baseline_sources & set(new_source_ids)) / max(
            len(baseline_sources), 1
        )

        results.append(
            {
                "query": query,
                "verdict": verdict["verdict"],
                "reasoning": verdict["reasoning"],
                "source_overlap": round(overlap, 2),
            }
        )
        print(f"[{verdict['verdict']}] {query[:60]!r} (source overlap {overlap:.0%})")
        if verdict["verdict"] != "PASS":
            print(f"    {verdict['reasoning']}")

    flagged = [r for r in results if r["verdict"] != "PASS"]
    print(
        f"\n{len(results) - len(flagged)}/{len(results)} PASS, {len(flagged)} flagged"
    )

    report_dir = Path("evals/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"report_{ts}.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote report to {report_path}")

    sys.exit(1 if flagged else 0)


if __name__ == "__main__":
    main()
