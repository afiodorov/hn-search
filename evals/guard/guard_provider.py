"""promptfoo's view of the scope filter: a query in, ALLOW or REFUSE out.

promptfoo is a Node tool, so this file is the whole bridge to the Python under
test. It deliberately calls `guard.classify()` — the real thing the graph wires
in — rather than reimplementing the prompt, so an eval run exercises the same
classifier a deployed query meets. Nothing here is importable by the app; the
dependency points one way.
"""

from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

from hn_search.rag import guard

load_dotenv()  # the classifier needs DEEPSEEK_API_KEY; promptfoo does not load .env

# The app logs to stdout, which is the pipe promptfoo's Python worker talks
# JSON over; one httpx log line there kills the worker (EPIPE). Everything
# goes to stderr here instead.
for _handler in logging.getLogger().handlers:
    if isinstance(_handler, logging.StreamHandler):
        _handler.setStream(sys.stderr)
logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per call is noise


def call_api(prompt: str, options: dict, context: dict) -> dict:
    """promptfoo's provider contract. `prompt` is the rendered `{{query}}`."""
    variables = (context or {}).get("vars", {})
    query = variables.get("query", prompt)
    try:
        verdict = guard.classify(query)
    except Exception as exc:  # noqa: BLE001 — the only channel back to Node
        return {"error": f"{type(exc).__name__}: {exc}"}
    if verdict is None:
        # The guard fails open in production, but for the eval an unreachable
        # classifier is an error, not a passing ALLOW: a broken bridge (missing
        # key, unimportable package) must not score as a working filter.
        return {"error": "no verdict: the classifier errored or was unparseable"}
    return {"output": verdict}
