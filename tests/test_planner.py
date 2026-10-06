"""The planner loop: seed search, rounds of tool calls, the planner's pick of
sources, and the guarantees around it (a round cap, a fallback when it names
nothing usable, tool errors that don't end the run).

The models and the search backend are stubbed: the planner is a script of
replies, and each tool returns rows keyed by what it was asked. What is under
test is the graph, not the prompts.

Run with `uv run --group test pytest`.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from hn_search.rag import agent, guard, pipeline


def row(hn_id, ts="2025-01-02T03:04:05", text="text"):
    return {
        "id": hn_id,
        "author": f"user{hn_id}",
        "type": "comment",
        "text": text,
        "timestamp": ts,
        "distance": 0.3,
    }


def calls(*specs):
    """AIMessage asking for tools: specs are (name, args) pairs."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": n, "args": a, "id": f"c{i}", "type": "tool_call"}
            for i, (n, a) in enumerate(specs)
        ],
    )


class Planner:
    """Scripted planner model. Records the messages and binding of each turn."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.turns: list[list] = []
        self.tool_choices: list = []
        self._choice = None

    def bind_tools(self, _tools, tool_choice=None):
        self._choice = tool_choice
        return self

    def invoke(self, messages):
        self.turns.append(list(messages))
        self.tool_choices.append(self._choice)
        return self.replies.pop(0)


class Writer:
    def __init__(self):
        self.prompts: list[str] = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content="An answer.")


class Rig:
    def __init__(self, monkeypatch, planner_replies, results=None, failing=()):
        """`results` maps a tool call's key (the query, the hn_id, or the
        joined ids) to the rows it returns; `failing` names keys that raise."""
        self.planner = Planner(planner_replies)
        self.writer = Writer()
        self.results = results or {}
        self.tool_log: list[tuple[str, dict]] = []
        monkeypatch.setattr(guard, "is_in_scope", lambda q: True)
        monkeypatch.setattr(
            agent,
            "make_llm",
            lambda temperature=0.7, thinking=True: self.planner if temperature == 0 else self.writer,
        )
        monkeypatch.setattr(agent, "archive_start", lambda: "2023-01-01")
        monkeypatch.setattr(agent, "_fetch_parent_texts", lambda sources: {})
        monkeypatch.setattr(agent, "get_cached_answer", lambda q, c: None)
        monkeypatch.setattr(agent, "cache_answer", lambda q, c, a: None)

        def fake_tool(name):
            def invoke(args):
                self.tool_log.append((name, args))
                key = args.get("query") or args.get("hn_id") or ",".join(
                    args.get("hn_ids", [])
                )
                if key in failing:
                    raise RuntimeError(f"backend said no to {key}")
                return self.results.get(key, [])

            return type("T", (), {"name": name, "invoke": staticmethod(invoke)})()

        monkeypatch.setattr(
            agent,
            "_TOOLS_BY_NAME",
            {n: fake_tool(n) for n in agent._TOOLS_BY_NAME},
        )

    def run(self, query):
        self.events = list(pipeline.search_stream(query))
        assert not [e for e in self.events if e["type"] == "error"], self.events
        return self

    @property
    def source_ids(self):
        [event] = [e for e in self.events if e["type"] == "sources"]
        return [s["id"] for s in event["sources"]]

    @property
    def steps(self):
        return [
            (e["step"], e["status"], e["label"])
            for e in self.events
            if e["type"] == "progress"
        ]


def test_the_seed_search_is_the_planners_first_observation(monkeypatch):
    rig = Rig(
        monkeypatch,
        [AIMessage(content='{"sources": ["2", "1"], "notes": ""}')],
        results={"zig?": [row("1"), row("2")]},
    ).run("zig?")

    assert rig.tool_log == [("semantic_search", {"query": "zig?", "k": 10})]
    first_turn = rig.planner.turns[0]
    assert isinstance(first_turn[1], HumanMessage) and first_turn[1].content == "zig?"
    assert first_turn[2].tool_calls[0]["args"]["query"] == "zig?"
    assert isinstance(first_turn[3], ToolMessage)
    assert "[1] user1 · 2025-01-02" in first_turn[3].content
    assert "2023-01-01" in first_turn[0].content, "the archive start is in the prompt"
    # The planner's order, not the search's.
    assert rig.source_ids == ["2", "1"]


def test_a_dated_search_round_feeds_back_and_the_pick_spans_both(monkeypatch):
    early = {"query": "ChatGPT", "time_after": "2023-01-01", "time_before": "2023-01-07"}
    rig = Rig(
        monkeypatch,
        [
            calls(("semantic_search", early)),
            AIMessage(
                content='```json\n{"sources": ["10", "11", "999"], '
                '"notes": "Searched the first week; the archive starts then."}\n```'
            ),
        ],
        results={
            "first mention of chatgpt": [row("1")],
            "ChatGPT": [row("10", "2023-01-01T00:14:00"), row("11", "2023-01-01T02:00:00")],
        },
    ).run("first mention of chatgpt")

    assert rig.tool_log[1] == ("semantic_search", early)
    second_turn = rig.planner.turns[1]
    assert isinstance(second_turn[-1], ToolMessage) and "[10]" in second_turn[-1].content
    # 999 was never found, so it can't be cited.
    assert rig.source_ids == ["10", "11"]
    [prompt] = rig.writer.prompts
    assert "Searched the first week; the archive starts then." in prompt
    assert "from 2023-01-01 onward" in prompt
    labels = [label for step, status, label in rig.steps if step == "tools"]
    assert labels == ["Searching “ChatGPT” · 2023-01-01 → 2023-01-07"] * 2


def test_a_reply_without_usable_ids_falls_back_to_everything_found(monkeypatch):
    rig = Rig(
        monkeypatch,
        [calls(("semantic_search", {"query": "b"})), AIMessage(content="Looks good.")],
        results={"a": [row("1"), row("2")], "b": [row("2"), row("3")]},
    ).run("a")

    # Discovery order, seed first, deduplicated.
    assert rig.source_ids == ["1", "2", "3"]


def test_the_planner_must_finish_after_the_round_cap(monkeypatch):
    forever = [calls(("semantic_search", {"query": f"q{i}"})) for i in range(10)]
    rig = Rig(
        monkeypatch,
        forever[: agent._MAX_ROUNDS] + [AIMessage(content='{"sources": ["7"]}')],
        results={"q2": [row("7")]},
    ).run("start")

    assert len(rig.planner.turns) == agent._MAX_ROUNDS + 1
    assert rig.planner.tool_choices[-1] == "none"
    assert rig.planner.tool_choices[:-1] == [None] * agent._MAX_ROUNDS
    assert rig.planner.turns[-1][-1].content == agent._FINISH_NOW
    assert rig.source_ids == ["7"]


def test_a_failing_tool_is_reported_to_the_planner_not_raised(monkeypatch):
    rig = Rig(
        monkeypatch,
        [
            calls(("similar_comments", {"hn_id": "404"})),
            AIMessage(content='{"sources": ["1"]}'),
        ],
        results={"comments like 404": [row("1")]},
        failing={"404"},
    ).run("comments like 404")

    assert "The call failed: backend said no to 404" in rig.planner.turns[1][-1].content
    assert rig.source_ids == ["1"]


def test_several_calls_in_a_round_all_run(monkeypatch):
    rig = Rig(
        monkeypatch,
        [
            calls(
                ("semantic_search", {"query": "rust"}),
                ("semantic_search", {"query": "go"}),
                ("get_comments", {"hn_ids": ["5"]}),
            ),
            AIMessage(content='{"sources": ["5", "3", "4"]}'),
        ],
        results={"rust": [row("3")], "go": [row("4")], "5": [row("5")]},
    ).run("rust vs go")

    assert [n for n, _ in rig.tool_log] == [
        "semantic_search",
        "semantic_search",
        "semantic_search",
        "get_comments",
    ]
    assert len([m for m in rig.planner.turns[1] if isinstance(m, ToolMessage)]) == 4
    assert rig.source_ids == ["5", "3", "4"]
    [label] = {label for step, _, label in rig.steps if step == "tools"}
    assert label == "Searching “rust”; “go”; reading 1 comments"


def test_sources_are_capped(monkeypatch):
    many = [row(str(100 + i)) for i in range(30)]
    rig = Rig(
        monkeypatch,
        [AIMessage(content="none of these")],
        results={"x": many},
    ).run("x")

    assert len(rig.source_ids) == agent._MAX_SOURCES


@pytest.mark.parametrize(
    "content, ids, notes",
    [
        ('{"sources": ["1", "2"], "notes": "n"}', ["1", "2"], "n"),
        ('```json\n{"sources": [123456]}\n```', ["123456"], ""),
        ("I'd use 4567890 and 4567891.", ["4567890", "4567891"], ""),
        ('{"sources": "oops"', [], ""),
        ("", [], ""),
    ],
)
def test_parse_selection(content, ids, notes):
    assert agent._parse_selection(content) == (ids, notes)
