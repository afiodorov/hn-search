"""Non-English queries are embedded in English.

The translator model, the encoder and the search backend are all stubbed; what
is under test is the wiring: plain ASCII never costs a model call, anything
else is embedded as its translation (replacing the original, not searched
beside it), the cache stays keyed on what the user typed, and a failure falls
back to the original.

Run with `uv run --group test pytest`.
"""

import pytest

from hn_search.rag import tools, translate

ROW = ("43000001", "PNG is fine.", "alice", "2025-01-02T03:04:05", "comment", 0.31)


class _Reply:
    def __init__(self, content):
        self.content = content


class _StubModel:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return _Reply(self.reply)


class _RecordingEncoder:
    def __init__(self):
        self.texts: list[str] = []

    def encode(self, texts):
        self.texts.extend(texts)
        return [[0.0] * 768 for _ in texts]


@pytest.fixture
def translator(monkeypatch):
    model = _StubModel("lossless image compression technology")
    monkeypatch.setattr(translate, "_model", lambda: model)
    return model


@pytest.fixture
def search_rig(monkeypatch):
    encoder = _RecordingEncoder()
    cached: list[str] = []
    searches: list[int] = []

    def fake_search(embedding, n_results, time_after=None, time_before=None):
        searches.append(n_results)
        return [ROW]

    monkeypatch.setattr(tools, "get_model", lambda: encoder)
    monkeypatch.setattr(tools, "search", fake_search)
    monkeypatch.setattr(tools, "get_cached_vector_search", lambda *a: None)
    monkeypatch.setattr(
        tools, "cache_vector_search", lambda query, *a: cached.append(query)
    )
    return encoder, cached, searches


def test_plain_ascii_never_calls_the_model(translator):
    assert translate.to_english("rust vs go") == "rust vs go"
    assert translator.prompts == []


def test_non_ascii_is_translated(translator):
    assert translate.to_english("图片无损压缩技术") == (
        "lossless image compression technology"
    )
    [messages] = translator.prompts
    assert "<query>\n图片无损压缩技术\n</query>" in messages[-1].content


def test_accented_latin_script_is_translated_too(translator):
    translate.to_english("compression d'image sans perte, ça marche ?")
    assert len(translator.prompts) == 1


def test_a_translator_error_is_reported_as_none(monkeypatch):
    monkeypatch.setattr(translate, "_model", lambda: _StubModel(RuntimeError("down")))
    assert translate.to_english("图片无损压缩技术") is None


def test_an_empty_reply_is_reported_as_none(monkeypatch):
    monkeypatch.setattr(translate, "_model", lambda: _StubModel("  "))
    assert translate.to_english("图片无损压缩技术") is None


def test_search_embeds_the_translation_instead_of_the_original(translator, search_rig):
    encoder, cached, searches = search_rig

    tools.semantic_search.invoke({"query": "图片无损压缩技术"})

    # One search, over the English text only: the original is replaced, not
    # searched alongside it.
    assert encoder.texts == ["lossless image compression technology"]
    assert searches == [10]
    # The cache stays keyed on what the user typed, so a repeat skips both the
    # translation and the search.
    assert cached == ["图片无损压缩技术"]


def test_a_cache_hit_skips_the_translation(monkeypatch, translator, search_rig):
    encoder, _, _ = search_rig
    monkeypatch.setattr(tools, "get_cached_vector_search", lambda *a: [{"id": "1"}])

    assert tools.semantic_search.invoke({"query": "图片无损压缩技术"}) == [{"id": "1"}]
    assert translator.prompts == []
    assert encoder.texts == []


def test_a_failed_translation_searches_the_original_but_is_not_cached(
    monkeypatch, search_rig
):
    """Caching the untranslated search would serve its noise for the whole
    cache TTL, long after the translator is back."""
    encoder, cached, searches = search_rig
    monkeypatch.setattr(translate, "_model", lambda: _StubModel(RuntimeError("down")))

    hits = tools.semantic_search.invoke({"query": "图片无损压缩技术"})

    assert encoder.texts == ["图片无损压缩技术"]
    assert [h["id"] for h in hits] == ["43000001"]
    assert cached == []


def test_plain_ascii_is_still_cached(translator, search_rig):
    _, cached, _ = search_rig
    tools.semantic_search.invoke({"query": "rust vs go"})
    assert cached == ["rust vs go"]
    assert translator.prompts == []
