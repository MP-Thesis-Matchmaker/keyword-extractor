"""Template determinism and LLM synthesis against a mocked transport."""

import httpx

from keyword_extractor.config import Settings
from keyword_extractor.synthesis import QueryContext, TemplateSynthesizer, build_synthesizer
from keyword_extractor.synthesis.llm import LLMSynthesizer

KEYWORDS = ["auction theory", "mechanism design", "market equilibria"]


def test_template_deterministic_per_seed():
    context = QueryContext(seed="uuid-1")
    synth = TemplateSynthesizer()
    assert synth.synthesize(KEYWORDS, context) == synth.synthesize(KEYWORDS, context)


def test_template_varies_across_seeds():
    synth = TemplateSynthesizer()
    outputs = {synth.synthesize(KEYWORDS, QueryContext(seed=f"uuid-{i}")) for i in range(30)}
    assert len(outputs) > 1


def test_template_contains_all_keywords():
    query = TemplateSynthesizer().synthesize(KEYWORDS, QueryContext(seed="x"))
    for keyword in KEYWORDS:
        assert keyword in query


def test_template_handles_empty_keywords():
    query = TemplateSynthesizer().synthesize([], QueryContext(seed="x"))
    assert "open research topic" in query


def test_build_synthesizer_defaults_to_template():
    synth = build_synthesizer(Settings(llm_base_url=None))
    assert isinstance(synth, TemplateSynthesizer)


def test_build_synthesizer_llm_when_base_url_set():
    synth = build_synthesizer(Settings(llm_base_url="http://localhost:11434/v1"))
    assert isinstance(synth, LLMSynthesizer)


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_llm_synthesizer_returns_endpoint_text():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "A thesis on auction design."}}]},
        )

    synth = LLMSynthesizer("http://test/v1", "m", client=_client(handler))
    assert synth.synthesize(KEYWORDS, QueryContext(seed="x")) == "A thesis on auction design."


def test_llm_synthesizer_falls_back_to_template_on_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    synth = LLMSynthesizer("http://test/v1", "m", client=_client(handler))
    context = QueryContext(seed="x")
    assert synth.synthesize(KEYWORDS, context) == TemplateSynthesizer().synthesize(
        KEYWORDS, context
    )


def test_llm_synthesizer_falls_back_on_empty_content():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "  "}}]})

    synth = LLMSynthesizer("http://test/v1", "m", client=_client(handler))
    context = QueryContext(seed="x")
    assert synth.synthesize(KEYWORDS, context) == TemplateSynthesizer().synthesize(
        KEYWORDS, context
    )
