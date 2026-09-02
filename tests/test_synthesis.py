"""Template determinism and LLM synthesis against a mocked transport."""

import json
import logging

import httpx
import pytest

from keyword_extractor.config import Settings
from keyword_extractor.synthesis import (
    COMPLEXITIES,
    QueryContext,
    TemplateSynthesizer,
    build_synthesizer,
)
from keyword_extractor.synthesis.llm import LLMSynthesizer
from keyword_extractor.synthesis.template import select_keywords

# The endpoint's verbatim refusal of temperature=0, which turned a 100-row LLM
# run into 100 silent template fallbacks.
_TEMPERATURE_400 = {
    "error": {
        "message": "Unsupported value: 'temperature' does not support 0 with this model. "
        "Only the default (1) value is supported.",
        "type": "invalid_request_error",
        "param": "temperature",
        "code": "unsupported_value",
    }
}


@pytest.fixture(autouse=True)
def _clear_unsupported_cache():
    """The learned-field cache is module state; keep tests independent."""
    from keyword_extractor.synthesis import llm as llm_module

    llm_module._UNSUPPORTED_FIELDS.clear()
    yield
    llm_module._UNSUPPORTED_FIELDS.clear()


KEYWORDS = ["auction theory", "mechanism design", "market equilibria"]


def _text(keywords, context) -> str:
    return TemplateSynthesizer().synthesize(keywords, context).text


def test_template_deterministic_per_seed():
    context = QueryContext(seed="uuid-1")
    assert _text(KEYWORDS, context) == _text(KEYWORDS, context)


def test_template_reports_its_own_style():
    assert TemplateSynthesizer().synthesize(KEYWORDS, QueryContext(seed="x")).style == "template"


def test_template_varies_across_seeds():
    outputs = {_text(KEYWORDS, QueryContext(seed=f"uuid-{i}")) for i in range(30)}
    assert len(outputs) > 1


def test_template_contains_all_keywords():
    query = _text(KEYWORDS, QueryContext(seed="x"))
    for keyword in KEYWORDS:
        assert keyword in query


def test_template_handles_empty_keywords():
    assert "open research topic" in _text([], QueryContext(seed="x"))


GERMAN_KEYWORDS = ["Kommunikationsprozesse", "lokale Politik", "Medienwandel"]


def test_german_context_selects_german_template():
    query = _text(GERMAN_KEYWORDS, QueryContext(seed="uuid-de", language="deu"))
    assert "Abschlussarbeit" in query or "Betreuung" in query
    assert " und " in query  # German conjunction, not "and"
    for keyword in GERMAN_KEYWORDS:
        assert keyword in query


def test_german_empty_keywords_stays_german():
    assert "Forschungsthema" in _text([], QueryContext(seed="x", language="deu"))


def test_non_german_languages_fall_back_to_english():
    context_none = QueryContext(seed="uuid-1", language=None)
    for language in (None, "eng", "fra", "ita"):
        query = _text(KEYWORDS, QueryContext(seed="uuid-1", language=language))
        assert query == _text(KEYWORDS, context_none)
        assert "Abschlussarbeit" not in query


def test_llm_prompt_requests_german_when_language_is_german():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content.decode("utf-8")
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "Eine Arbeit über Medienwandel."}}]}
        )

    synth = LLMSynthesizer("http://test/v1", "m", client=_client(handler))
    synth.synthesize(GERMAN_KEYWORDS, QueryContext(seed="x", language="deu"))
    assert "Write the query in German." in seen["body"]


def test_llm_prompt_omits_language_instruction_for_english():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content.decode("utf-8")
        return httpx.Response(200, json={"choices": [{"message": {"content": "A thesis."}}]})

    synth = LLMSynthesizer("http://test/v1", "m", client=_client(handler))
    synth.synthesize(KEYWORDS, QueryContext(seed="x", language="eng"))
    assert "German" not in seen["body"]


MANY_KEYWORDS = [f"topic{i}" for i in range(10)]


def test_complexity_controls_keyword_count():
    assert select_keywords(MANY_KEYWORDS, "low") == ["topic0", "topic1"]
    assert select_keywords(MANY_KEYWORDS, "medium") == MANY_KEYWORDS[:4]
    assert select_keywords(MANY_KEYWORDS, "high") == MANY_KEYWORDS[:8]


def test_low_complexity_query_lists_only_two_keywords():
    query = _text(MANY_KEYWORDS, QueryContext(seed="x", complexity="low"))
    assert "topic0" in query and "topic1" in query
    assert "topic2" not in query


def test_levels_differ_for_the_same_row():
    contexts = [QueryContext(seed="x", complexity=c) for c in COMPLEXITIES]
    texts = [_text(MANY_KEYWORDS, c) for c in contexts]
    assert len(set(texts)) == 3
    # High is the elaborate one, low the terse one.
    assert len(texts[0]) < len(texts[2])


def test_high_complexity_names_the_department_when_known():
    context = QueryContext(seed="x", complexity="high", department="Department of Economics")
    assert "Department of Economics" in _text(MANY_KEYWORDS, context)


def test_high_complexity_without_department_has_no_dangling_text():
    query = _text(MANY_KEYWORDS, QueryContext(seed="x", complexity="high"))
    assert "  " not in query  # no gap where the department would have gone
    assert query.strip().endswith((".", "?"))


def test_german_covers_every_complexity_level():
    for complexity in COMPLEXITIES:
        query = _text(
            GERMAN_KEYWORDS, QueryContext(seed="x", language="deu", complexity=complexity)
        )
        assert "thesis" not in query.lower()
        # German carrier text, whatever the sentence shape ("Ich suche ...",
        # "In meiner Masterarbeit möchte ich ...").
        assert any(word in query.lower() for word in ("ich", "meine", "arbeit"))


def test_llm_prompt_carries_a_per_level_instruction():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content.decode("utf-8")
        return httpx.Response(200, json={"choices": [{"message": {"content": "A thesis."}}]})

    synth = LLMSynthesizer("http://test/v1", "m", client=_client(handler))
    synth.synthesize(MANY_KEYWORDS, QueryContext(seed="x", complexity="low"))
    assert "one short sentence" in seen["body"]
    # `low` must not leak the keywords it is supposed to leave out.
    assert "topic5" not in seen["body"]


def _temperature_rejecting_handler(sent: list[dict]):
    """Mimics gpt-5-mini: 400 while temperature is present, 200 once it is gone."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        if "temperature" in body:
            return httpx.Response(400, json=_TEMPERATURE_400)
        return httpx.Response(200, json={"choices": [{"message": {"content": "A real query."}}]})

    return handler


def test_unsupported_temperature_is_dropped_and_the_call_retried():
    sent: list[dict] = []
    handler = _temperature_rejecting_handler(sent)
    synth = LLMSynthesizer("http://test/v1", "m", client=_client(handler))

    result = synth.synthesize(KEYWORDS, QueryContext(seed="x"))

    assert result.text == "A real query."
    assert result.style == "llm"  # succeeded, so not a fallback
    assert len(sent) == 2
    assert "temperature" in sent[0] and "temperature" not in sent[1]


def test_the_doomed_request_is_paid_once_per_process_not_once_per_row():
    """This is what turned one bad field into 100 failed calls."""
    sent: list[dict] = []
    handler = _temperature_rejecting_handler(sent)
    synth = LLMSynthesizer("http://test/v1", "m", client=_client(handler))

    synth.synthesize(KEYWORDS, QueryContext(seed="a"))  # learns: 2 requests
    sent.clear()
    # A *fresh* instance against the same endpoint and model must not relearn.
    other = LLMSynthesizer("http://test/v1", "m", client=_client(handler))
    result = other.synthesize(KEYWORDS, QueryContext(seed="b"))

    assert result.style == "llm"
    assert len(sent) == 1
    assert "temperature" not in sent[0]


def test_only_the_blamed_field_is_dropped():
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        if "reasoning_effort" in body:
            return httpx.Response(
                400,
                json={"error": {"message": "Unknown parameter", "param": "reasoning_effort"}},
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": "Query."}}]})

    synth = LLMSynthesizer("http://test/v1", "m", reasoning_effort="low", client=_client(handler))
    assert synth.synthesize(KEYWORDS, QueryContext(seed="x")).style == "llm"
    # reasoning_effort was blamed, so temperature must survive the retry.
    assert "temperature" in sent[1]
    assert "reasoning_effort" not in sent[1]


def test_reasoning_effort_is_sent_only_when_configured():
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "Query."}}]})

    LLMSynthesizer("http://test/v1", "m", client=_client(handler)).synthesize(
        KEYWORDS, QueryContext(seed="x")
    )
    assert "reasoning_effort" not in sent[0]

    LLMSynthesizer(
        "http://test/v1", "m2", reasoning_effort="low", client=_client(handler)
    ).synthesize(KEYWORDS, QueryContext(seed="x"))
    assert sent[1]["reasoning_effort"] == "low"


def test_endpoint_error_message_reaches_the_log(caplog):
    """httpx's bare "Client error '400 Bad Request'" is what cost the diagnosis."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"error": {"message": "Model 'llama3.1' does not exist.", "param": None}}
        )

    synth = LLMSynthesizer("http://test/v1", "m", client=_client(handler))
    with caplog.at_level(logging.WARNING):
        result = synth.synthesize(KEYWORDS, QueryContext(seed="x"))

    assert result.style == "template"
    assert "Model 'llama3.1' does not exist." in caplog.text


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

    result = LLMSynthesizer("http://test/v1", "m", client=_client(handler)).synthesize(
        KEYWORDS, QueryContext(seed="x")
    )
    assert result.text == "A thesis on auction design."
    assert result.style == "llm"


def test_llm_synthesizer_falls_back_to_template_on_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    synth = LLMSynthesizer("http://test/v1", "m", client=_client(handler))
    context = QueryContext(seed="x")
    result = synth.synthesize(KEYWORDS, context)
    assert result.text == _text(KEYWORDS, context)
    # The row must admit it is template text, even though the run asked for LLM.
    assert result.style == "template"


def test_llm_synthesizer_falls_back_on_empty_content():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "  "}}]})

    synth = LLMSynthesizer("http://test/v1", "m", client=_client(handler))
    context = QueryContext(seed="x")
    result = synth.synthesize(KEYWORDS, context)
    assert result.text == _text(KEYWORDS, context)
    assert result.style == "template"


# --- transient failures ------------------------------------------------------
# Distinct from the 400 above: those mean the payload is wrong and retrying it
# unchanged fails identically. These mean "not now", and a run of thousands of
# sequential calls will meet them.

_OK = {"choices": [{"message": {"content": "A real query."}}]}


def _scripted(responses: list[httpx.Response], calls: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return responses.pop(0) if responses else httpx.Response(200, json=_OK)

    return handler


def test_rate_limit_is_retried_and_the_row_keeps_its_llm_style():
    calls: list[httpx.Request] = []
    handler = _scripted(
        [httpx.Response(429, json={"error": {"message": "Rate limit reached"}})], calls
    )
    synth = LLMSynthesizer(
        "http://test/v1", "m", max_attempts=3, retry_backoff=0.0, client=_client(handler)
    )

    result = synth.synthesize(KEYWORDS, QueryContext(seed="x"))

    assert result.style == "llm"  # a blip must not mislabel the row
    assert result.text == "A real query."
    assert len(calls) == 2


def test_network_errors_are_retried_too():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ConnectError("connection refused")

    synth = LLMSynthesizer(
        "http://test/v1", "m", max_attempts=2, retry_backoff=0.0, client=_client(handler)
    )

    assert synth.synthesize(KEYWORDS, QueryContext(seed="x")).style == "template"
    assert len(calls) == 2


def test_exhausted_attempts_fall_back_and_say_why(caplog):
    calls: list[httpx.Request] = []
    handler = _scripted(
        [httpx.Response(503, json={"error": {"message": "upstream unavailable"}})] * 3, calls
    )
    synth = LLMSynthesizer(
        "http://test/v1", "m", max_attempts=3, retry_backoff=0.0, client=_client(handler)
    )

    with caplog.at_level(logging.WARNING):
        result = synth.synthesize(KEYWORDS, QueryContext(seed="x"))

    assert result.style == "template"
    assert len(calls) == 3
    # The warning has to carry the endpoint's own words and the attempt count.
    assert "upstream unavailable" in caplog.text
    assert "after 3 attempts" in caplog.text


def test_retry_after_header_overrides_the_computed_backoff(monkeypatch):
    from keyword_extractor.synthesis import llm as llm_module

    slept: list[float] = []
    monkeypatch.setattr(llm_module.time, "sleep", slept.append)
    calls: list[httpx.Request] = []
    handler = _scripted(
        [
            httpx.Response(
                429, headers={"Retry-After": "2"}, json={"error": {"message": "slow down"}}
            )
        ],
        calls,
    )
    synth = LLMSynthesizer(
        "http://test/v1", "m", max_attempts=3, retry_backoff=30.0, client=_client(handler)
    )

    assert synth.synthesize(KEYWORDS, QueryContext(seed="x")).style == "llm"
    assert slept == [2.0]  # the endpoint's instruction, not our 30s backoff


def test_correcting_the_payload_costs_no_attempt():
    """A dropped field is a fix, not a retry -- it must not spend the budget."""
    sent: list[dict] = []
    handler = _temperature_rejecting_handler(sent)
    synth = LLMSynthesizer(
        "http://test/v1", "m", max_attempts=1, retry_backoff=0.0, client=_client(handler)
    )

    result = synth.synthesize(KEYWORDS, QueryContext(seed="x"))

    assert result.style == "llm"
    assert len(sent) == 2  # two requests despite a single attempt
