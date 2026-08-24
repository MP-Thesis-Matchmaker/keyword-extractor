"""LLM-backed query synthesis over an OpenAI-compatible chat endpoint.

The retry machinery is adapted from backend-core's
`src/thesis_matchmaker/llm.py`, which solved this exact problem first. It is
copied rather than imported because backend-core is only installed under the
`eval` extra, while synthesis has to work in the base install.

Why it is needed: every optional field below is an extension some endpoint does
not implement, and a 400 is how it says so. `temperature: 0` is rejected
outright by OpenAI's gpt-5 and o-series ("Unsupported value: 'temperature' does
not support 0 with this model"), which once turned a 100-row LLM run into 100
silent template fallbacks. The set of refused fields is remembered per
(endpoint, model) at MODULE level, so the doomed request is paid once per
process rather than once per row -- that, not the retry itself, is what makes a
large run survive an unsupported field.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import httpx

from keyword_extractor.synthesis.base import Complexity, QueryContext, SynthesizedQuery
from keyword_extractor.synthesis.template import TemplateSynthesizer, select_keywords

logger = logging.getLogger(__name__)

# Which optional request fields an endpoint refuses, keyed by (url, model).
# Module-level rather than per instance: it is a property of the endpoint, not
# of the object asking.
_UNSUPPORTED_FIELDS: dict[tuple[str, str], set[str]] = {}

_SYSTEM_PROMPT = (
    "You write realistic queries a student would send to a thesis-supervisor "
    "matching assistant. Rewrite the given keywords into a natural-language query "
    "describing the thesis topic the student is looking for. "
    "Do not mention any person, author or supervisor names. "
    "Return only the query text, no quotes and no explanations."
)

# What each rung of the ladder should sound like. These mirror the template
# sets, so an LLM run and a template run are comparable at the same level.
_COMPLEXITY_INSTRUCTIONS: dict[Complexity, str] = {
    "low": (
        "Write one short sentence naming only the broad topic, as someone would type "
        "who knows their field but has no specific project yet."
    ),
    "medium": "Write one or two sentences covering the topics given.",
    "high": (
        "Write two or three sentences, as someone with a specific project in mind: "
        "mention the subject area if one is given and ask for a supervisor whose own "
        "research combines these topics."
    ),
}


class LLMError(RuntimeError):
    """Raised when the chat endpoint is unreachable or returns something odd."""


def _error_message(response: httpx.Response) -> str | None:
    """The endpoint's own explanation, when it provides one."""
    try:
        error = response.json().get("error")
    except ValueError:
        return None
    if isinstance(error, dict):
        message = error.get("message")
        return message if isinstance(message, str) else None
    return error if isinstance(error, str) else None


def _blamed_fields(response: httpx.Response, sent: set[str]) -> set[str]:
    """Which of the fields we sent the endpoint is complaining about.

    OpenAI names it in `error.param`; others only say so in prose, so the
    message is searched too. With no clue at all every optional field is
    blamed -- a worse-shaped answer beats none.
    """
    try:
        error = response.json().get("error") or {}
    except ValueError:
        return set(sent)
    if not isinstance(error, dict):
        return set(sent)
    param = error.get("param")
    if isinstance(param, str) and param in sent:
        return {param}
    message = error.get("message")
    if isinstance(message, str):
        named = {field for field in sent if field in message}
        if named:
            return named
    return set(sent)


class LLMSynthesizer:
    """Query phrasing via an OpenAI-compatible endpoint, template on failure."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        reasoning_effort: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._timeout = timeout
        # Reasoning models can spend their whole wall-clock budget on hidden
        # reasoning before emitting a word. Left unset by default because the
        # field is meaningless to a non-reasoning model, and an endpoint that
        # does not know it may reject the request -- which the retry handles.
        self._reasoning_effort = reasoning_effort
        # Injectable for tests (httpx.MockTransport); owned here otherwise.
        self._client = client or httpx.Client(timeout=timeout)
        self._unsupported = _UNSUPPORTED_FIELDS.setdefault((self._url, model), set())
        self._fallback = TemplateSynthesizer()

    def synthesize(self, keywords: Sequence[str], context: QueryContext) -> SynthesizedQuery:
        selected = select_keywords(keywords, context.complexity)
        user = "Keywords: " + "; ".join(selected)
        if context.department:
            user += f"\nSubject area: {context.department}"
        # The keywords are in the language of the source publications, so the
        # carrier sentence has to match or the query reads as code-switching.
        if context.language == "deu":
            user += "\nWrite the query in German."
        user += "\n" + _COMPLEXITY_INSTRUCTIONS[context.complexity]

        try:
            text = self._chat(user, allow_retry=True)
        except LLMError as exc:
            logger.warning("LLM synthesis failed (%s); using template fallback", exc)
            return self._fallback.synthesize(keywords, context)
        if not text:
            logger.warning("LLM returned an empty query; using template fallback")
            return self._fallback.synthesize(keywords, context)
        return SynthesizedQuery(text=text, style="llm")

    def _chat(self, user: str, *, allow_retry: bool) -> str:
        payload: dict = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        }
        # Anything already refused once by this endpoint is left out from the start.
        sent: set[str] = set()
        # Determinism where the model supports it. gpt-5/o-series reject any
        # temperature but the default.
        if "temperature" not in self._unsupported:
            payload["temperature"] = 0
            sent.add("temperature")
        if self._reasoning_effort and "reasoning_effort" not in self._unsupported:
            payload["reasoning_effort"] = self._reasoning_effort
            sent.add("reasoning_effort")

        try:
            response = self._client.post(self._url, json=payload, headers=self._headers)
        except httpx.HTTPError as exc:
            raise LLMError(str(exc)) from exc

        # Retry once without whatever was refused, rather than failing outright.
        # Only the fields actually blamed are dropped.
        if response.status_code in (400, 422) and allow_retry and sent:
            blamed = _blamed_fields(response, sent)
            self._unsupported |= blamed
            logger.info(
                "endpoint rejected %s (%s); retrying without it and omitting it from "
                "later requests",
                ", ".join(sorted(blamed)),
                _error_message(response) or "no message",
            )
            return self._chat(user, allow_retry=False)

        if response.status_code >= 400:
            # The endpoint explains itself; httpx's "Client error '400 Bad
            # Request'" alone is what made this expensive to diagnose.
            raise LLMError(
                f"HTTP {response.status_code}: {_error_message(response) or response.text[:200]}"
            )
        try:
            return str(response.json()["choices"][0]["message"]["content"]).strip().strip('"')
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape: {exc}") from exc
