"""LLM-backed query synthesis over an OpenAI-compatible chat endpoint.

The retry machinery is adapted from backend-core's
`src/thesis_matchmaker/llm.py`, which solved this exact problem first. It is
copied rather than imported because backend-core is only installed under the
`eval` extra, while synthesis has to work in the base install.

Two retries live here, and they are answers to different questions.

**The payload is wrong** (400/422): every optional field below is an extension
some endpoint does not implement, and a 400 is how it says so. `temperature: 0`
is rejected outright by OpenAI's gpt-5 and o-series ("Unsupported value:
'temperature' does not support 0 with this model"), which once turned a 100-row
LLM run into 100 silent template fallbacks. The refused fields are remembered
per (endpoint, model) at MODULE level, so the doomed request is paid once per
process rather than once per row -- that, not the retry, is what makes a large
run survive an unsupported field. Correcting the payload costs no attempt.

**The endpoint is momentarily unable** (429, 5xx, timeout, reset): retried up
to `max_attempts` times with exponential backoff, honouring `Retry-After`. A
full run is thousands of sequential calls over hours, so without this a single
blip silently downgrades one row to template phrasing and the benchmark carries
the scar with no way to tell it apart from a real fallback.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

import httpx

from keyword_extractor.synthesis.base import Complexity, QueryContext, SynthesizedQuery
from keyword_extractor.synthesis.template import TemplateSynthesizer, select_keywords

logger = logging.getLogger(__name__)

# Which optional request fields an endpoint refuses, keyed by (url, model).
# Module-level rather than per instance: it is a property of the endpoint, not
# of the object asking.
_UNSUPPORTED_FIELDS: dict[tuple[str, str], set[str]] = {}

# Statuses that mean "not now" rather than "not ever": worth another attempt.
# Distinct from the 400/422 handled above, which mean the payload is wrong and
# retrying it unchanged would fail identically.
_TRANSIENT_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
# An endpoint asking us to wait longer than this is telling us to stop.
_MAX_RETRY_AFTER = 60.0

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


class _Transient(Exception):
    """Retryable: the endpoint is momentarily unable, not permanently wrong.

    Private, because it never escapes this module -- exhausting the attempts
    turns it into an LLMError, which is the only failure callers handle.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """The endpoint's own wait instruction, when it gives a usable one.

    Only the delta-seconds form is honoured; the HTTP-date form is rare here
    and the computed backoff covers it. Clamped, so a rate-limit reset an hour
    out cannot wedge a run.
    """
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        return None
    return min(max(seconds, 0.0), _MAX_RETRY_AFTER)


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
        max_attempts: int = 3,
        retry_backoff: float = 1.0,
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
        # Attempts against transient failure. A long run is thousands of
        # sequential calls, so one 429 must not cost a row its LLM phrasing.
        self._max_attempts = max(1, max_attempts)
        self._retry_backoff = retry_backoff
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
            text = self._chat(user)
        except LLMError as exc:
            logger.warning("LLM synthesis failed (%s); using template fallback", exc)
            return self._fallback.synthesize(keywords, context)
        if not text:
            logger.warning("LLM returned an empty query; using template fallback")
            return self._fallback.synthesize(keywords, context)
        return SynthesizedQuery(text=text, style="llm")

    def _chat(self, user: str) -> str:
        """One query, retried while the endpoint is only transiently unable."""
        last = "no attempt made"
        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._chat_once(user, allow_field_retry=True)
            except _Transient as exc:
                last = str(exc)
                if attempt == self._max_attempts:
                    break
                delay = exc.retry_after
                if delay is None:
                    delay = self._retry_backoff * 2 ** (attempt - 1)
                logger.info(
                    "transient failure (%s); attempt %d/%d, retrying in %.1fs",
                    last,
                    attempt,
                    self._max_attempts,
                    delay,
                )
                if delay > 0:
                    time.sleep(delay)
        raise LLMError(f"{last} (after {self._max_attempts} attempts)")

    def _chat_once(self, user: str, *, allow_field_retry: bool) -> str:
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
        except httpx.TransportError as exc:
            # Timeouts, connection resets, DNS -- none of them say the request
            # was wrong. (httpx.TimeoutException is a TransportError.)
            raise _Transient(f"{type(exc).__name__}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(str(exc)) from exc

        # Retry once without whatever was refused, rather than failing outright.
        # Only the fields actually blamed are dropped.
        if response.status_code in (400, 422) and allow_field_retry and sent:
            blamed = _blamed_fields(response, sent)
            self._unsupported |= blamed
            logger.info(
                "endpoint rejected %s (%s); retrying without it and omitting it from "
                "later requests",
                ", ".join(sorted(blamed)),
                _error_message(response) or "no message",
            )
            # Not an attempt: the payload was corrected, not the endpoint retried.
            return self._chat_once(user, allow_field_retry=False)

        if response.status_code in _TRANSIENT_STATUS:
            raise _Transient(
                f"HTTP {response.status_code}: "
                f"{_error_message(response) or response.reason_phrase}",
                retry_after=_retry_after_seconds(response),
            )

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
