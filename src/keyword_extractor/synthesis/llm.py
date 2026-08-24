"""LLM-backed query synthesis over an OpenAI-compatible chat endpoint.

Modelled on backend-core's llm.py, cut down to what this repo needs: one POST
to /chat/completions, temperature 0 for reproducibility, and a fall-back to
the deterministic template on any transport or shape error -- a benchmark run
should degrade, not die, when Ollama is not up.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import httpx

from keyword_extractor.synthesis.base import QueryContext
from keyword_extractor.synthesis.template import TemplateSynthesizer

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You write realistic queries a student would send to a thesis-supervisor "
    "matching assistant. Rewrite the given keywords into one short natural-language "
    "query of one or two sentences describing the thesis topic the student is "
    "looking for. Do not mention any person, author or supervisor names. "
    "Return only the query text, no quotes and no explanations."
)


class LLMSynthesizer:
    """Query phrasing via an OpenAI-compatible endpoint, template on failure."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._timeout = timeout
        # Injectable for tests (httpx.MockTransport); owned here otherwise.
        self._client = client or httpx.Client(timeout=timeout)
        self._fallback = TemplateSynthesizer()

    def synthesize(self, keywords: Sequence[str], context: QueryContext) -> str:
        user = "Keywords: " + "; ".join(keywords)
        if context.department:
            user += f"\nSubject area: {context.department}"
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
        }
        try:
            response = self._client.post(self._url, json=payload, headers=self._headers)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            logger.warning("LLM synthesis failed (%s); using template fallback", exc)
            return self._fallback.synthesize(keywords, context)
        text = str(content).strip().strip('"')
        if not text:
            return self._fallback.synthesize(keywords, context)
        return text
