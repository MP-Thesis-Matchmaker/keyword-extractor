"""Query synthesis: LLM when configured, deterministic template otherwise."""

from __future__ import annotations

from keyword_extractor.config import Settings
from keyword_extractor.synthesis.base import (
    COMPLEXITIES,
    COMPLEXITY_KEYWORD_COUNTS,
    Complexity,
    QueryContext,
    QuerySynthesizer,
    SynthesizedQuery,
)
from keyword_extractor.synthesis.template import TemplateSynthesizer

__all__ = [
    "COMPLEXITIES",
    "COMPLEXITY_KEYWORD_COUNTS",
    "Complexity",
    "QueryContext",
    "QuerySynthesizer",
    "SynthesizedQuery",
    "TemplateSynthesizer",
    "build_synthesizer",
]


def build_synthesizer(settings: Settings) -> QuerySynthesizer:
    """LLM synthesis iff llm_base_url is set -- backend-core's parser toggle."""
    if settings.llm_base_url:
        from keyword_extractor.synthesis.llm import LLMSynthesizer

        return LLMSynthesizer(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            reasoning_effort=settings.llm_reasoning_effort,
        )
    return TemplateSynthesizer()
