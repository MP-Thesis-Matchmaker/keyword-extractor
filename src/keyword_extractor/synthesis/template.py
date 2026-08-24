"""Deterministic template synthesis -- the offline default.

Reproducibility is a benchmark virtue: the same DB state, seed and settings
must yield byte-identical queries. Rotation across a few phrasings comes from
a stable hash of the row seed (Python's hash() is salted per process, so
hashlib is used instead).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from keyword_extractor.synthesis.base import QueryContext

_TEMPLATES = (
    "I am looking for a thesis about {keywords}.",
    "I want to write my thesis on {keywords}. Who could supervise me?",
    "Looking for a supervisor for a thesis project on {keywords}.",
    "I'm interested in {keywords} and I'm searching for a thesis topic in this area.",
)


def _join(keywords: Sequence[str]) -> str:
    words = [k for k in keywords if k]
    if not words:
        return "an open research topic"
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + " and " + words[-1]


class TemplateSynthesizer:
    """Deterministic, offline query phrasing."""

    def synthesize(self, keywords: Sequence[str], context: QueryContext) -> str:
        digest = hashlib.sha256(context.seed.encode("utf-8")).digest()
        template = _TEMPLATES[digest[0] % len(_TEMPLATES)]
        return template.format(keywords=_join(keywords))
