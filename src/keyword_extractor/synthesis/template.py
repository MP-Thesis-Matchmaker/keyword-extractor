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

_ENGLISH_TEMPLATES = (
    "I am looking for a thesis about {keywords}.",
    "I want to write my thesis on {keywords}. Who could supervise me?",
    "Looking for a supervisor for a thesis project on {keywords}.",
    "I'm interested in {keywords} and I'm searching for a thesis topic in this area.",
)

# German phrasings mirroring the English set one for one. Only German gets its
# own set: the fra/ita/spa tail is ~700 publications, too few supervisors to
# dominate a row, so those keep native keywords inside an English carrier phrase.
_GERMAN_TEMPLATES = (
    "Ich suche eine Abschlussarbeit über {keywords}.",
    "Ich möchte meine Abschlussarbeit über {keywords} schreiben. Wer könnte mich betreuen?",
    "Ich suche eine Betreuerin oder einen Betreuer für eine Arbeit zu {keywords}.",
    "Mich interessieren {keywords} und ich suche dazu ein Thema für meine Abschlussarbeit.",
)

_TEMPLATES_BY_LANGUAGE = {"deu": _GERMAN_TEMPLATES, "eng": _ENGLISH_TEMPLATES}

_CONJUNCTION = {"deu": "und", "eng": "and"}

_EMPTY_TOPIC = {"deu": "ein offenes Forschungsthema", "eng": "an open research topic"}


def _join(keywords: Sequence[str], language: str) -> str:
    words = [k for k in keywords if k]
    if not words:
        return _EMPTY_TOPIC[language]
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + f" {_CONJUNCTION[language]} " + words[-1]


class TemplateSynthesizer:
    """Deterministic, offline query phrasing."""

    def synthesize(self, keywords: Sequence[str], context: QueryContext) -> str:
        # Anything that is not German is phrased in English, including None.
        language = context.language if context.language in _TEMPLATES_BY_LANGUAGE else "eng"
        templates = _TEMPLATES_BY_LANGUAGE[language]
        digest = hashlib.sha256(context.seed.encode("utf-8")).digest()
        return templates[digest[0] % len(templates)].format(keywords=_join(keywords, language))
