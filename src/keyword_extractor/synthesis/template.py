"""Deterministic template synthesis -- the offline default.

Reproducibility is a benchmark virtue: the same DB state, seed and settings
must yield byte-identical queries. Rotation across a few phrasings comes from
a stable hash of the row seed (Python's hash() is salted per process, so
hashlib is used instead).

The three complexity levels differ in kind rather than only in length. `low` is
a student who knows roughly which field they want, `high` is one who already has
a specific project and names their subject area. Only the `high` templates use
`department`, and only when it is known.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from keyword_extractor.synthesis.base import (
    COMPLEXITY_KEYWORD_COUNTS,
    Complexity,
    QueryContext,
    SynthesizedQuery,
)

_ENGLISH: dict[Complexity, tuple[str, ...]] = {
    "low": (
        "I want to write my thesis on {keywords}.",
        "I'm looking for a thesis topic in {keywords}.",
        "Looking for a supervisor for a thesis on {keywords}.",
    ),
    "medium": (
        "I am looking for a thesis about {keywords}.",
        "I want to write my thesis on {keywords}. Who could supervise me?",
        "Looking for a supervisor for a thesis project on {keywords}.",
        "I'm interested in {keywords} and I'm searching for a thesis topic in this area.",
    ),
    "high": (
        "I'd like to work on {keywords}, ideally combining several of these in one project. "
        "Who could supervise a thesis like that?",
        "My interests are {keywords}. I'm looking for a supervisor whose own research covers "
        "this ground, for a master's thesis.",
        "For my thesis I want to focus on {keywords}. I'm looking for someone working on these "
        "topics who takes on students.",
    ),
}

# High complexity gains a subject-area sentence when the department is known.
# Kept as its own set rather than an optional fragment so both variants read as
# whole sentences instead of one with a clause bolted on.
_ENGLISH_HIGH_WITH_DEPARTMENT: tuple[str, ...] = (
    "I'm looking for a thesis supervisor in {department}. I'd like to work on {keywords}, "
    "ideally combining several of these in one project.",
    "I study in {department} and my interests are {keywords}. Who works on these topics and "
    "supervises master's theses?",
    "For a thesis in {department} I want to focus on {keywords}. Who could supervise that?",
)

_GERMAN: dict[Complexity, tuple[str, ...]] = {
    "low": (
        "Ich möchte meine Abschlussarbeit über {keywords} schreiben.",
        "Ich suche ein Thema für meine Abschlussarbeit im Bereich {keywords}.",
        "Ich suche eine Betreuung für eine Arbeit zu {keywords}.",
    ),
    "medium": (
        "Ich suche eine Abschlussarbeit über {keywords}.",
        "Ich möchte meine Abschlussarbeit über {keywords} schreiben. Wer könnte mich betreuen?",
        "Ich suche eine Betreuerin oder einen Betreuer für eine Arbeit zu {keywords}.",
        "Mich interessieren {keywords} und ich suche dazu ein Thema für meine Abschlussarbeit.",
    ),
    "high": (
        "Ich möchte zu {keywords} arbeiten und diese Themen möglichst in einem Projekt "
        "verbinden. Wer könnte eine solche Arbeit betreuen?",
        "Meine Interessen sind {keywords}. Ich suche eine Betreuung, deren eigene Forschung "
        "diese Themen abdeckt.",
        "In meiner Masterarbeit möchte ich mich auf {keywords} konzentrieren. Wer forscht zu "
        "diesen Themen und betreut Studierende?",
    ),
}

_GERMAN_HIGH_WITH_DEPARTMENT: tuple[str, ...] = (
    "Ich suche eine Betreuung für meine Abschlussarbeit am {department}. Ich möchte zu "
    "{keywords} arbeiten und diese Themen verbinden.",
    "Ich studiere am {department} und interessiere mich für {keywords}. Wer forscht dazu und "
    "betreut Masterarbeiten?",
    "Für eine Arbeit am {department} möchte ich mich auf {keywords} konzentrieren. Wer käme "
    "als Betreuung infrage?",
)

_BY_LANGUAGE = {"deu": _GERMAN, "eng": _ENGLISH}
_HIGH_WITH_DEPARTMENT = {
    "deu": _GERMAN_HIGH_WITH_DEPARTMENT,
    "eng": _ENGLISH_HIGH_WITH_DEPARTMENT,
}

_CONJUNCTION = {"deu": "und", "eng": "and"}

_EMPTY_TOPIC = {"deu": "ein offenes Forschungsthema", "eng": "an open research topic"}


def _join(keywords: Sequence[str], language: str) -> str:
    words = [k for k in keywords if k]
    if not words:
        return _EMPTY_TOPIC[language]
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + f" {_CONJUNCTION[language]} " + words[-1]


def select_keywords(keywords: Sequence[str], complexity: Complexity) -> list[str]:
    """The front slice of the score-ordered keywords this level asks for."""
    return list(keywords[: COMPLEXITY_KEYWORD_COUNTS[complexity]])


def _templates(language: str, context: QueryContext) -> tuple[str, ...]:
    if context.complexity == "high" and context.department:
        return _HIGH_WITH_DEPARTMENT[language]
    return _BY_LANGUAGE[language][context.complexity]


class TemplateSynthesizer:
    """Deterministic, offline query phrasing."""

    def synthesize(self, keywords: Sequence[str], context: QueryContext) -> SynthesizedQuery:
        # Anything that is not German is phrased in English, including None.
        language = context.language if context.language in _BY_LANGUAGE else "eng"
        templates = _templates(language, context)
        # The complexity is part of the seed so the three levels of one row do
        # not all land on the same index of their respective template tuples.
        digest = hashlib.sha256(f"{context.seed}:{context.complexity}".encode()).digest()
        text = templates[digest[0] % len(templates)].format(
            keywords=_join(select_keywords(keywords, context.complexity), language),
            department=context.department or "",
        )
        return SynthesizedQuery(text=text, style="template")
