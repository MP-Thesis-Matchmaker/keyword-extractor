"""Contract for query synthesisers, plus the complexity ladder."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

#: How elaborate a student's query is. The levels are meant to differ in kind,
#: not only in length: `low` is someone who knows the field they want, `high` is
#: someone who already has a specific project in mind.
Complexity = Literal["low", "medium", "high"]

COMPLEXITIES: tuple[Complexity, ...] = ("low", "medium", "high")

#: Keywords each level uses, taken from the front of the score-ordered list that
#: `aggregate_keywords` returns -- so all three levels come out of one KeyBERT
#: run, and `low` gets the supervisor's most representative topics rather than
#: an arbitrary pair.
COMPLEXITY_KEYWORD_COUNTS: dict[Complexity, int] = {"low": 2, "medium": 4, "high": 8}


@dataclass(frozen=True)
class QueryContext:
    """Per-row context for synthesis.

    `seed` identifies the row (supervisor CRIS UUID / name, or posting id) and
    drives deterministic template rotation. Deliberately NOT the supervisor's
    display name in any form the template would print: the query must never
    lexically leak its own gold label.
    """

    seed: str
    department: str | None = None
    # Language of the text the keywords came from ("deu" / "eng" / None),
    # detected rather than read from the database -- see language.py for why.
    # Chooses the phrasing; anything but "deu" is phrased in English.
    language: str | None = None
    complexity: Complexity = "medium"


@dataclass(frozen=True)
class SynthesizedQuery:
    """A query and the route that actually produced it.

    `style` is what happened, not what was asked for: an LLM synthesiser that
    falls back to the template reports "template". Recording the request instead
    of the outcome is how a run of 100 rows once claimed LLM phrasing while
    every row held template text.
    """

    text: str
    style: Literal["template", "llm"]


class QuerySynthesizer(Protocol):
    """Turns a keyword list into one natural-language student query."""

    def synthesize(self, keywords: Sequence[str], context: QueryContext) -> SynthesizedQuery: ...
