"""Contract for query synthesisers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


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


class QuerySynthesizer(Protocol):
    """Turns a keyword list into one natural-language student query."""

    def synthesize(self, keywords: Sequence[str], context: QueryContext) -> str: ...
