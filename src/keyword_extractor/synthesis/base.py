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


class QuerySynthesizer(Protocol):
    """Turns a keyword list into one natural-language student query."""

    def synthesize(self, keywords: Sequence[str], context: QueryContext) -> str: ...
