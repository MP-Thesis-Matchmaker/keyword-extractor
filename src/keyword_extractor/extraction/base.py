"""Contract for keyword extractors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class KeywordExtractor(Protocol):
    """Extracts scored keyphrases from documents.

    One inner list per input document, each entry a (phrase, score) pair with
    higher scores meaning more representative of the document.
    """

    def extract(self, docs: Sequence[str]) -> list[list[tuple[str, float]]]: ...
