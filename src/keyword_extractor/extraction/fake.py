"""Deterministic offline keyword extractor.

Frequency-ranked unigrams after a small stop-word strip. No semantics and no
dependencies -- the point is a real, deterministic implementation (backend-core's
hash-fake embedder precedent) so tests and CI exercise the full pipeline
without torch or a model download.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from keyword_extractor.language import ENGLISH_FUNCTION_WORDS, GERMAN_FUNCTION_WORDS

# Unicode-aware so German words survive tokenisation intact; three characters
# minimum, which drops most inflectional debris.
_WORD_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)

# English and German, so the fake behaves sensibly on the German share of the
# corpus instead of returning "der" and "und" as a supervisor's research topics.
# NOT a linguistic claim (see backend-core indexing/documents.py on why real
# transformer pipelines keep their stop words).
_STOPWORDS = (
    ENGLISH_FUNCTION_WORDS
    | GERMAN_FUNCTION_WORDS
    | frozenset("using use based just does doing did".split())
)


class FakeExtractor:
    """Deterministic frequency-based extractor for tests and offline runs."""

    def __init__(self, top_n: int = 8) -> None:
        self._top_n = top_n

    def extract(self, docs: Sequence[str]) -> list[list[tuple[str, float]]]:
        results = []
        for doc in docs:
            words = [w.lower() for w in _WORD_RE.findall(doc)]
            words = [w for w in words if w not in _STOPWORDS]
            counts = Counter(words)
            total = sum(counts.values()) or 1
            ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            results.append([(w, c / total) for w, c in ranked[: self._top_n]])
        return results
