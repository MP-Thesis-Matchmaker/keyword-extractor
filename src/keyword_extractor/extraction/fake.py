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

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z-]{2,}")

# Just enough to keep the fake's output looking like keywords rather than
# function words; NOT a linguistic claim (see backend-core indexing/documents.py
# on why real pipelines keep stop words for transformers).
_STOPWORDS = frozenset(
    """a an and are as at be but by for from has have in is it its of on or that the
    this to was we were which with not their our they using use based can also than
    these those between during into through over under both more most other some such
    only own same so too very will just about against each few how all any been being
    does doing did further here there when where why what who whom""".split()
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
