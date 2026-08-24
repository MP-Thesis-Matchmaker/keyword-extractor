"""KeyBERT-backed keyword extractor.

Imports keybert lazily inside __init__ so the default install (no `extraction`
extra, no torch) can import this module's package without paying for the
dependency it does not have.
"""

from __future__ import annotations

from collections.abc import Sequence

from keyword_extractor.config import Settings


class KeyBertExtractor:
    """Real KeyBERT extraction; requires the `extraction` extra."""

    def __init__(self, settings: Settings) -> None:
        from keybert import KeyBERT  # lazy: needs the extraction extra

        self._model = KeyBERT(model=settings.keybert_model)
        self._top_n = settings.keybert_top_n
        self._ngram_range = (1, settings.keybert_ngram_max)
        self._use_mmr = settings.keybert_use_mmr
        self._diversity = settings.keybert_diversity

    def extract(self, docs: Sequence[str]) -> list[list[tuple[str, float]]]:
        if not docs:
            return []
        # One call with the whole list: KeyBERT batches the embedding forward
        # passes, which is what makes a few hundred documents take minutes on
        # CPU instead of hours.
        result = self._model.extract_keywords(
            list(docs),
            keyphrase_ngram_range=self._ngram_range,
            stop_words="english",
            use_mmr=self._use_mmr,
            diversity=self._diversity,
            top_n=self._top_n,
        )
        # KeyBERT returns a flat [(phrase, score), ...] for a single document
        # and [[...], ...] for several; normalise to one shape.
        if result and isinstance(result[0], tuple):
            return [result]
        return result
