"""Keyword extraction: KeyBERT behind a Protocol, with a deterministic fake."""

from __future__ import annotations

from keyword_extractor.config import Settings
from keyword_extractor.extraction.base import KeywordExtractor
from keyword_extractor.extraction.fake import FakeExtractor

__all__ = ["KeywordExtractor", "FakeExtractor", "aggregate_keywords", "build_extractor"]


def build_extractor(settings: Settings) -> KeywordExtractor:
    """The extractor the settings ask for; "fake" needs no model download.

    Same dispatch shape as backend-core's embedding_model == "hash-fake".
    """
    if settings.keybert_model == "fake":
        return FakeExtractor(top_n=settings.keybert_top_n)
    from keyword_extractor.extraction.keybert_extractor import KeyBertExtractor

    return KeyBertExtractor(settings)


def aggregate_keywords(per_doc: list[list[tuple[str, float]]], top_n: int = 10) -> list[str]:
    """Pool per-document keyphrases into one supervisor-level keyword list.

    Scores are summed per casefolded phrase across documents, so a phrase that
    recurs through a supervisor's work outranks a one-off. A phrase whose words
    are a subset (or superset) of an already-kept phrase's words is dropped --
    "machine learning" and "learning" say the same thing once each. Words are
    compared with a trailing-s strip so plural variants ("ureteral stent" /
    "ureteral stents") collapse too.
    """
    totals: dict[str, float] = {}
    surface: dict[str, str] = {}
    for doc_keywords in per_doc:
        for phrase, score in doc_keywords:
            key = phrase.casefold().strip()
            if not key:
                continue
            totals[key] = totals.get(key, 0.0) + float(score)
            surface.setdefault(key, phrase.strip())
    ranked = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    kept: list[str] = []
    kept_words: list[set[str]] = []
    for key, _ in ranked:
        words = {w.rstrip("s") or w for w in key.split()}
        if any(words <= other or words >= other for other in kept_words):
            continue
        kept.append(surface[key])
        kept_words.append(words)
        if len(kept) >= top_n:
            break
    return kept
