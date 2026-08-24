"""Fake extractor, factory dispatch and keyword aggregation -- no model download."""

from keyword_extractor.config import Settings
from keyword_extractor.extraction import FakeExtractor, aggregate_keywords, build_extractor


def test_fake_extractor_deterministic():
    docs = ["Graph neural networks for molecule property prediction. Graph methods."]
    first = FakeExtractor(top_n=3).extract(docs)
    second = FakeExtractor(top_n=3).extract(docs)
    assert first == second
    assert first[0][0][0] == "graph"  # most frequent content word wins


def test_fake_extractor_strips_stopwords():
    [keywords] = FakeExtractor().extract(["the and of learning learning"])
    phrases = [p for p, _ in keywords]
    assert phrases == ["learning"]


def test_build_extractor_dispatches_fake():
    extractor = build_extractor(Settings(keybert_model="fake"))
    assert isinstance(extractor, FakeExtractor)


def test_aggregate_sums_scores_across_docs():
    per_doc = [
        [("machine learning", 0.5), ("optimization", 0.4)],
        [("machine learning", 0.5), ("game theory", 0.6)],
    ]
    keywords = aggregate_keywords(per_doc, top_n=3)
    assert keywords[0] == "machine learning"  # 1.0 beats 0.6


def test_aggregate_drops_word_subset_phrases():
    per_doc = [[("machine learning", 0.9), ("learning", 0.8), ("deep learning", 0.7)]]
    keywords = aggregate_keywords(per_doc, top_n=5)
    assert "machine learning" in keywords
    assert "learning" not in keywords  # subset of a higher-ranked phrase
    assert "deep learning" in keywords  # overlaps but is not a subset


def test_aggregate_collapses_plural_variants():
    per_doc = [[("ureteral stents", 0.9), ("ureteral stent", 0.8)]]
    assert aggregate_keywords(per_doc, top_n=5) == ["ureteral stents"]


def test_aggregate_respects_top_n():
    per_doc = [[(f"kw{i}", 1.0 - i / 10) for i in range(10)]]
    assert len(aggregate_keywords(per_doc, top_n=4)) == 4
