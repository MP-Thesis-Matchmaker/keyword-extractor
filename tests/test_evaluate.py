"""Metrics against an injected fake retriever -- backend-core not required."""

import pytest

from keyword_extractor.benchmark import BenchmarkRow, GoldSupervisor
from keyword_extractor.evaluate import evaluate


def _row(
    query: str, gold: str, source_type: str = "publication", complexity: str = "medium"
) -> BenchmarkRow:
    return BenchmarkRow(
        query=query,
        keywords=["k"],
        source_type=source_type,
        source_ids=["id-1"],
        gold_supervisors=[GoldSupervisor(name=gold)],
        granularity="supervisor" if source_type == "publication" else "posting",
        query_style="template",
        complexity=complexity,
    )


def _find_factory(ranked_by_query):
    def find(query, top_k):
        return [{"supervisor": name} for name in ranked_by_query[query][:top_k]]

    return find


def test_exact_metrics():
    rows = [
        _row("q1", "Doe, Jane"),
        _row("q2", "Roe, Richard", source_type="posting"),
        _row("q3", "Poe, Anna"),
    ]
    find = _find_factory(
        {
            "q1": ["Doe, Jane", "Other"],  # rank 1
            "q2": ["A", "B", "roe, richard"],  # rank 3, casefold match
            "q3": ["A", "B", "C"],  # miss
        }
    )
    report = evaluate(rows, top_k=10, find=find)
    overall = report.summary["overall"]
    assert overall["n"] == 3
    assert overall["hit@1"] == pytest.approx(round(1 / 3, 4))
    assert overall["hit@5"] == pytest.approx(round(2 / 3, 4))
    assert overall["hit@10"] == pytest.approx(round(2 / 3, 4))
    assert overall["mrr"] == pytest.approx(round((1 + 1 / 3 + 0) / 3, 4))


def test_split_by_source_type():
    rows = [_row("q1", "Doe, Jane"), _row("q2", "Roe, Richard", source_type="posting")]
    find = _find_factory({"q1": ["Doe, Jane"], "q2": ["X"]})
    report = evaluate(rows, top_k=5, find=find)
    assert report.summary["publication"]["hit@1"] == 1.0
    assert report.summary["posting"]["hit@1"] == 0.0


def test_split_by_complexity():
    """The point of three levels: what does a vaguer query cost?"""
    rows = [
        _row("q-low", "Doe, Jane", complexity="low"),
        _row("q-high", "Doe, Jane", complexity="high"),
    ]
    find = _find_factory({"q-low": ["Someone Else"], "q-high": ["Doe, Jane"]})
    report = evaluate(rows, top_k=5, find=find)
    assert report.summary["complexity:low"]["hit@5"] == 0.0
    assert report.summary["complexity:high"]["hit@5"] == 1.0
    assert "complexity:medium" not in report.summary  # absent levels are not invented


def test_rank_none_recorded_for_miss():
    rows = [_row("q1", "Doe, Jane")]
    find = _find_factory({"q1": ["Nobody"]})
    report = evaluate(rows, top_k=5, find=find)
    assert report.rows[0]["rank"] is None


def test_result_key_fallback_to_name():
    rows = [_row("q1", "Doe, Jane")]

    def find(query, top_k):
        return [{"name": "Doe, Jane"}]

    report = evaluate(rows, top_k=5, find=find)
    assert report.rows[0]["rank"] == 1
