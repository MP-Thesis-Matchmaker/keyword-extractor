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


def test_concurrency_preserves_row_order():
    """Reassembly is by index, never by completion order."""
    import random
    import time

    def find(query, top_k):
        time.sleep(random.uniform(0, 0.01))
        # Echo the query so the mapping from row to result is checkable.
        return [{"supervisor": query}]

    rows = [_row(query=f"q{i}", gold=f"q{i}") for i in range(24)]
    report = evaluate(rows, top_k=5, find=find, concurrency=4)
    assert [r["query"] for r in report.rows] == [f"q{i}" for i in range(24)]
    # Every row matched its own echoed gold, so a shuffle would show as misses.
    assert report.summary["overall"]["hit@1"] == 1.0


def test_find_error_aborts_the_run():
    """A failed call must not be averaged in as a miss."""
    calls = []

    def find(query, top_k):
        calls.append(query)
        if len(calls) == 3:
            raise RuntimeError("matcher fell over")
        return [{"supervisor": "Doe, Jane"}]

    rows = [_row(query=f"q{i}", gold="Doe, Jane") for i in range(6)]
    with pytest.raises(RuntimeError, match="matcher fell over"):
        evaluate(rows, top_k=5, find=find)


def test_matcher_client_satisfies_the_find_seam():
    """The real client, over a mock transport, is a drop-in FindFn."""
    import httpx

    from keyword_extractor.matcher_client import MatcherClient

    def handler(request):
        return httpx.Response(200, json={"matches": [{"supervisor": "Doe, Jane"}]})

    client = MatcherClient(
        base_url="http://matcher.test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    report = evaluate([_row(query="q", gold="Doe, Jane")], top_k=5, find=client.match)
    assert report.summary["overall"]["hit@1"] == 1.0


def _multi_gold_row(
    query: str,
    golds: list[str],
    source_type: str = "posting",
    complexity: str = "medium",
    metadata: dict | None = None,
) -> BenchmarkRow:
    return BenchmarkRow(
        query=query,
        keywords=["k"],
        source_type=source_type,
        source_ids=["id-1"],
        gold_supervisors=[GoldSupervisor(name=g) for g in golds],
        granularity="posting",
        query_style="template",
        complexity=complexity,
        metadata=metadata or {},
    )


def test_recall_at_k_multi_gold():
    """recall@k with 2 gold supervisors, only one found → recall = 0.5, hit = 1.0."""
    rows = [_multi_gold_row("q1", ["Doe, Jane", "Smith, John"])]
    find = _find_factory({"q1": ["Doe, Jane", "Other, Person"]})
    report = evaluate(rows, top_k=10, find=find)
    overall = report.summary["overall"]
    # hit@10 = 1.0 (at least one gold found)
    assert overall["hit@10"] == 1.0
    # recall@10 = 0.5 (1 of 2 golds found)
    assert overall["recall@10"] == 0.5
    # Per-row recall_at recorded
    assert report.rows[0]["recall_at"][10] == 0.5


def test_recall_equals_hit_for_single_gold():
    """For single-gold rows, recall@k should equal hit@k."""
    rows = [_row("q1", "Doe, Jane"), _row("q2", "Roe, Richard")]
    find = _find_factory({"q1": ["Doe, Jane"], "q2": ["Nobody"]})
    report = evaluate(rows, top_k=10, find=find)
    overall = report.summary["overall"]
    assert overall["recall@1"] == overall["hit@1"]
    assert overall["recall@5"] == overall["hit@5"]
    assert overall["recall@10"] == overall["hit@10"]


def test_difficulty_breakdown():
    """Difficulty tiers from metadata are reported in the summary."""
    rows = [
        _multi_gold_row(
            "q1", ["Doe, Jane"], source_type="publication", metadata={"difficulty": "easy"}
        ),
        _multi_gold_row(
            "q2", ["Roe, Richard"], source_type="publication", metadata={"difficulty": "hard"}
        ),
    ]
    find = _find_factory({"q1": ["Doe, Jane"], "q2": ["Nobody"]})
    report = evaluate(rows, top_k=5, find=find)
    assert report.summary["difficulty:easy"]["hit@1"] == 1.0
    assert report.summary["difficulty:hard"]["hit@1"] == 0.0
