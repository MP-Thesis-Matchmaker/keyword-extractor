"""Trial retrieval: run benchmark queries against the matchmaker and score them.

hit@k and MRR of the gold supervisor in the ranked results. The retriever is
reached through an injectable `find` callable so the metrics are testable
without backend-core installed; the default builds the REAL VectorRetriever
explicitly and refuses to run without an index manifest -- backend-core's
`_default_pipeline` silently falls back to a FakeRetriever that returns three
canned matches, which would benchmark noise without failing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from keyword_extractor.benchmark import BenchmarkRow

# (query, top_k) -> ranked researcher dicts, each carrying a "supervisor" name.
FindFn = Callable[[str, int], list[dict[str, Any]]]


@dataclass(frozen=True)
class EvalReport:
    """Aggregate metrics plus the per-row ranks that drive the adjust loop."""

    summary: dict[str, dict[str, float | int]]
    rows: list[dict[str, Any]]


def _metrics(row_results: list[dict[str, Any]], top_k: int) -> dict[str, float | int]:
    n = len(row_results)

    def hit(k: int) -> float:
        return sum(1 for r in row_results if r["rank"] is not None and r["rank"] <= k) / n

    mrr = sum(1.0 / r["rank"] for r in row_results if r["rank"] is not None) / n
    return {
        "n": n,
        "hit@1": round(hit(1), 4),
        "hit@5": round(hit(5), 4),
        f"hit@{top_k}": round(hit(top_k), 4),
        "mrr": round(mrr, 4),
    }


def evaluate(
    rows: Sequence[BenchmarkRow], *, top_k: int = 10, find: FindFn | None = None
) -> EvalReport:
    """Rank of the first gold supervisor per query, aggregated to hit@k / MRR.

    Matching is by casefolded name: retrieval output has no researcher UUID,
    so evaluation cannot be stricter than the system's own output vocabulary.
    """
    find = find or _default_find()
    row_results: list[dict[str, Any]] = []
    for row in rows:
        results = find(row.query, top_k)
        names = [str(r.get("supervisor") or r.get("name") or "").casefold() for r in results]
        gold = {g.name.casefold() for g in row.gold_supervisors}
        rank = next((i + 1 for i, name in enumerate(names) if name in gold), None)
        row_results.append(
            {
                "query": row.query,
                "source_type": row.source_type,
                "complexity": row.complexity,
                "query_style": row.query_style,
                "source_ids": row.source_ids,
                "gold": [g.name for g in row.gold_supervisors],
                "rank": rank,
            }
        )
    summary = {"overall": _metrics(row_results, top_k)}
    for source_type in ("publication", "posting"):
        subset = [r for r in row_results if r["source_type"] == source_type]
        if subset:
            summary[source_type] = _metrics(subset, top_k)
    # The reason three complexity levels exist: how much does a vaguer query
    # cost you? Reported separately rather than averaged away.
    for complexity in ("low", "medium", "high"):
        subset = [r for r in row_results if r["complexity"] == complexity]
        if subset:
            summary[f"complexity:{complexity}"] = _metrics(subset, top_k)
    return EvalReport(summary=summary, rows=row_results)


def _default_find() -> FindFn:
    """The real backend-core retriever; requires the `eval` extra and an index."""
    from thesis_matchmaker.adapters.service import find_researchers
    from thesis_matchmaker.config import get_settings
    from thesis_matchmaker.indexing import read_manifest
    from thesis_matchmaker.pipeline import Pipeline
    from thesis_matchmaker.retrieval import build_retriever

    settings = get_settings()
    if read_manifest(settings) is None:
        raise RuntimeError(
            "No index manifest found in the database -- refusing to evaluate against "
            "the canned FakeRetriever. Build the index in backend-core first "
            "(thesis-matchmaker index), or point DATABASE_URL at the demo DB."
        )
    pipeline = Pipeline(retriever=build_retriever(settings))
    return lambda query, top_k: find_researchers(query, top_k=top_k, pipeline=pipeline)
