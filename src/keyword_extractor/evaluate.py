"""Trial retrieval: run benchmark queries against the matcher and score them.

hit@k and MRR of the gold supervisor in the ranked results. The retriever is
reached through an injectable `find` callable, which keeps the metrics testable
with no matcher and no network; the default calls the matcher's HTTP API.

That default used to build backend-core's VectorRetriever in this process, and
refused to start without an index manifest because backend-core would otherwise
have fallen back to a FakeRetriever and benchmarked three canned matches. The
manifest check has not gone away, it has moved: retrieval is a separate process
now, so what has to be established up front is not "is there an index" but
"WHICH index is about to answer". `keyword-extractor evaluate` asks
`GET /v1/index/status` before the first query and records the answer in the
report -- a score that does not name the index that produced it cannot be
reproduced, which is exactly how the previous benchmark went stale unnoticed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from keyword_extractor.benchmark import BenchmarkRow
from keyword_extractor.matcher_client import MatcherClient

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

    def recall(k: int) -> float:
        return sum(r.get("recall_at", {}).get(k, 0.0) for r in row_results) / n

    mrr = sum(1.0 / r["rank"] for r in row_results if r["rank"] is not None) / n
    return {
        "n": n,
        "hit@1": round(hit(1), 4),
        "hit@5": round(hit(5), 4),
        f"hit@{top_k}": round(hit(top_k), 4),
        "mrr": round(mrr, 4),
        "recall@1": round(recall(1), 4),
        "recall@5": round(recall(5), 4),
        f"recall@{top_k}": round(recall(top_k), 4),
    }


def evaluate(
    rows: Sequence[BenchmarkRow],
    *,
    top_k: int = 10,
    find: FindFn | None = None,
    concurrency: int = 1,
) -> EvalReport:
    """Rank of the first gold supervisor per query, aggregated to hit@k / MRR.

    Matching is by casefolded name: retrieval output has no researcher UUID,
    so evaluation cannot be stricter than the system's own output vocabulary.

    `concurrency` > 1 dispatches queries over a thread pool -- the calls are
    network-bound, so this is most of the wall-clock in a real run. Results are
    reassembled by index, never by completion order, so the per-row output file
    is byte-identical whatever the threads do. At 1 no pool is created at all,
    which keeps the offline tests single-threaded.

    A `find` that raises aborts the run rather than recording a miss. A failed
    call and a genuine miss are indistinguishable once both are `rank: None`,
    and quietly averaging retrieval failures into hit@k understates the system
    while looking like a result.
    """
    find = find or _default_find()
    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            all_results = list(pool.map(lambda row: find(row.query, top_k), rows))
    else:
        all_results = [find(row.query, top_k) for row in rows]

    row_results: list[dict[str, Any]] = []
    for row, results in zip(rows, all_results, strict=True):

        def _normalize_name(name: str) -> frozenset[str]:
            # "Sennrich, Rico" -> {"sennrich", "rico"}
            # "Rico Sennrich" -> {"rico", "sennrich"}
            return frozenset(part.strip() for part in name.casefold().replace(",", " ").split())

        names_norm = [
            _normalize_name(str(r.get("supervisor") or r.get("name") or "")) for r in results
        ]
        gold_norm = {_normalize_name(g.name) for g in row.gold_supervisors}

        rank = next((i + 1 for i, name_n in enumerate(names_norm) if name_n in gold_norm), None)

        # recall@k: fraction of gold supervisors found in top-k.
        # For single-gold rows (publications), recall@k ≡ hit@k.
        # For multi-gold rows (postings with several supervisors), this is where
        # recall carries independent signal.
        gold_count = len(gold_norm)
        recall_at = {}
        for k in (1, 5, top_k):
            found = len({name_n for name_n in names_norm[:k] if name_n in gold_norm})
            recall_at[k] = found / gold_count if gold_count else 0.0
        row_results.append(
            {
                "query": row.query,
                "source_type": row.source_type,
                "complexity": row.complexity,
                "query_style": row.query_style,
                "source_ids": row.source_ids,
                "gold": [g.name for g in row.gold_supervisors],
                "rank": rank,
                "recall_at": recall_at,
                "difficulty": row.metadata.get("difficulty"),
                "rung": row.metadata.get("rung"),
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
    # Difficulty tiers (Part 1b): easy/hard based on gold supervisor's indexed
    # publication count. Only present on rows generated with --difficulty-threshold.
    for difficulty in ("easy", "hard"):
        subset = [r for r in row_results if r.get("difficulty") == difficulty]
        if subset:
            summary[f"difficulty:{difficulty}"] = _metrics(subset, top_k)
        pub_subset = [r for r in subset if r["source_type"] == "publication"]
        if pub_subset:
            summary[f"difficulty:{difficulty} (pubs)"] = _metrics(pub_subset, top_k)
    return EvalReport(summary=summary, rows=row_results)


def _default_find(client: MatcherClient | None = None) -> FindFn:
    """Retrieval through the matcher's HTTP API.

    No index check here: the CLI preflights `GET /v1/index/status` before the
    first query, so a matcher with nothing indexed fails the run in a second
    rather than on row one of several hundred.
    """
    return (client or MatcherClient()).match
