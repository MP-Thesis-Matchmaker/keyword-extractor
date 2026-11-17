#!/usr/bin/env python3
"""Analyse student-background ablation results.

Reads the per-row evaluate output for a background-ablation benchmark and
reports how each arm (topic_only / plus_program / plus_background / full)
performs, plus deltas vs the topic_only baseline.

Usage:
    python scripts/analyze_ablation.py data/ablation_out.jsonl [--top-k 10]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="per-row evaluate output JSONL (--out from evaluate)")
    parser.add_argument("--top-k", type=int, default=10, help="top-k used in evaluate run")
    args = parser.parse_args()

    rows_by_arm: dict[str, list[dict]] = defaultdict(list)

    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        # Determine arm from query_style (e.g. "ablation:topic_only")
        qs = row.get("query_style", "")
        if qs.startswith("ablation:"):
            arm = qs.split(":", 1)[1]
        else:
            arm = row.get("ablation_arm") or row.get("metadata", {}).get("ablation_arm", "unknown")
        rows_by_arm[arm].append(row)

    if not rows_by_arm:
        print("ERROR: no ablation rows found in the input file.")
        return 1

    # Compute metrics per arm
    arm_metrics: dict[str, dict[str, float]] = {}
    for arm, rows in sorted(rows_by_arm.items()):
        n = len(rows)
        ranks = [r.get("rank") for r in rows]

        def hit(k: int, ranks: list = ranks, n: int = n) -> float:
            return sum(1 for r in ranks if r is not None and r <= k) / n

        mrr = sum(1.0 / r for r in ranks if r is not None) / n

        # Recall from recall_at if available
        recall_5 = (
            sum(r.get("recall_at", {}).get(5, r.get("recall_at", {}).get("5", 0.0)) for r in rows)
            / n
        )

        arm_metrics[arm] = {
            "n": n,
            "hit@1": hit(1),
            "hit@5": hit(5),
            f"hit@{args.top_k}": hit(args.top_k),
            "mrr": mrr,
            "recall@5": recall_5,
        }

    # Print results table
    header = (
        f"{'Arm':<20} {'n':>4} {'hit@1':>7} {'hit@5':>7} {'hit@10':>7} {'MRR':>7} {'recall@5':>9}"
    )
    print(header)
    print("-" * len(header))

    baseline = arm_metrics.get("topic_only", {})

    for arm in ("topic_only", "plus_program", "plus_background", "full"):
        if arm not in arm_metrics:
            continue
        m = arm_metrics[arm]
        print(
            f"{arm:<20} {m['n']:>4} "
            f"{m['hit@1']:>7.3f} {m['hit@5']:>7.3f} {m[f'hit@{args.top_k}']:>7.3f} "
            f"{m['mrr']:>7.3f} {m['recall@5']:>9.3f}"
        )

    # Print deltas vs baseline
    if baseline:
        print()
        print("Deltas vs topic_only:")
        print(f"{'Arm':<20} {'d_hit@1':>7} {'d_hit@5':>7} {'d_MRR':>7}")
        print("-" * 45)
        for arm in ("plus_program", "plus_background", "full"):
            if arm not in arm_metrics:
                continue
            m = arm_metrics[arm]
            print(
                f"{arm:<20} "
                f"{m['hit@1'] - baseline['hit@1']:>+7.3f} "
                f"{m['hit@5'] - baseline['hit@5']:>+7.3f} "
                f"{m['mrr'] - baseline['mrr']:>+7.3f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
