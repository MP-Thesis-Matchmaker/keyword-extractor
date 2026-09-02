#!/usr/bin/env python3
"""Analyse baseline evaluation results and difficulty tiers.

Reads the per-row evaluate output (e.g. data/eval_report_v4.jsonl) and prints
summary tables for source types, complexity levels, and difficulty tiers.

Usage:
    python scripts/analyze_eval.py data/eval_report_v4.jsonl [--top-k 10]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from keyword_extractor.evaluate import _metrics


def _fmt(m: dict[str, float | int]) -> str:
    return (
        f"{m['n']:>5}  "
        f"{m['hit@1']:>7.3f}  "
        f"{m['hit@5']:>7.3f}  "
        f"{m['hit@10']:>7.3f}  "
        f"{m['mrr']:>7.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="per-row evaluate output JSONL (--out from evaluate)")
    parser.add_argument("--top-k", type=int, default=10, help="top-k used in evaluate run")
    args = parser.parse_args()

    lines = Path(args.input).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]

    if not rows:
        print("ERROR: no rows found in input file.")
        return 1

    header = f"{'Subset':<25} {'n':>5}  {'hit@1':>7}  {'hit@5':>7}  {'hit@10':>7}  {'MRR':>7}"
    sep = "-" * len(header)

    print("=== Evaluation Summary ===")
    print(header)
    print(sep)

    print(f"{'overall':<25} {_fmt(_metrics(rows, args.top_k))}")

    # By source type
    for st in ("publication", "posting"):
        sub = [r for r in rows if r.get("source_type") == st]
        if sub:
            print(f"{st:<25} {_fmt(_metrics(sub, args.top_k))}")

    # By complexity
    print("\n--- By Complexity ---")
    print(header)
    print(sep)
    for comp in ("low", "medium", "high"):
        sub = [r for r in rows if r.get("complexity") == comp]
        if sub:
            print(f"{comp:<25} {_fmt(_metrics(sub, args.top_k))}")

    # By difficulty
    diff_rows = [r for r in rows if r.get("difficulty")]
    if diff_rows:
        print("\n--- Difficulty Tiers (Publication Only) ---")
        print(header)
        print(sep)
        for diff in ("easy", "hard"):
            sub = [
                r
                for r in diff_rows
                if r.get("difficulty") == diff and r.get("source_type") == "publication"
            ]
            if sub:
                label = f"{diff} (>=10 pubs)" if diff == "easy" else f"{diff} (<10 pubs)"
                print(f"{label:<25} {_fmt(_metrics(sub, args.top_k))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
