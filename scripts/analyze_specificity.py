#!/usr/bin/env python3
"""Analyse specificity ablation results.

Reads the per-row evaluate output for a specificity benchmark and reports
whether the embedding separates each supervisor's true subfield from the
general field and an adjacent subfield.

Usage:
    python scripts/analyze_specificity.py data/specificity_out.jsonl [--top-k 10]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="per-row evaluate output JSONL (--out from evaluate)")
    parser.add_argument("--top-k", type=int, default=10, help="top-k used in evaluate run")
    args = parser.parse_args()

    rows_by_supervisor: dict[str, dict[str, int | None]] = defaultdict(dict)

    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        golds = row.get("gold", [])
        if not golds:
            continue
        supervisor = golds[0]
        # The rung is stored in the benchmark row's metadata, but the evaluate
        # output may or may not carry it. Try multiple locations.
        rung = row.get("rung") or row.get("metadata", {}).get("rung")
        if rung is None:
            # Fall back: check if the query text gives us a hint
            continue
        rank = row.get("rank")
        rows_by_supervisor[supervisor][rung] = rank

    if not rows_by_supervisor:
        print("ERROR: no specificity rows found in the input file.", file=sys.stderr)
        return 1

    miss = args.top_k + 1  # treat miss as rank beyond top-k

    # Header
    print(f"{'Supervisor':<30} {'general':>8} {'adjacent':>8} {'true':>8} {'separation':>11}")
    print("-" * 70)

    successes = 0
    failures = 0

    for supervisor in sorted(rows_by_supervisor):
        ranks = rows_by_supervisor[supervisor]
        r_general = ranks.get("general")
        r_adjacent = ranks.get("adjacent")
        r_true = ranks.get("true")

        # For display and computation, treat None (miss) as top_k + 1
        rg = r_general if r_general is not None else miss
        ra = r_adjacent if r_adjacent is not None else miss
        rt = r_true if r_true is not None else miss

        separation = ra - rt  # positive = embedding correctly separates

        # Format rank for display
        def fmt(r: int | None) -> str:
            return str(r) if r is not None else f">{args.top_k}"

        is_failure = rt >= ra or rt >= rg
        marker = " ✗ FAIL" if is_failure else " ✓"
        if is_failure:
            failures += 1
        else:
            successes += 1

        print(
            f"{supervisor:<30} {fmt(r_general):>8} {fmt(r_adjacent):>8} "
            f"{fmt(r_true):>8} {separation:>+11}{marker}"
        )

    print("-" * 70)
    total = successes + failures
    print(f"Positive separation: {successes}/{total} ({100*successes/total:.0f}%)")
    if failures:
        print(
            f"Failures: {failures}/{total} — the embedding matched the neighbourhood, "
            "not the specialisation",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
