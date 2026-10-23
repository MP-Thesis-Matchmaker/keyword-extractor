#!/usr/bin/env python3
"""Paired significance tests for the student-background ablation.

Compares each arm against topic_only over the targets all four arms share:
an exact McNemar test (two-sided binomial on the discordant pairs) for Hit@1
and Hit@5, and a paired sign-flip randomisation test plus a bootstrap 95%
interval for the difference in reciprocal rank (a miss counts as 0).

Usage:
    python scripts/analyze_ablation_significance.py data/eval_ablation_v4.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ARMS = ("plus_program", "plus_background", "full")


def load(path: str) -> dict[str, dict[tuple, int | None]]:
    ranks_by_arm: dict[str, dict[tuple, int | None]] = defaultdict(dict)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qs = row.get("query_style", "")
        if not qs.startswith("ablation:"):
            continue
        arm = qs.split(":", 1)[1]
        # A target is a source document with its gold authors, shared by all arms
        key = (tuple(row["source_ids"]), tuple(sorted(row["gold"])))
        if key in ranks_by_arm[arm]:
            raise ValueError(f"duplicate target in arm {arm}")
        ranks_by_arm[arm][key] = row["rank"]
    return ranks_by_arm


def reciprocal(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def hit(rank: int | None, k: int) -> int:
    return int(rank is not None and rank <= k)


def mcnemar_exact(lost: int, gained: int) -> float:
    n = lost + gained
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(lost, gained) + 1)) / 2**n
    return min(1.0, 2 * tail)


def sign_flip(diffs: list[float], n_perm: int = 200_000, seed: int = 7) -> float:
    rng = random.Random(seed)
    observed = abs(sum(diffs))
    nonzero = [d for d in diffs if d != 0]
    extreme = sum(
        abs(sum(d if rng.random() < 0.5 else -d for d in nonzero)) >= observed - 1e-12
        for _ in range(n_perm)
    )
    return (extreme + 1) / (n_perm + 1)


def bootstrap_ci(diffs: list[float], n_boot: int = 20_000, seed: int = 11) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="per-row evaluate output JSONL (--out from evaluate)")
    args = parser.parse_args()

    ranks_by_arm = load(args.input)
    if "topic_only" not in ranks_by_arm:
        print("ERROR: no topic_only rows found in the input file.")
        return 1

    base = ranks_by_arm["topic_only"]
    targets = sorted(base)
    if any(set(ranks_by_arm.get(arm, {})) != set(targets) for arm in ARMS):
        print("ERROR: the arms do not share the same targets.")
        return 1

    n = len(targets)
    print(f"{n} shared targets\n")
    print(f"{'arm':<16} {'metric':<6} {'base':>6} {'arm':>6} {'diff':>7}  {'detail':<20} {'p':>6}")
    for arm in ARMS:
        other = ranks_by_arm[arm]
        for k in (1, 5):
            b = [hit(base[t], k) for t in targets]
            a = [hit(other[t], k) for t in targets]
            lost = sum(x == 1 and y == 0 for x, y in zip(b, a, strict=True))
            gained = sum(x == 0 and y == 1 for x, y in zip(b, a, strict=True))
            print(
                f"{arm:<16} Hit@{k:<2} {sum(b) / n:>6.3f} {sum(a) / n:>6.3f} "
                f"{(sum(a) - sum(b)) / n:>+7.3f}  lost {lost:>2} / gained {gained:>2}  "
                f"{mcnemar_exact(lost, gained):>6.3f}"
            )
        diffs = [reciprocal(other[t]) - reciprocal(base[t]) for t in targets]
        lo, hi = bootstrap_ci(diffs)
        print(
            f"{arm:<16} {'MRR':<6} {sum(reciprocal(base[t]) for t in targets) / n:>6.3f} "
            f"{sum(reciprocal(other[t]) for t in targets) / n:>6.3f} {sum(diffs) / n:>+7.3f}  "
            f"95% CI [{lo:+.3f}, {hi:+.3f}]  {sign_flip(diffs):>6.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
