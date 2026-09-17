#!/usr/bin/env python3
"""Combine batch1 (555101-555105, 6 checkpoints) and batch2_extended
(555201-555220, checkpoints 2500/3000 only) blind-eval results for Task1
edge-pinch v2, and compute the pre-registered decision statistics the user
asked for before this extended run was launched:

  - per-checkpoint (2500, 3000) success rate on the NEW 20-seed batch2 data
    alone
  - a 95% Wilson score confidence interval for that batch2-only rate
    (Wilson rather than a normal approximation because n=20 and observed
    proportions may be small/extreme, where the normal approximation can
    produce out-of-[0,1] or misleadingly narrow intervals)
  - the COMBINED success rate for each checkpoint merging batch1 (n=5) and
    batch2 (n=20) -> n=25, with its own Wilson 95% CI
  - an explicit checkpoint-regression check: whether the 2500 vs 3000
    combined-rate 95% CIs overlap, as a simple, pre-declared (not
    post-hoc-cherry-picked) signal for "is the 3000 regression distinguishable
    from noise given this sample size"

This script only READS existing result JSON files; it never launches an
eval, deletes anything, or writes into either result directory -- output
goes to a separate --output-json path.

Does NOT itself decide whether Fold02 unlocks -- that remains an explicit
user decision per this project's standing rules. This script only produces
the pre-registered numbers the user said they'd use to make that decision.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def wilson_ci(successes: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """95% Wilson score interval. Returns (lower, upper), both in [0, 1].
    n=0 returns (0.0, 1.0) (undefined / maximally uninformative) rather than
    raising, so a missing checkpoint's absence is visible in the output
    instead of crashing the whole report."""
    if n == 0:
        return (0.0, 1.0)
    phat = successes / n
    denom = 1 + (z * z) / n
    center = phat + (z * z) / (2 * n)
    half = z * math.sqrt((phat * (1 - phat) / n) + (z * z) / (4 * n * n))
    lower = (center - half) / denom
    upper = (center + half) / denom
    return (max(0.0, lower), min(1.0, upper))


def load_checkpoint_records(root: Path, step: int) -> list[dict[str, Any]]:
    ckpt_dir = root / f"checkpoint_{step:06d}"
    if not ckpt_dir.is_dir():
        return []
    records = []
    for f in sorted(ckpt_dir.glob("seed_*.json")):
        with open(f) as fh:
            records.append(json.load(fh))
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    successes = sum(1 for r in records if r.get("success") is True)
    seeds = sorted(r["seed"] for r in records)
    success_seeds = sorted(r["seed"] for r in records if r.get("success") is True)
    lower, upper = wilson_ci(successes, n)
    return {
        "n": n,
        "successes": successes,
        "success_rate": (successes / n) if n else None,
        "wilson_95ci_lower": round(lower, 4),
        "wilson_95ci_upper": round(upper, 4),
        "seeds": seeds,
        "success_seeds": success_seeds,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch1-dir", type=str, required=True,
                    help="output_dir of the original 6x5 run (555101-555105)")
    p.add_argument("--batch2-dir", type=str, required=True,
                    help="output_dir of the extended 2x20 run (555201-555220)")
    p.add_argument("--checkpoints", type=int, nargs="+", default=[2500, 3000])
    p.add_argument("--output-json", type=str, required=True)
    args = p.parse_args()

    batch1_root = Path(args.batch1_dir)
    batch2_root = Path(args.batch2_dir)

    result: dict[str, Any] = {
        "schema_version": "task1_edge_pinch_v2_extended_blind_eval_analysis_v1",
        "batch1_dir": str(batch1_root),
        "batch2_dir": str(batch2_root),
        "checkpoints_analyzed": args.checkpoints,
        "per_checkpoint": {},
    }

    combined_rates: dict[int, tuple[float, float]] = {}

    for step in args.checkpoints:
        b1_records = load_checkpoint_records(batch1_root, step)
        b2_records = load_checkpoint_records(batch2_root, step)

        b1_seeds = {r["seed"] for r in b1_records}
        b2_seeds = {r["seed"] for r in b2_records}
        overlap = b1_seeds & b2_seeds
        if overlap:
            result.setdefault("warnings", []).append(
                f"checkpoint {step}: seed overlap between batch1 and batch2: {sorted(overlap)} "
                "-- combined count may double-count; investigate before trusting combined stats"
            )

        combined_records = b1_records + b2_records

        entry = {
            "batch1_only": summarize(b1_records),
            "batch2_only": summarize(b2_records),
            "combined": summarize(combined_records),
        }
        result["per_checkpoint"][str(step)] = entry
        combined_rates[step] = (
            entry["combined"]["wilson_95ci_lower"],
            entry["combined"]["wilson_95ci_upper"],
        )

    # Pre-declared checkpoint-regression check: compare every pair of
    # analyzed checkpoints' combined-sample Wilson CIs for overlap.
    regression_checks = []
    steps_sorted = sorted(combined_rates.keys())
    for i in range(len(steps_sorted)):
        for j in range(i + 1, len(steps_sorted)):
            a, b = steps_sorted[i], steps_sorted[j]
            lo_a, hi_a = combined_rates[a]
            lo_b, hi_b = combined_rates[b]
            overlap = not (hi_a < lo_b or hi_b < lo_a)
            regression_checks.append({
                "checkpoint_a": a,
                "checkpoint_b": b,
                "combined_ci_a": [lo_a, hi_a],
                "combined_ci_b": [lo_b, hi_b],
                "cis_overlap": overlap,
                "interpretation": (
                    "CIs overlap: difference between these two checkpoints' combined success "
                    "rates is NOT clearly distinguishable from sampling noise at this sample size"
                    if overlap else
                    "CIs do NOT overlap: difference between these two checkpoints' combined "
                    "success rates is unlikely to be pure sampling noise at this sample size"
                ),
            })
    result["checkpoint_regression_checks"] = regression_checks
    result["note"] = (
        "This script reports pre-registered statistics only. It does not conclude "
        "whether Fold02 unlocks or whether this fine-tune 'passes' strict closed-loop "
        "evaluation -- those remain explicit user decisions per this project's standing rules."
    )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"ANALYSIS_COMPLETE -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
