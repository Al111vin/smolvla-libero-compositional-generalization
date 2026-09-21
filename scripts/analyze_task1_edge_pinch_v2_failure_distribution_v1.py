#!/usr/bin/env python3
"""Task1 edge-pinch v2 fine-tune -- failure-distribution report for the
checkpoint 2500 / 3000 blind-eval results (batch1: 555101-555105 + batch2:
555201-555220, n=25 per checkpoint after combining).

READ-ONLY: this script only reads existing per-seed result JSON files under
--batch1-dir and --batch2-dir. It never launches an eval, never touches any
checkpoint or dataset file, and never writes anywhere except --output-json.

Produces, per checkpoint and combined:
  - failure count (out of n)
  - termination_reason distribution (all records, and failures-only)
  - steps_run distribution (min/max/mean/median) for failures
  - reward_max / reward_final distribution (min/max/mean/median, count of
    strictly-positive values) for failures -- a nonzero reward_max on a
    "failed" (success=False) rollout would mean partial task progress
    (e.g. object picked up but not placed correctly), which is worth
    knowing even though it didn't cross the success threshold
  - whether any failure terminated early without success (done=True /
    wait_phase_done before max_steps, but success stayed False) --
    distinguished from "ran the full max_steps with no signal at all"
  - the explicit list of failed seeds
  - a checkpoint_2500 vs checkpoint_3000 comparison block

This is intentionally "layer 1" -- summary-field analysis only, no new GPU
run, no per-step trajectory data (the blind-eval script does not record
per-step actions/observations). If this report doesn't give enough signal
to explain WHY rollouts fail, a "layer 2" re-run with finer-grained
per-step logging would be a separate, explicitly-confirmed experiment.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def load_checkpoint_records(root: Path, step: int) -> list[dict[str, Any]]:
    ckpt_dir = root / f"checkpoint_{step:06d}"
    if not ckpt_dir.is_dir():
        return []
    records = []
    for f in sorted(ckpt_dir.glob("seed_*.json")):
        with open(f) as fh:
            records.append(json.load(fh))
    return records


def _dist(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
    }


def analyze_checkpoint(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    successes = [r for r in records if r.get("success") is True]
    failures = [r for r in records if r.get("success") is not True]

    all_termination_reasons = Counter(r.get("termination_reason", "MISSING") for r in records)
    fail_termination_reasons = Counter(r.get("termination_reason", "MISSING") for r in failures)

    fail_steps_run = [r.get("steps_run", 0) for r in failures]
    fail_reward_max = [r.get("reward_max", 0.0) for r in failures]
    fail_reward_final = [r.get("reward_final", 0.0) for r in failures]
    fail_nonzero_reward_max_seeds = sorted(
        r["seed"] for r in failures if r.get("reward_max", 0.0) and r["reward_max"] > 0
    )

    # "early terminated but not successful": done fired (env_done_no_success
    # or wait_phase_done) before max_steps was reached, and success stayed
    # False. Distinct from "ran the full max_steps budget with nothing
    # happening" (termination_reason == max_steps_reached).
    early_terminated_no_success = sorted(
        r["seed"] for r in failures
        if r.get("termination_reason") in ("env_done_no_success", "wait_phase_done")
    )

    return {
        "n_total": total,
        "n_success": len(successes),
        "n_failure": len(failures),
        "failure_rate": round(len(failures) / total, 4) if total else None,
        "termination_reason_distribution_all": dict(all_termination_reasons),
        "termination_reason_distribution_failures_only": dict(fail_termination_reasons),
        "failures_steps_run_distribution": _dist(fail_steps_run),
        "failures_reward_max_distribution": _dist(fail_reward_max),
        "failures_reward_final_distribution": _dist(fail_reward_final),
        "failures_with_nonzero_reward_max_seeds": fail_nonzero_reward_max_seeds,
        "failures_with_nonzero_reward_max_count": len(fail_nonzero_reward_max_seeds),
        "early_terminated_but_not_successful_seeds": early_terminated_no_success,
        "early_terminated_but_not_successful_count": len(early_terminated_no_success),
        "success_seeds": sorted(r["seed"] for r in successes),
        "failure_seeds": sorted(r["seed"] for r in failures),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch1-dir", type=str, required=True)
    p.add_argument("--batch2-dir", type=str, required=True)
    p.add_argument("--checkpoints", type=int, nargs="+", default=[2500, 3000])
    p.add_argument("--output-json", type=str, required=True)
    args = p.parse_args()

    batch1_root = Path(args.batch1_dir)
    batch2_root = Path(args.batch2_dir)

    result: dict[str, Any] = {
        "schema_version": "task1_edge_pinch_v2_failure_distribution_v1",
        "batch1_dir": str(batch1_root),
        "batch2_dir": str(batch2_root),
        "checkpoints_analyzed": args.checkpoints,
        "per_checkpoint": {},
    }

    per_ckpt_summaries = {}
    for step in args.checkpoints:
        b1 = load_checkpoint_records(batch1_root, step)
        b2 = load_checkpoint_records(batch2_root, step)
        combined = b1 + b2
        seeds = {r["seed"] for r in combined}
        if len(seeds) != len(combined):
            result.setdefault("warnings", []).append(
                f"checkpoint {step}: duplicate seed(s) detected across batch1+batch2 combined "
                f"records ({len(combined)} records, {len(seeds)} unique seeds) -- investigate "
                "before trusting this checkpoint's distribution"
            )
        analysis = analyze_checkpoint(combined)
        result["per_checkpoint"][str(step)] = analysis
        per_ckpt_summaries[step] = analysis

    if len(args.checkpoints) >= 2:
        steps_sorted = sorted(per_ckpt_summaries.keys())
        comparisons = []
        for i in range(len(steps_sorted)):
            for j in range(i + 1, len(steps_sorted)):
                a, b = steps_sorted[i], steps_sorted[j]
                sa, sb = per_ckpt_summaries[a], per_ckpt_summaries[b]
                comparisons.append({
                    "checkpoint_a": a,
                    "checkpoint_b": b,
                    "failure_rate_a": sa["failure_rate"],
                    "failure_rate_b": sb["failure_rate"],
                    "termination_reason_distribution_a": sa["termination_reason_distribution_failures_only"],
                    "termination_reason_distribution_b": sb["termination_reason_distribution_failures_only"],
                    "early_terminated_but_not_successful_count_a": sa["early_terminated_but_not_successful_count"],
                    "early_terminated_but_not_successful_count_b": sb["early_terminated_but_not_successful_count"],
                    "nonzero_reward_max_failures_count_a": sa["failures_with_nonzero_reward_max_count"],
                    "nonzero_reward_max_failures_count_b": sb["failures_with_nonzero_reward_max_count"],
                })
        result["checkpoint_comparison"] = comparisons

    result["note"] = (
        "Layer 1 (summary-field) failure analysis only -- no per-step trajectory data available "
        "in the underlying records. Does not itself explain the causal failure mode (e.g. missed "
        "grasp vs. misplaced release); a layer-2 re-run with finer-grained per-step logging would "
        "be a separate, explicitly-confirmed experiment."
    )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"FAILURE_DISTRIBUTION_ANALYSIS_COMPLETE -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
