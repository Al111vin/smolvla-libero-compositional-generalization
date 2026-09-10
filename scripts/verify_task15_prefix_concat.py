#!/usr/bin/env python3
"""Diagnostic replay audit for a task-15 prefix plus continuation trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts import calibrate_libero_36_push as c


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--prefix-actions", type=int, required=True)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    row = next(
        item
        for item in c.reset_validator.read_layout_spec(Path("data/libero_36/layout_spec.csv"))
        if int(item["task_id"]) == 15 and int(item["layout_id"]) == 1
    )
    seed = c.reset_validator.seed_for(
        row, c.GATE3_CALIBRATION_RESET_INDEX, c.reset_validator.FULL_SEED_BASE
    )
    with np.load(args.prefix, allow_pickle=False) as prefix, np.load(
        args.continuation, allow_pickle=False
    ) as continuation:
        actions = np.concatenate(
            [prefix["actions"][: args.prefix_actions], continuation["actions"]]
        )
        handoff_exact = bool(
            np.array_equal(
                prefix["states"][args.prefix_actions], continuation["states"][0]
            )
        )
        replay = c.replay_attempt(
            row, seed, actions, "left", allow_non_grasping_gripper_motion=True
        )
        prefix_exact = bool(
            np.array_equal(
                replay["states"][: args.prefix_actions + 1],
                prefix["states"][: args.prefix_actions + 1],
            )
        )
        continuation_exact = bool(
            np.array_equal(
                replay["states"][args.prefix_actions :], continuation["states"]
            )
        )
        continuation_max_abs_error = float(
            np.max(
                np.abs(
                    replay["states"][args.prefix_actions :] - continuation["states"]
                )
            )
        )
    result = {
        "diagnostic_only": True,
        "actions": int(len(actions)),
        "handoff_state_exact": handoff_exact,
        "prefix_state_exact": prefix_exact,
        "continuation_state_exact": continuation_exact,
        "continuation_max_abs_state_error": continuation_max_abs_error,
        "passed": handoff_exact and prefix_exact and continuation_exact,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
