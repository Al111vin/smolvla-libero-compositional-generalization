#!/usr/bin/env python3
"""Diagnostic exact-replay audit for a complete task-15 action trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts import calibrate_libero_36_push as c


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
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
    with np.load(args.attempt, allow_pickle=False) as arrays:
        expected = {key: arrays[key].copy() for key in arrays.files}
    replay = c.replay_attempt(
        row, seed, expected["actions"], "left",
        allow_non_grasping_gripper_motion=True,
    )
    keys = (
        "states", "left_contact", "right_contact", "grasp_proxy",
        "table_support", "unexpected_object_contact", "robot_distractor_contact",
        "nonselected_robot_target_contact", "relation", "xy_in_target",
    )
    checks = {key: bool(np.array_equal(expected[key], replay[key])) for key in keys}
    result = {
        "diagnostic_only": True,
        "actions": int(len(expected["actions"])),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
