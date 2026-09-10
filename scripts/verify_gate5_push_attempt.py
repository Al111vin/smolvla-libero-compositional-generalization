#!/usr/bin/env python3
"""Validate a saved LIBERO-36 Gate-5 push attempt and its replay.

The calibration pilot deliberately never promotes a threshold.  This tool
validates one isolated trajectory using the evidence embedded in its ``.npz``:
the original trace, a fresh-reset replay trace, contact telemetry, target
support, and the terminal goal-hold window.  It does not alter a trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TRACE_KEYS = (
    "actions",
    "states",
    "raw_initial_state",
    "settle_actions",
    "settle_states",
    "settle_target_table_support",
    "settle_source_official",
    "settle_source_xy",
    "left_contact",
    "right_contact",
    "grasp_proxy",
    "nonselected_robot_target_contact",
    "robot_distractor_contact",
    "unexpected_object_contact",
    "table_support",
    "relation",
    "xy_in_target",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--terminal-hold-steps", type=int, default=20)
    return parser.parse_args()


def exact_match(original: np.ndarray, replayed: np.ndarray) -> bool:
    return (
        original.shape == replayed.shape
        and original.dtype == replayed.dtype
        and np.array_equal(original, replayed)
    )


def main() -> None:
    args = parse_args()
    if args.terminal_hold_steps <= 0:
        raise ValueError("terminal hold must be positive")

    with np.load(args.attempt, allow_pickle=False) as trace:
        missing = [
            key
            for key in TRACE_KEYS
            if key not in trace or f"replay_{key}" not in trace
        ]
        if missing:
            raise ValueError(f"attempt is missing replay evidence: {missing}")

        arrays = {key: np.asarray(trace[key]) for key in TRACE_KEYS}
        replay = {
            key: np.asarray(trace[f"replay_{key}"])
            for key in TRACE_KEYS
        }
        metadata = json.loads(str(trace["metadata_json"].item()))
        phases = np.asarray(trace["phase"]).astype(str)

    matches = {
        key: exact_match(arrays[key], replay[key])
        for key in TRACE_KEYS
    }
    hold = args.terminal_hold_steps
    if len(phases) < hold:
        raise ValueError("attempt is shorter than its requested terminal hold")

    candidate = metadata.get("candidate", {})
    selected_side = str(candidate["pusher_finger"])
    selected_contact_key = f"{selected_side}_contact"
    opposite_contact_key = (
        "right_contact" if selected_side == "left" else "left_contact"
    )
    terminal_slice = slice(-hold, None)
    checks = {
        "state_exact": matches["states"],
        "raw_initial_state_exact": matches["raw_initial_state"],
        "actions_exact": matches["actions"],
        "settle_trace_exact": all(
            matches[key]
            for key in (
                "settle_actions",
                "settle_states",
                "settle_target_table_support",
                "settle_source_official",
                "settle_source_xy",
            )
        ),
        "contact_and_goal_trace_exact": all(
            matches[key]
            for key in (
                "left_contact",
                "right_contact",
                "grasp_proxy",
                "nonselected_robot_target_contact",
                "robot_distractor_contact",
                "unexpected_object_contact",
                "table_support",
                "relation",
                "xy_in_target",
            )
        ),
        "selected_pad_contact_observed": bool(
            arrays[selected_contact_key].astype(bool).any()
        ),
        "no_opposite_pad_contact": not bool(
            arrays[opposite_contact_key].astype(bool).any()
        ),
        "no_grasp_proxy": not bool(arrays["grasp_proxy"].astype(bool).any()),
        "no_nonselected_target_contact": not bool(
            arrays["nonselected_robot_target_contact"].astype(bool).any()
        ),
        "no_robot_distractor_contact": not bool(
            arrays["robot_distractor_contact"].astype(bool).any()
        ),
        "no_unexpected_object_contact": not bool(
            arrays["unexpected_object_contact"].astype(bool).any()
        ),
        "support_all": bool(arrays["table_support"].astype(bool).all()),
        "terminal_hold_present": bool(
            np.all(phases[terminal_slice] == "terminal_hold")
        ),
        "terminal_relation_hold": bool(
            arrays["relation"][terminal_slice].astype(bool).all()
        ),
        "terminal_xy_hold": bool(
            arrays["xy_in_target"][terminal_slice].astype(bool).all()
        ),
        "terminal_support_hold": bool(
            arrays["table_support"][terminal_slice].astype(bool).all()
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    summary = {
        "task_id": int(metadata["task_id"]),
        "layout_id": int(metadata["layout_id"]),
        "candidate_index": int(candidate["candidate_index"]),
        "pusher_finger": selected_side,
        "actions": int(len(arrays["actions"])),
        "checks": checks,
        "failed_checks": failed,
        "passed": not failed,
        "state_max_abs_diff": float(
            np.max(np.abs(arrays["states"] - replay["states"]))
        ),
        "attempt": str(args.attempt),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
