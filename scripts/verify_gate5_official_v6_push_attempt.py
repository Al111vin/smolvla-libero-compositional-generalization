#!/usr/bin/env python3
"""Verify one strict, replayed LIBERO-36 Gate-5 v6 push attempt.

The verifier implements the written Gate-5 requirements and deliberately
reports (but does not fail on) the v5 one-finger controller diagnostics.  It
never simulates, mutates, or promotes an attempt.
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
    "table_support",
    "unexpected_object_contact",
    "robot_distractor_contact",
    "nonselected_robot_target_contact",
    "robot_target_contact_pairs",
    "relation",
    "xy_in_target",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--terminal-hold-steps", type=int, default=20)
    parser.add_argument("--max-lift-m", type=float, default=0.03)
    parser.add_argument("--max-final-step-motion-m", type=float, default=0.001)
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
    if args.max_lift_m < 0.0 or args.max_final_step_motion_m < 0.0:
        raise ValueError("thresholds must be non-negative")

    with np.load(args.attempt, allow_pickle=False) as trace:
        missing = [
            key
            for key in (*TRACE_KEYS, "phase", "target_xyz", "stabilized_target_z_m")
            if key not in trace or (key in TRACE_KEYS and f"replay_{key}" not in trace)
        ]
        if missing:
            raise ValueError(f"attempt is missing v6 replay evidence: {missing}")
        arrays = {key: np.asarray(trace[key]) for key in TRACE_KEYS}
        replay = {key: np.asarray(trace[f"replay_{key}"]) for key in TRACE_KEYS}
        phases = np.asarray(trace["phase"]).astype(str)
        target_xyz = np.asarray(trace["target_xyz"], dtype=np.float64)
        baseline_z = float(np.asarray(trace["stabilized_target_z_m"]).reshape(-1)[0])
        metadata = json.loads(str(trace["metadata_json"].item()))
        diagnostics = {
            "nonselected_robot_target_contact_steps": int(
                np.asarray(trace["nonselected_robot_target_contact"], dtype=bool).sum()
            ) if "nonselected_robot_target_contact" in trace else None,
            "opposite_pad_contact_steps": int(
                np.asarray(trace["right_contact"], dtype=bool).sum()
                if metadata["candidate"]["pusher_finger"] == "left"
                else np.asarray(trace["left_contact"], dtype=bool).sum()
            ),
            "actions": int(len(arrays["actions"])),
        }

    matches = {key: exact_match(arrays[key], replay[key]) for key in TRACE_KEYS}
    hold = args.terminal_hold_steps
    if len(phases) < hold or len(target_xyz) < hold:
        raise ValueError("attempt is shorter than its requested terminal hold")
    terminal = slice(-hold, None)
    push_mask = phases == "push"
    final_xyz = target_xyz[-min(5, len(target_xyz)):]
    final_step_motion = (
        float(np.max(np.linalg.norm(np.diff(final_xyz, axis=0), axis=1)))
        if len(final_xyz) > 1
        else float("inf")
    )
    max_lift = float(max(0.0, np.max(target_xyz[:, 2]) - baseline_z))
    checks = {
        "state_exact": matches["states"],
        "raw_initial_state_exact": matches["raw_initial_state"],
        "actions_exact": matches["actions"],
        "settle_trace_exact": all(
            matches[key]
            for key in (
                "settle_actions", "settle_states", "settle_target_table_support",
                "settle_source_official", "settle_source_xy",
            )
        ),
        "contact_support_and_goal_trace_exact": all(
            matches[key]
            for key in (
                "left_contact", "right_contact", "grasp_proxy", "table_support",
                "unexpected_object_contact", "robot_distractor_contact", "relation",
                "xy_in_target",
            )
        ),
        "push_phase_present": bool(push_mask.any()),
        "robot_target_contact_during_push": bool(
            push_mask.any()
            and (
                arrays["left_contact"][push_mask].astype(bool).any()
                or arrays["right_contact"][push_mask].astype(bool).any()
                or np.asarray(
                    arrays.get(
                        "nonselected_robot_target_contact",
                        np.zeros(len(push_mask), dtype=bool),
                    ),
                    dtype=bool,
                )[push_mask].any()
            )
        ),
        "no_grasp_proxy": not bool(arrays["grasp_proxy"].astype(bool).any()),
        "no_unexpected_object_contact": not bool(
            arrays["unexpected_object_contact"].astype(bool).any()
        ),
        "no_robot_distractor_contact": not bool(
            arrays["robot_distractor_contact"].astype(bool).any()
        ),
        "v5_controller_action_cap_disabled": (
            metadata["candidate"].get("active_action_limit", "v5-default")
            is None
        ),
        "table_support_all": bool(arrays["table_support"].astype(bool).all()),
        "max_lift_within_0_030_m": max_lift <= args.max_lift_m,
        "terminal_hold_present": bool(np.all(phases[terminal] == "terminal_hold")),
        "terminal_relation_hold": bool(arrays["relation"][terminal].astype(bool).all()),
        "terminal_xy_hold": bool(arrays["xy_in_target"][terminal].astype(bool).all()),
        "terminal_table_support_hold": bool(
            arrays["table_support"][terminal].astype(bool).all()
        ),
        "terminal_target_motion_within_0_001_m": (
            final_step_motion <= args.max_final_step_motion_m
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    summary = {
        "protocol_version": "libero_36_gate5_official_v6",
        "task_id": int(metadata["task_id"]),
        "layout_id": int(metadata["layout_id"]),
        "candidate_index": int(metadata["candidate"]["candidate_index"]),
        "checks": checks,
        "diagnostics": diagnostics,
        "max_target_lift_m": max_lift,
        "max_final_step_motion_m": final_step_motion,
        "failed_checks": failed,
        "passed": not failed,
        "attempt": str(args.attempt),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
