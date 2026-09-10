#!/usr/bin/env python3
"""Diagnostic-only re-contact probe from a frozen task-15 trajectory state.

This never writes a Gate-5 artifact.  It is used to measure whether a
geometrically valid north-side contact can produce a table-supported southward
push before that segment is integrated into a replayable policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts import calibrate_libero_36_push as c
from scripts.probe_gate5_pad_entries import yaw_rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--state-index", type=int, required=True)
    parser.add_argument("--replay-prefix-actions", type=int,
                        help="Replay this many actions from --trace before probing."
                        " Requires --state-index to name the resulting state.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--yaw", type=float, default=30.0)
    parser.add_argument("--pad", choices=("left", "right"), default="right")
    parser.add_argument("--x-offset", type=float, default=-0.03)
    parser.add_argument("--guard-y-offset", type=float, default=0.075)
    parser.add_argument("--contact-y-offset", type=float, default=0.04)
    parser.add_argument("--push-x", type=float, default=0.0)
    parser.add_argument("--push-y", type=float, default=-0.3)
    parser.add_argument("--push-z", type=float, default=-0.12)
    parser.add_argument("--push-steps", type=int, default=60)
    parser.add_argument("--gripper-action", type=float, default=1.0,
                        help="Non-grasping jaw command used throughout the probe.")
    parser.add_argument("--contact-gripper-action", type=float,
                        help="Optional jaw command used only for approach and push.")
    parser.add_argument("--settle-steps", type=int, default=0,
                        help="Zero-action settling steps after the diagnostic push.")
    parser.add_argument("--terminal-hold-steps", type=int, default=0,
                        help="Terminal-hold actions after settling (diagnostic only).")
    parser.add_argument("--direct-guard", action="store_true",
                        help="Skip the high transit when the pad is already above obstacles.")
    parser.add_argument("--detour-left", type=float, default=0.0,
                        help="Optional safe leftward pad detour before the guard point.")
    return parser.parse_args()


def task_row() -> dict:
    return next(
        row
        for row in c.reset_validator.read_layout_spec(Path("data/libero_36/layout_spec.csv"))
        if int(row["task_id"]) == 15 and int(row["layout_id"]) == 1
    )


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    row = task_row()
    contact_gripper_action = (
        args.gripper_action
        if args.contact_gripper_action is None else args.contact_gripper_action
    )
    seed = c.reset_validator.seed_for(
        row, c.GATE3_CALIBRATION_RESET_INDEX, c.reset_validator.FULL_SEED_BASE
    )
    target = c.target_instance(row)
    env = c.reset_validator.make_environment(Path(row["bddl_path"]))
    try:
        settled = c.settle_environment(env, row, seed)
        geometry = c.contact_geometry(env, target)
        sim = env.env.sim
        with np.load(args.trace, allow_pickle=False) as arrays:
            if not 0 <= args.state_index < len(arrays["states"]):
                raise IndexError("--state-index is outside trace states")
            trace_states = arrays["states"].copy()
            trace_actions = arrays["actions"].copy()
        recorder = c.AttemptRecorder(
            env, row, target, geometry, "left", settled["stabilized_z"], None,
            allow_non_grasping_gripper_motion=True,
        )
        if args.replay_prefix_actions is None:
            sim.set_state_from_flattened(trace_states[args.state_index])
            sim.forward()
            obs = env.env._get_observations(force_update=True)
        else:
            if args.replay_prefix_actions != args.state_index:
                raise ValueError("--replay-prefix-actions must equal --state-index")
            if not 0 <= args.replay_prefix_actions <= len(trace_actions):
                raise IndexError("--replay-prefix-actions is outside trace actions")
            obs = settled["obs"]
            for action in trace_actions[: args.replay_prefix_actions]:
                obs = recorder.step(obs, action, "replayed_prefix")
            if not np.array_equal(np.asarray(env.get_sim_state()), trace_states[args.state_index]):
                raise RuntimeError("replayed prefix did not reach the requested trace state")
        site = sim.model.site_name2id("gripper0_grip_site")
        rotation = yaw_rotation(args.yaw) @ sim.data.site_xmat[site].reshape(3, 3).copy()

        def unsafe() -> bool:
            return bool(
                recorder.grasp_proxy[-1]
                or recorder.robot_distractor_contact[-1]
                or recorder.unexpected_object_contact[-1]
                or recorder.target_positions[-1][2] - settled["stabilized_z"] > 0.030
            )

        target_xyz = c.target_xyz(obs, target)
        guard = target_xyz + np.array(
            [args.x_offset, args.guard_y_offset, 0.090], dtype=np.float64
        )
        if args.direct_guard:
            lift_ok = transit_ok = True
        else:
            active_pad = c.finger_positions(env.env, geometry)[args.pad].copy()
            active_pad[2] = 1.050
            obs, lift_ok, _ = c.move_pad_to(
                recorder, obs, geometry, args.pad, active_pad, rotation,
                "prefix_lift", 100, 0.50, args.gripper_action,
            )
            high = guard.copy()
            high[2] = 1.050
            obs, transit_ok, _ = c.move_pad_to(
                recorder, obs, geometry, args.pad, high, rotation,
                "prefix_transit", 180, 0.50, args.gripper_action,
            )
        if args.detour_left:
            detour = c.finger_positions(env.env, geometry)[args.pad].copy()
            detour[0] += args.detour_left
            detour[2] = guard[2]
            obs, detour_ok, _ = c.move_pad_to(
                recorder, obs, geometry, args.pad, detour, rotation,
                "prefix_detour", 100, 0.50, args.gripper_action,
            )
            transit_ok = transit_ok and detour_ok
        obs, guard_ok, _ = c.move_pad_to(
            recorder, obs, geometry, args.pad, guard, rotation,
            "prefix_guard", 100, 0.20, args.gripper_action,
        )
        contact = False
        approach_steps = 0
        if lift_ok and transit_ok and guard_ok and not unsafe():
            for approach_steps in range(80):
                target_xyz = c.target_xyz(obs, target)
                desired_pad = target_xyz + np.array(
                    [args.x_offset, args.contact_y_offset, 0.090], dtype=np.float64
                )
                desired_eef = c.desired_eef_for_pad(
                    env, obs, geometry, args.pad, desired_pad, rotation
                )
                obs = recorder.step(
                    obs, c.osc_action(obs, desired_eef, rotation, 0.15, contact_gripper_action),
                    "prefix_approach",
                )
                contact = bool(
                    recorder.left_contact[-1]
                    or recorder.right_contact[-1]
                    or recorder.nonselected_robot_target_contact[-1]
                )
                if contact or unsafe():
                    break
        push_steps = 0
        if contact and not unsafe():
            for push_steps in range(args.push_steps):
                action = c.osc_action(obs, None, rotation, 0.08, contact_gripper_action)
                action[:3] = (args.push_x, args.push_y, args.push_z)
                obs = recorder.step(obs, action, "prefix_push")
                if unsafe():
                    break
        settle_steps = 0
        if not unsafe():
            for settle_steps in range(args.settle_steps):
                obs = recorder.step(
                    obs, np.zeros(7, dtype=np.float32), "prefix_settle"
                )
                if unsafe():
                    break
        terminal_hold_steps = 0
        if not unsafe():
            for terminal_hold_steps in range(args.terminal_hold_steps):
                obs = recorder.step(
                    obs, np.zeros(7, dtype=np.float32), "terminal_hold"
                )
                if unsafe():
                    break
        positions = np.asarray(recorder.target_positions, dtype=np.float64)
        summary = {
            "diagnostic_only": True,
            "lift_ok": bool(lift_ok),
            "transit_ok": bool(transit_ok),
            "guard_ok": bool(guard_ok),
            "contact": contact,
            "approach_steps": int(approach_steps),
            "push_steps": int(push_steps),
            "settle_steps": int(settle_steps),
            "terminal_hold_steps": int(terminal_hold_steps),
            "unsafe": bool(unsafe()) if recorder.target_positions else False,
            "terminal_target_xyz": positions[-1].tolist() if len(positions) else None,
            "max_lift_m": float(max(0.0, positions[:, 2].max() - settled["stabilized_z"]))
            if len(positions) else 0.0,
            "unexpected_object_contact": bool(any(recorder.unexpected_object_contact)),
            "robot_distractor_contact": bool(any(recorder.robot_distractor_contact)),
            "grasp_proxy": bool(any(recorder.grasp_proxy)),
        }
        trajectory = args.output.with_suffix(".npz")
        c.write_npz_atomic(trajectory, **recorder.arrays())
        summary["trajectory"] = str(trajectory)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()
