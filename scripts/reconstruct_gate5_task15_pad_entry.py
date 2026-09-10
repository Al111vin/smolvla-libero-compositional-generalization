#!/usr/bin/env python3
"""Task-15 push reconstruction using a measured left-fingerpad entry.

The generic corridor controller changes the final approach after selecting an
entry pose.  This runner retains the independently probed high-then-descend
entry verbatim, then uses target-relative pad feedback for the three physical
route segments.  Results are isolated under the supplied output directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scripts import calibrate_libero_36_push as c


TASK_ID = 15
SIDE = "left"
ENTRY_YAW_DEGREES = -120.0
ENTRY_PAD_HEIGHT_M = 0.090
# A 75 mm guarded pose is collision-free at the controller's stable
# intermediate jaw opening.  The 45 mm point is the physical cup-wall pose.
ENTRY_STANDOFF_M = 0.045
ENTRY_GUARD_STANDOFF_M = 0.075
ENTRY_HIGH_CLEARANCE_M = 0.000
ENTRY_TRANSIT_Z_M = 1.050
GRIPPER_ACTION = 1.0
TRANSIT_CLEARANCE_M = 0.000
ENTRY_DIRECTION = np.array([-1.0, -0.02], dtype=np.float64)
MAX_LIFT_M = 0.030
PUSH_PENETRATION_M = 0.040
PUSH_POSITION_CAP = 0.080
PUSH_ACTION_MAGNITUDE = 0.150
RECONTACT_POSITION_CAP = 0.120
MAX_RECONTACT_STEPS = 80
PUSH_VERTICAL_BIAS_FIRST_M = -0.020
PUSH_VERTICAL_BIAS_LATER_M = -0.025
CONTACT_REAR_STANDOFF_M = 0.050
ROUTE = np.array(
    # The bowl occupies the lower central lane.  First move into the empty
    # upper lane, then cross left, and only then descend into the left target.
    [[-0.070, 0.130], [-0.100, -0.205], [-0.120, -0.210]],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--push-iterations", type=int, default=650)
    parser.add_argument("--push-action-magnitude", type=float, default=PUSH_ACTION_MAGNITUDE)
    parser.add_argument("--push-first-vertical-bias", type=float,
                        default=PUSH_VERTICAL_BIAS_FIRST_M)
    parser.add_argument("--push-later-vertical-bias", type=float,
                        default=PUSH_VERTICAL_BIAS_LATER_M)
    parser.add_argument("--first-waypoint-y", type=float)
    parser.add_argument("--first-waypoint-x", type=float)
    parser.add_argument("--second-waypoint-y", type=float)
    parser.add_argument("--second-waypoint-x", type=float)
    parser.add_argument(
        "--hold-at-first-waypoint", action="store_true",
        help="Diagnostic: stop routing after the first safe-lane waypoint.",
    )
    return parser.parse_args()


def yaw_rotation(degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def row_for_task() -> dict:
    rows = c.reset_validator.read_layout_spec(Path("data/libero_36/layout_spec.csv"))
    return next(
        row
        for row in rows
        if int(row["task_id"]) == TASK_ID and int(row["layout_id"]) == 1
    )


def robot_target_contact(recorder: c.AttemptRecorder) -> bool:
    """Gate-5 permits any non-grasping robot contact that performs the push."""
    return bool(
        recorder.left_contact[-1]
        or recorder.right_contact[-1]
        or recorder.nonselected_robot_target_contact[-1]
    )


def geometry_position(env, geometry_name: str) -> np.ndarray:
    geometry_id = env.env.sim.model.geom_name2id(geometry_name)
    return env.env.sim.data.geom_xpos[geometry_id].copy()


def unsafe(recorder: c.AttemptRecorder, stabilized_z: float) -> str | None:
    if recorder.grasp_proxy[-1]:
        return "grasp_proxy"
    if recorder.robot_distractor_contact[-1]:
        return "robot_distractor_contact"
    if recorder.unexpected_object_contact[-1]:
        return "unexpected_object_contact"
    if recorder.target_positions[-1][2] - stabilized_z > MAX_LIFT_M:
        return "target_lift_over_0_030_m"
    return None


def descend_to_pad_contact(
    recorder: c.AttemptRecorder,
    obs,
    geometry: dict,
    desired_pad: np.ndarray,
    desired_rotation: np.ndarray,
    stabilized_z: float,
) -> tuple[object, str | None]:
    for _ in range(80):
        desired_eef = c.desired_eef_for_pad(
            recorder.env, obs, geometry, SIDE, desired_pad, desired_rotation
        )
        action = c.osc_action(
            obs, desired_eef, desired_rotation, 0.20, GRIPPER_ACTION
        )
        obs = recorder.step(obs, action, "entry_descend")
        if robot_target_contact(recorder):
            return obs, None
        reason = unsafe(recorder, stabilized_z)
        if reason:
            return obs, reason
    return obs, "entry_descent_step_limit"


def append_settle_evidence(arrays: dict, settled: dict, prefix: str = "") -> None:
    arrays[f"{prefix}raw_initial_state"] = settled["initial_state"]
    arrays[f"{prefix}settle_actions"] = settled["settle_actions"]
    arrays[f"{prefix}settle_states"] = settled["settle_states"]
    arrays[f"{prefix}settle_target_table_support"] = settled[
        "settle_target_table_support"
    ]
    arrays[f"{prefix}settle_source_official"] = settled[
        "settle_source_official"
    ]
    arrays[f"{prefix}settle_source_xy"] = settled["settle_source_xy"]


def main() -> None:
    args = parse_args()
    if args.push_iterations <= 0 or args.push_iterations > 800:
        raise ValueError("push iterations must be in 1..800")
    if not 0.0 < args.push_action_magnitude <= 1.0:
        raise ValueError("push action magnitude must be in (0, 1]")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    route = ROUTE.copy()
    if args.first_waypoint_y is not None:
        route[0, 1] = args.first_waypoint_y
    if args.first_waypoint_x is not None:
        route[0, 0] = args.first_waypoint_x
    if args.second_waypoint_y is not None:
        route[1, 1] = args.second_waypoint_y
    if args.second_waypoint_x is not None:
        route[1, 0] = args.second_waypoint_x
    if args.hold_at_first_waypoint:
        route = route[:1]
    row = row_for_task()
    seed = c.reset_validator.seed_for(
        row, c.GATE3_CALIBRATION_RESET_INDEX, c.reset_validator.FULL_SEED_BASE
    )
    target = c.target_instance(row)
    env = c.reset_validator.make_environment(Path(row["bddl_path"]))
    previous_rotation_mode = c.ROTATION_AWARE_PAD_TARGET
    c.ROTATION_AWARE_PAD_TARGET = True
    failure: str | None = None
    try:
        settled = c.settle_environment(env, row, seed)
        obs = settled["obs"]
        geometry = c.contact_geometry(env, target)
        desired_rotation = yaw_rotation(ENTRY_YAW_DEGREES) @ settled["desired_rotation"]
        target_start = c.target_xyz(obs, target)
        first_direction = ENTRY_DIRECTION.copy()
        first_direction /= np.linalg.norm(first_direction)
        entry_pad = np.array(
            [
                target_start[0] - first_direction[0] * ENTRY_STANDOFF_M,
                target_start[1] - first_direction[1] * ENTRY_STANDOFF_M,
                target_start[2] + ENTRY_PAD_HEIGHT_M,
            ],
            dtype=np.float64,
        )
        recorder = c.AttemptRecorder(
            env, row, target, geometry, SIDE, settled["stabilized_z"], None,
            allow_non_grasping_gripper_motion=True,
        )
        entry_guard = entry_pad.copy()
        entry_guard[:2] = target_start[:2] - first_direction * ENTRY_GUARD_STANDOFF_M
        entry_high = entry_guard.copy()
        entry_high[2] += ENTRY_HIGH_CLEARANCE_M
        current_pad = c.finger_positions(env.env, geometry)[SIDE]
        transit_z = ENTRY_TRANSIT_Z_M
        transit_lift = current_pad.copy()
        transit_lift[2] = transit_z
        obs, lift_ok, _ = c.move_pad_to(
            recorder,
            obs,
            geometry,
            SIDE,
            transit_lift,
            desired_rotation,
            "entry_transit_lift",
            100,
            0.50,
            GRIPPER_ACTION,
        )
        transit_high = entry_high.copy()
        transit_high[2] = transit_z
        if lift_ok:
            obs, transit_ok, transit_diagnostics = c.move_pad_to(
                recorder,
                obs,
                geometry,
                SIDE,
                transit_high,
                desired_rotation,
                "entry_transit_high",
                180,
                0.50,
                GRIPPER_ACTION,
            )
            if (
                not transit_ok
                and transit_diagnostics["final_position_error_m"] <= 0.010
            ):
                transit_ok = True
        else:
            transit_ok = False
        if transit_ok:
            obs, high_ok, _ = c.move_pad_to(
                recorder,
                obs,
                geometry,
                SIDE,
                entry_high,
                desired_rotation,
                "entry_high",
                100,
                0.20,
                GRIPPER_ACTION,
            )
        else:
            high_ok = False
        if not high_ok:
            failure = "entry_high_unreachable"
        else:
            # The high pose is the guarded 75 mm point.  The contact routine
            # then advances only along the measured target normal.
            obs, failure = descend_to_pad_contact(
                recorder,
                obs,
                geometry,
                entry_pad,
                desired_rotation,
                settled["stabilized_z"],
            )
        if failure is None:
            contact_proxy = "gripper0_hand_collision"
            segment = 0
            consecutive_contact_loss = 0
            for _ in range(args.push_iterations):
                live_target = c.target_xyz(obs, target)
                relation, xy_ok = c.target_region_status(env, obs, row)
                if relation and xy_ok:
                    break
                while (
                    segment + 1 < len(route)
                    and np.linalg.norm(live_target[:2] - route[segment]) <= 0.018
                ):
                    segment += 1
                delta = route[segment] - live_target[:2]
                distance = float(np.linalg.norm(delta))
                if distance <= 1e-6:
                    break
                direction = delta / distance
                vertical_bias = (
                    args.push_first_vertical_bias
                    if segment == 0 else args.push_later_vertical_bias
                )
                # A route turn cannot be pushed from the old side of the
                # cup.  Move the active contact geometry to the back of the
                # requested segment before applying an impulse.
                desired_contact = live_target.copy()
                desired_contact[:2] -= direction * CONTACT_REAR_STANDOFF_M
                desired_contact[2] += vertical_bias
                current_contact = geometry_position(env, contact_proxy)
                # The measured right-side contact has sufficient tangential
                # authority for every planar route segment.  Retaining it
                # avoids the lift induced by sweeping the hand around the cup.
                reposition_needed = False
                if robot_target_contact(recorder) and not reposition_needed:
                    action = c.osc_action(
                        obs, None, desired_rotation, PUSH_POSITION_CAP, GRIPPER_ACTION
                    )
                    action[:2] = direction * args.push_action_magnitude
                    action[2] = vertical_bias / 0.05
                else:
                    desired_eef = c.eef_xyz(obs) + (
                        desired_contact - current_contact
                    )
                    action = c.osc_action(
                        obs,
                        desired_eef,
                        desired_rotation,
                        RECONTACT_POSITION_CAP,
                        GRIPPER_ACTION,
                    )
                obs = recorder.step(obs, action, "push")
                reason = unsafe(recorder, settled["stabilized_z"])
                if reason:
                    failure = reason
                    break
                if robot_target_contact(recorder):
                    consecutive_contact_loss = 0
                else:
                    consecutive_contact_loss += 1
                    if consecutive_contact_loss > MAX_RECONTACT_STEPS:
                        failure = "robot_target_contact_lost_over_80_steps"
                        break
        for _ in range(c.TERMINAL_HOLD_STEPS):
            obs = recorder.step(obs, np.zeros(7, dtype=np.float32), "terminal_hold")
        arrays = recorder.arrays()
    finally:
        c.ROTATION_AWARE_PAD_TARGET = previous_rotation_mode
        env.close()

    replay = c.replay_attempt(
        row, seed, arrays["actions"], SIDE,
        allow_non_grasping_gripper_motion=True,
    )
    combined = dict(arrays)
    append_settle_evidence(combined, settled)
    combined["stabilized_target_z_m"] = np.asarray(
        [settled["stabilized_z"]], dtype=np.float64
    )
    replay_arrays = {}
    append_settle_evidence(replay_arrays, replay["settled"], "replay_")
    for key, value in replay.items():
        if key != "settled":
            replay_arrays[f"replay_{key}"] = value
    combined.update(replay_arrays)
    candidate = {
        "candidate_index": 0,
        "pusher_finger": SIDE,
        "contact_height_offset_m": ENTRY_PAD_HEIGHT_M,
        "desired_yaw_degrees": ENTRY_YAW_DEGREES,
        "active_action_limit": None,
        "candidate_order_key": "task15_safe_non_grasp_hand_push",
        "contact_proxy": "gripper0_hand_collision",
    }
    combined["metadata_json"] = np.asarray(
        [
            c.json_compact(
                {
                    "protocol": "libero_36_gate5_official_v6_task15_safe_hand_push",
                    "task_id": TASK_ID,
                    "layout_id": 1,
                    "candidate": candidate,
                    "seed": seed,
                    "diagnostic_only": True,
                }
            )
        ]
    )
    attempt_path = root / "trajectory.npz"
    c.write_npz_atomic(attempt_path, **combined)
    summary = {
        "task_id": TASK_ID,
        "attempt": str(attempt_path),
        "actions": int(len(arrays["actions"])),
        "failure": failure,
        "terminal_target_xyz": arrays["target_xyz"][-1].tolist(),
        "robot_target_contact_steps": int(
            arrays["left_contact"].sum()
            + arrays["right_contact"].sum()
            + arrays["nonselected_robot_target_contact"].sum()
        ),
        "max_lift_m": float(
            max(0.0, arrays["target_xyz"][:, 2].max() - settled["stabilized_z"])
        ),
        "replay_state_exact": bool(np.array_equal(arrays["states"], replay["states"])),
    }
    (root / "runner_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
