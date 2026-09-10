#!/usr/bin/env python3
"""Find a reachable, single-fingerpad entry for a remaining Gate-5 push.

This is an isolated geometry probe.  It never pushes the target or writes
formal Gate-5 evidence.  Every candidate starts from the validated frozen
reset and records whether the *selected fingerpad* (rather than a finger body
or palm) reaches the target without lifting it or contacting a distractor.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scripts import calibrate_libero_36_push as calibrator
from scripts.gate5_contact_direction import lateral_push_contacts


ENTRY_DIRECTIONS = {
    15: np.array([-1.0, -0.02], dtype=np.float64),
    16: np.array([-1.0, -0.32], dtype=np.float64),
    17: np.array([-0.72, 0.70], dtype=np.float64),
}
YAW_DEGREES = (-135.0, -105.0, -75.0, -45.0)
PAD_HEIGHTS_M = (0.000, 0.010)
STANDOFF_M = 0.025
HIGH_CLEARANCE_M = 0.100
# The frozen reset already starts the selected pad high over the work surface;
# adding more height reaches the OSC workspace boundary during the lateral leg.
TRANSIT_CLEARANCE_M = 0.000
MAX_LIFT_M = 0.030
MAX_ENTRY_HORIZONTAL_DRIFT_M = 0.005
TRANSIT_POSITION_TOLERANCE_M = 0.010


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, choices=sorted(ENTRY_DIRECTIONS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, help="Save diagnostic states for later geometric inspection.")
    parser.add_argument("--side", choices=("left", "right"))
    parser.add_argument("--yaw-degrees", type=float)
    parser.add_argument(
        "--pitch-degrees",
        type=float,
        default=0.0,
        help="World-frame pitch applied after the yaw for hand-clearance probing.",
    )
    parser.add_argument(
        "--entry-direction",
        type=float,
        nargs=2,
        metavar=("DX", "DY"),
        help="Override the target-relative entry direction for an angled approach.",
    )
    parser.add_argument("--pad-height-m", type=float)
    parser.add_argument("--high-clearance-m", type=float, default=HIGH_CLEARANCE_M,
                        help="Extra height above guarded pad waypoints (diagnostic only).")
    parser.add_argument("--transit-z", type=float, help="Optional absolute transit height; diagnostic only.")
    parser.add_argument("--transit-tolerance-m", type=float, default=TRANSIT_POSITION_TOLERANCE_M)
    parser.add_argument(
        "--gripper-action",
        type=float,
        default=-1.0,
        help="Gripper command used only while descending to the diagnostic contact.",
    )
    parser.add_argument(
        "--contact-standoff-m",
        type=float,
        default=STANDOFF_M,
        help="Target-relative lateral standoff for the intended fingerpad contact.",
    )
    parser.add_argument(
        "--guard-standoff-m",
        type=float,
        help=(
            "Use a guarded entry: descend at this farther target-relative "
            "standoff, then advance horizontally to the normal contact pose."
        ),
    )
    return parser.parse_args()


def yaw_rotation(degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def pitch_rotation(degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return np.array(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ],
        dtype=np.float64,
    )


def selected_contact(recorder: calibrator.AttemptRecorder, side: str) -> bool:
    return bool(
        recorder.left_contact[-1]
        if side == "left"
        else recorder.right_contact[-1]
    )


def descend_until_contact(
    recorder: calibrator.AttemptRecorder,
    obs,
    geometry: dict,
    side: str,
    desired_pad: np.ndarray,
    desired_rotation: np.ndarray,
    stabilized_z: float,
    gripper_action: float,
    phase: str = "entry_descend",
) -> tuple[object, bool, str]:
    """Descend toward one pad waypoint, stopping at first decisive event."""
    for _ in range(80):
        desired_eef = calibrator.desired_eef_for_pad(
            recorder.env,
            obs,
            geometry,
            side,
            desired_pad,
            desired_rotation,
        )
        action = calibrator.osc_action(obs, desired_eef, desired_rotation, 0.20)
        action[6] = gripper_action
        obs = recorder.step(obs, action, phase)
        if recorder.grasp_proxy[-1]:
            return obs, False, "grasp_proxy"
        if recorder.robot_distractor_contact[-1]:
            return obs, False, "robot_distractor_contact"
        if recorder.unexpected_object_contact[-1]:
            return obs, False, "unexpected_object_contact"
        if (
            recorder.target_positions[-1][2] - stabilized_z
            > MAX_LIFT_M
        ):
            return obs, False, "target_lift_over_0_030_m"
        if selected_contact(recorder, side):
            target_position = calibrator.target_xyz(obs, recorder.target)
            approach = target_position[:2] - desired_pad[:2]
            if np.linalg.norm(approach) <= 1e-9:
                return obs, False, "undefined_lateral_approach_direction"
            contacts = lateral_push_contacts(
                recorder.env.env.sim, geometry["target"], geometry[side], approach
            )
            if not any(record["lateral_aligned"] for record in contacts):
                return obs, False, "selected_pad_wrong_contact_normal"
            return obs, True, "selected_fingerpad_contact"
    return obs, False, "descent_step_limit"


def task_row(task_id: int) -> dict:
    rows = calibrator.reset_validator.read_layout_spec(
        Path("data/libero_36/layout_spec.csv")
    )
    return next(
        row
        for row in rows
        if int(row["task_id"]) == task_id and int(row["layout_id"]) == 1
    )


def probe_one(
    row: dict,
    side: str,
    yaw_degrees: float,
    pitch_degrees: float,
    pad_height_m: float,
    high_clearance_m: float,
    contact_standoff_m: float,
    guard_standoff_m: float | None,
    direction_override: tuple[float, float] | None,
    gripper_action: float,
    transit_z_override: float | None,
    transit_tolerance_m: float,
    trace_dir: Path | None,
) -> dict:
    seed = calibrator.reset_validator.seed_for(
        row,
        calibrator.GATE3_CALIBRATION_RESET_INDEX,
        calibrator.reset_validator.FULL_SEED_BASE,
    )
    target = calibrator.target_instance(row)
    env = calibrator.reset_validator.make_environment(Path(row["bddl_path"]))
    previous_rotation_mode = calibrator.ROTATION_AWARE_PAD_TARGET
    calibrator.ROTATION_AWARE_PAD_TARGET = True
    try:
        settled = calibrator.settle_environment(env, row, seed)
        obs = settled["obs"]
        geometry = calibrator.contact_geometry(env, target)
        desired_rotation = (
            yaw_rotation(yaw_degrees)
            @ pitch_rotation(pitch_degrees)
            @ settled["desired_rotation"]
        )
        direction = (
            np.asarray(direction_override, dtype=np.float64)
            if direction_override is not None
            else ENTRY_DIRECTIONS[int(row["task_id"])].copy()
        )
        direction /= np.linalg.norm(direction)
        target_start = calibrator.target_xyz(obs, target)
        desired_pad = np.array(
            [
                target_start[0] - direction[0] * contact_standoff_m,
                target_start[1] - direction[1] * contact_standoff_m,
                target_start[2] + pad_height_m,
            ],
            dtype=np.float64,
        )
        recorder = calibrator.AttemptRecorder(
            env,
            row,
            target,
            geometry,
            side,
            settled["stabilized_z"],
            active_action_limit=calibrator.MAX_ACTIVE_ACTIONS,
            allow_non_grasping_gripper_motion=(gripper_action != -1.0),
        )
        high_pad = desired_pad.copy()
        high_pad[2] += high_clearance_m
        if guard_standoff_m is not None:
            if guard_standoff_m <= contact_standoff_m:
                raise ValueError("guard standoff must exceed the contact standoff")
            guard_pad = desired_pad.copy()
            guard_pad[:2] = target_start[:2] - direction * guard_standoff_m
            high_pad = guard_pad.copy()
            high_pad[2] += high_clearance_m
        # The reset's fingers can start higher than ``high_pad``.  Going
        # directly to it then cuts diagonally down through the target.  Lift
        # at or above both poses first, translate only at that height, and only then
        # make the vertical approach.
        current_pad = calibrator.finger_positions(env.env, geometry)[side]
        transit_z = max(float(current_pad[2]), float(high_pad[2])) + TRANSIT_CLEARANCE_M
        if transit_z_override is not None:
            if transit_z_override < high_pad[2]:
                raise ValueError("Transit height must clear the high approach waypoint")
            transit_z = transit_z_override
        transit_lift = current_pad.copy()
        transit_lift[2] = transit_z
        obs, lift_ok, lift_diagnostics = calibrator.move_pad_to(
            recorder,
            obs,
            geometry,
            side,
            transit_lift,
            desired_rotation,
            "entry_transit_lift",
            100,
            0.50,
            gripper_action,
        )
        transit_high = high_pad.copy()
        transit_high[2] = transit_z
        if lift_ok:
            obs, transit_ok, transit_diagnostics = calibrator.move_pad_to(
                recorder,
                obs,
                geometry,
                side,
                transit_high,
                desired_rotation,
                "entry_transit_high",
                180,
                0.50,
                gripper_action,
            )
            transit_high_strict = transit_ok
            # This is a collision-free transit waypoint, not an interaction
            # pose.  A 10 mm residual is adequate to begin the vertical leg
            # and avoids treating an OSC limit cycle at 6--10 mm as a
            # geometry failure.
            if (
                not transit_ok
                and transit_diagnostics["final_position_error_m"]
                <= transit_tolerance_m
            ):
                transit_ok = True
        else:
            transit_ok = False
            transit_high_strict = False
            transit_diagnostics = {"final_position_error_m": float("inf")}
        if transit_ok:
            obs, high_ok, high_diagnostics = calibrator.move_pad_to(
                recorder,
                obs,
                geometry,
                side,
                high_pad,
                desired_rotation,
                "entry_high",
                100,
                0.20,
                gripper_action,
            )
        else:
            high_ok = False
            high_diagnostics = {"final_position_error_m": float("inf")}
        high_strict_ok = high_ok
        if (not high_ok and transit_ok
                and high_diagnostics["final_position_error_m"] <= transit_tolerance_m
                and not any(recorder.robot_target_contact_pairs[i] != "[]"
                            for i in range(len(recorder.actions)))
                and not any(recorder.robot_distractor_contact)):
            # This gate controls a free-space waypoint only. The low contact
            # pose and every physical acceptance check remain independent.
            high_ok = True
        if high_ok:
            if guard_standoff_m is None:
                obs, contact, stop_reason = descend_until_contact(
                    recorder,
                    obs,
                    geometry,
                    side,
                    desired_pad,
                    desired_rotation,
                    settled["stabilized_z"],
                    gripper_action,
                )
            else:
                obs, guard_ok, _ = calibrator.move_pad_to(
                    recorder,
                    obs,
                    geometry,
                    side,
                    guard_pad,
                    desired_rotation,
                    "entry_guard_descend",
                    100,
                    0.20,
                    gripper_action,
                )
                if not guard_ok:
                    contact = False
                    stop_reason = "entry_guard_descend_unreachable"
                else:
                    obs, contact, stop_reason = descend_until_contact(
                        recorder,
                        obs,
                        geometry,
                        side,
                        desired_pad,
                        desired_rotation,
                        settled["stabilized_z"],
                        gripper_action,
                        "entry_guard_advance",
                    )
        else:
            contact = False
            stop_reason = "entry_high_unreachable"
        hold_safe = False
        lateral_hold_steps = 0
        if contact:
            hold_safe = True
            for _ in range(calibrator.TERMINAL_HOLD_STEPS):
                obs = recorder.step(
                    obs, np.zeros(7, dtype=np.float32), "terminal_hold"
                )
                hold_normals = lateral_push_contacts(
                    env.env.sim, geometry["target"], geometry[side], direction
                )
                if any(record["lateral_aligned"] for record in hold_normals):
                    lateral_hold_steps += 1
                reason = (
                    "grasp_proxy" if recorder.grasp_proxy[-1]
                    else "robot_distractor_contact"
                    if recorder.robot_distractor_contact[-1]
                    else "unexpected_object_contact"
                    if recorder.unexpected_object_contact[-1]
                    else "target_lift_over_0_030_m"
                    if recorder.target_positions[-1][2] - settled["stabilized_z"]
                    > MAX_LIFT_M
                    else None
                )
                if reason:
                    hold_safe = False
                    stop_reason = f"entry_contact_hold_{reason}"
                    break
        arrays = recorder.arrays()
        trace_path = None
        if trace_dir is not None:
            trace_path = trace_dir / f"{side}_yaw{yaw_degrees}_height{pad_height_m}.npz"
            trace_arrays = dict(arrays)
            trace_arrays["metadata_json"] = np.asarray(json.dumps(dict(
                diagnostic_only=True, task_id=int(row["task_id"]),
                layout_id=int(row["layout_id"]), seed=seed,
                pusher_finger=side)))
            trace_arrays["desired_rotation"] = desired_rotation
            trace_arrays["desired_pad_xyz"] = desired_pad
            with trace_path.open("xb") as trace_file:
                np.savez_compressed(trace_file, **trace_arrays)
        final_pad = calibrator.finger_positions(env.env, geometry)[side].copy()
        final_target = calibrator.target_xyz(obs, target).copy()
        normals = lateral_push_contacts(env.env.sim, geometry["target"], geometry[side], direction)
        lateral_ok = any(record["lateral_aligned"] for record in normals)
        if hold_safe and lateral_hold_steps != calibrator.TERMINAL_HOLD_STEPS:
            hold_safe = False
            stop_reason = "no_sustained_lateral_selected_pad_contact"
        opposite_key = "right_contact" if side == "left" else "left_contact"
        for key in ("grasp_proxy", "robot_distractor_contact", "unexpected_object_contact",
                    "nonselected_robot_target_contact", opposite_key):
            if hold_safe and arrays[key].any():
                hold_safe = False
                stop_reason = "entry_history_" + key
        if hold_safe and not arrays["table_support"].all():
            hold_safe = False
            stop_reason = "entry_history_table_support_lost"
        if (hold_safe and np.max(arrays["target_xyz"][:, 2])
                - settled["stabilized_z"] > MAX_LIFT_M):
            hold_safe = False
            stop_reason = "entry_history_lift_limit"
        horizontal_drift = float(
            np.max(
                np.linalg.norm(
                    arrays["target_xyz"][:, :2] - target_start[:2], axis=1
                )
            )
        ) if len(arrays["target_xyz"]) else float("inf")
        if hold_safe and horizontal_drift > MAX_ENTRY_HORIZONTAL_DRIFT_M:
            hold_safe = False
            stop_reason = "entry_horizontal_drift_over_0_005_m"
        sim = env.env.sim
        joint_diagnostics = []
        for joint_id in range(sim.model.njnt):
            name = sim.model.joint_id2name(joint_id)
            if not name or not name.startswith("robot0_") or not sim.model.jnt_limited[joint_id]:
                continue
            q = float(sim.data.qpos[sim.model.jnt_qposadr[joint_id]])
            low, high = sim.model.jnt_range[joint_id]
            joint_diagnostics.append(dict(name=name, q=q, lower=float(low), upper=float(high),
                                          margin=float(min(q-low, high-q))))
        rotation_error = float(np.linalg.norm(calibrator.transform.quat2axisangle(
            calibrator.transform.mat2quat(desired_rotation @ calibrator.eef_rotation(obs).T))))
        all_contacts = []
        for i in range(sim.data.ncon):
            con = sim.data.contact[i]
            names = [sim.model.geom_id2name(int(g)) for g in (con.geom1, con.geom2)]
            if any(n and (n.startswith("robot0_") or n.startswith("gripper0_")) for n in names):
                all_contacts.append(dict(names=names, distance=float(con.dist)))
        return {
            "trace_path": str(trace_path) if trace_path else None,
            "joint_limits": joint_diagnostics,
            "final_rotation_error_rad": rotation_error,
            "all_robot_contacts": all_contacts,
            "side": side,
            "yaw_degrees": yaw_degrees,
            "pitch_degrees": pitch_degrees,
            "pad_height_m": pad_height_m,
            "contact_standoff_m": contact_standoff_m,
            "entry_direction": direction.tolist(),
            "gripper_action": gripper_action,
            "guard_standoff_m": guard_standoff_m,
            "high_reached": high_ok,
            "high_strict_reached": high_strict_ok,
            "high_final_pad_error_m": high_diagnostics["final_position_error_m"],
            "transit_lift_reached": lift_ok,
            "transit_lift_final_pad_error_m": lift_diagnostics[
                "final_position_error_m"
            ],
            "transit_high_reached": transit_ok,
            "transit_high_strict_reached": transit_high_strict,
            "transit_high_final_pad_error_m": transit_diagnostics[
                "final_position_error_m"
            ],
            "selected_fingerpad_contact": contact,
            "entry_contact_hold_safe": hold_safe,
            "stop_reason": stop_reason,
            "actions": int(len(arrays["actions"])),
            "selected_contact_steps": int(
                arrays[f"{side}_contact"].sum()
            ),
            "opposite_contact_steps": int(
                arrays["right_contact" if side == "left" else "left_contact"].sum()
            ),
            "nonselected_contact_steps": int(
                arrays["nonselected_robot_target_contact"].sum()
            ),
            "max_lift_m": float(
                max(
                    0.0,
                    arrays["target_xyz"][:, 2].max()
                    - settled["stabilized_z"],
                )
            ) if len(arrays["target_xyz"]) else None,
            "max_entry_horizontal_drift_m": horizontal_drift,
            "robot_target_contact_pairs": sorted(
                set(arrays["robot_target_contact_pairs"].astype(str)) - {"[]"}
            ),
            "desired_pad_xyz": desired_pad.tolist(),
            "selected_pad_contact_normals": normals,
            "lateral_selected_pad_contact": lateral_ok,
            "lateral_hold_steps": lateral_hold_steps,
            "transit_z_m": transit_z,
            "transit_tolerance_m": transit_tolerance_m,
            "final_pad_xyz": final_pad.tolist(),
            "final_pad_error_m": float(np.linalg.norm(final_pad - desired_pad)),
            "final_target_xyz": final_target.tolist(),
        }
    finally:
        calibrator.ROTATION_AWARE_PAD_TARGET = previous_rotation_mode
        env.close()


def main() -> None:
    args = parse_args()
    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=False)
    row = task_row(args.task_id)
    sides = (args.side,) if args.side else ("left", "right")
    yaws = (args.yaw_degrees,) if args.yaw_degrees is not None else YAW_DEGREES
    heights = (
        (args.pad_height_m,)
        if args.pad_height_m is not None
        else PAD_HEIGHTS_M
    )
    records = [
        probe_one(
            row,
            side,
            yaw,
            args.pitch_degrees,
            height,
            args.high_clearance_m,
            args.contact_standoff_m,
            args.guard_standoff_m,
            tuple(args.entry_direction) if args.entry_direction else None,
            args.gripper_action,
            args.transit_z,
            args.transit_tolerance_m,
            args.trace_dir,
        )
        for side in sides
        for yaw in yaws
        for height in heights
    ]
    records.sort(
        key=lambda item: (
            not item["entry_contact_hold_safe"],
            not item["selected_fingerpad_contact"],
            item["max_lift_m"] if item["max_lift_m"] is not None else float("inf"),
            item["actions"],
        )
    )
    report = {
        "diagnostic_only": True,
        "task_id": args.task_id,
        "frozen_layout_id": 1,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
