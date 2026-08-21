import csv
import json
import math
from pathlib import Path

import numpy as np

from scripts import calibrate_libero_36_push as calibrator
from scripts import validate_libero_36_envs as validator


OUTPUT = Path(
    "/tmp/libero36_gate5_cream_tasks27_29_selected"
)
SUMMARY = OUTPUT / "summary.csv"

OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)

layout_rows = validator.read_layout_spec(
    Path("data/libero_36/layout_spec.csv")
)
selected_rows = sorted(
    [
        row
        for row in layout_rows
        if row["task_id"] in {27, 28, 29}
        and row["layout_id"] == 1
    ],
    key=lambda row: row["task_id"],
)

if [row["task_id"] for row in selected_rows] != [27, 28, 29]:
    raise RuntimeError(
        "Expected exactly tasks 27, 28, and 29 on layout 1"
    )


def run_one(row, rim_offset_x_m, close_steps):
    task_id = int(row["task_id"])
    seed = validator.seed_for(
        row,
        calibrator.GATE3_CALIBRATION_RESET_INDEX,
        validator.FULL_SEED_BASE,
    )
    target = calibrator.target_instance(row)
    env = validator.make_environment(
        Path(row["bddl_path"])
    )

    metrics = {
        "actions": 0,
        "grasp_steps": 0,
        "distractor_steps": 0,
        "robot_plate_steps": 0,
        "unexpected_steps": 0,
    }
    relation_history = []
    grasp_events = []
    grasp_phase_steps = {}
    terminal_positions = []
    terminal_quaternions = []

    trace_actions = []
    trace_states = []
    trace_phases = []
    trace_left = []
    trace_right = []
    trace_grasp = []
    trace_relation = []
    trace_done = []
    trace_distractor = []
    trace_robot_plate = []
    trace_unexpected = []
    trace_target_xyz = []
    trace_eef_xyz = []
    initial_state = None

    try:
        settled = calibrator.settle_environment(
            env,
            row,
            seed,
        )
        obs = settled["obs"]
        inner = env.env
        geometry = calibrator.contact_geometry(
            env,
            target,
        )
        plate_geoms = calibrator.collision_geom_names(
            env,
            "plate_1",
        )
        start_target = calibrator.target_xyz(
            obs,
            target,
        ).copy()
        plate_xyz = np.asarray(
            obs["plate_1_pos"],
            dtype=np.float64,
        ).copy()

        initial_state = np.asarray(
            env.get_sim_state(),
            dtype=np.float64,
        ).copy()

        yaw = math.radians(-45.0)
        rotation_z = np.array(
            [
                [
                    math.cos(yaw),
                    -math.sin(yaw),
                    0.0,
                ],
                [
                    math.sin(yaw),
                    math.cos(yaw),
                    0.0,
                ],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        desired_rotation = (
            rotation_z @ settled["desired_rotation"]
        )

        def finger_center():
            pads = calibrator.finger_positions(
                inner,
                geometry,
            )
            return 0.5 * (
                pads["left"] + pads["right"]
            )

        def raw_step(action, phase):
            nonlocal obs

            action = np.asarray(
                action,
                dtype=np.float32,
            )
            if action.shape != (7,):
                raise RuntimeError(
                    f"Invalid action shape in {phase}"
                )
            if not np.isfinite(action).all():
                raise RuntimeError(
                    f"Non-finite action in {phase}"
                )
            if (
                np.any(action < -1.0)
                or np.any(action > 1.0)
            ):
                raise RuntimeError(
                    f"Action outside [-1, 1] in {phase}"
                )

            obs, _, done, _ = env.step(action)
            relation = bool(env.check_success())

            if bool(done) != relation:
                raise RuntimeError(
                    f"done mismatch in {phase}: "
                    f"done={done}, relation={relation}"
                )

            left, right, grasp = (
                calibrator.push_contact_record(
                    env,
                    geometry,
                )
            )

            distractor = any(
                inner.check_contact(
                    geometry["robot"],
                    geometry["distractors"][name],
                )
                for name in validator.MANIPULABLES
                if name != target
            )

            robot_plate = bool(
                inner.check_contact(
                    geometry["robot"],
                    plate_geoms,
                )
            )

            contact_pairs = validator.contact_pairs(
                inner,
                "plate_1",
            )
            unexpected = any(
                set(pair) != {target, "plate_1"}
                for pair in contact_pairs
            )

            metrics["actions"] += 1
            metrics["grasp_steps"] += int(grasp)
            grasp_phase_steps[phase] = (
                grasp_phase_steps.get(phase, 0)
                + int(grasp)
            )
            if grasp:
                grasp_events.append(
                    (metrics["actions"], phase)
                )
            metrics["distractor_steps"] += int(
                distractor
            )
            metrics["robot_plate_steps"] += int(
                robot_plate
            )
            metrics["unexpected_steps"] += int(
                unexpected
            )
            relation_history.append(relation)

            trace_actions.append(action.copy())
            trace_states.append(
                np.asarray(
                    env.get_sim_state(),
                    dtype=np.float64,
                ).copy()
            )
            trace_phases.append(phase)
            trace_left.append(bool(left))
            trace_right.append(bool(right))
            trace_grasp.append(bool(grasp))
            trace_relation.append(bool(relation))
            trace_done.append(bool(done))
            trace_distractor.append(bool(distractor))
            trace_robot_plate.append(bool(robot_plate))
            trace_unexpected.append(bool(unexpected))
            trace_target_xyz.append(
                calibrator.target_xyz(
                    obs,
                    target,
                ).copy()
            )
            trace_eef_xyz.append(
                calibrator.eef_xyz(obs).copy()
            )

            if phase == "terminal_hold":
                poses = validator.capture_poses(
                    obs,
                    "plate_1",
                )
                terminal_positions.append(
                    poses[target][0].copy()
                )
                terminal_quaternions.append(
                    poses[target][1].copy()
                )

        def control_step(
            desired_center,
            gripper_action,
            position_cap,
            phase,
        ):
            desired_eef = (
                calibrator.eef_xyz(obs)
                + desired_center
                - finger_center()
            )
            action = calibrator.osc_action(
                obs,
                desired_eef,
                desired_rotation,
                position_cap,
            )
            action[6] = np.float32(
                gripper_action
            )
            raw_step(action, phase)

        def move_to(
            desired_center,
            gripper_action,
            position_cap,
            phase,
            max_steps,
        ):
            stable_steps = 0
            final_error = math.inf

            for _ in range(max_steps):
                control_step(
                    desired_center,
                    gripper_action,
                    position_cap,
                    phase,
                )
                final_error = float(
                    np.linalg.norm(
                        finger_center()
                        - desired_center
                    )
                )

                if (
                    final_error
                    <= calibrator.POSITION_TOLERANCE_M
                ):
                    stable_steps += 1
                else:
                    stable_steps = 0

                if (
                    stable_steps
                    >= calibrator.STABLE_POSE_STEPS
                ):
                    return True, final_error

            return False, final_error

        # Use the task-0 validated single-rim pinch.
        rim_center = start_target.copy()
        rim_center[0] += rim_offset_x_m
        rim_center[2] += 0.000

        above_rim = rim_center.copy()
        above_rim[2] += 0.130

        approach_ok, approach_error = move_to(
            above_rim,
            -1.0,
            0.35,
            "approach_above_rim",
            140,
        )

        # Preserve the exact successful task-0 descent policy.
        for _ in range(130):
            control_step(
                rim_center,
                -1.0,
                0.16,
                "descend_straddle_rim",
            )

        pinch_center = finger_center().copy()

        for _ in range(close_steps):
            control_step(
                pinch_center,
                +1.0,
                0.14,
                "close_rim_pinch",
            )

        grasp_after_close = bool(
            inner._check_grasp(
                inner.robots[0].gripper,
                geometry["target"],
            )
        )

        # Task 1 grasps immediately after closing.
        # Task 2 requires a short upward seating motion.
        seat_ok = True
        seat_error = 0.0
        grasp_after_seat = grasp_after_close

        if task_id == 2:
            seat_center = pinch_center.copy()
            seat_center[2] += 0.020

            seat_ok, seat_error = move_to(
                seat_center,
                +1.0,
                0.12,
                "seat_grasp",
                60,
            )

            grasp_after_seat = bool(
                inner._check_grasp(
                    inner.robots[0].gripper,
                    geometry["target"],
                )
            )

        grasp_before_transport = bool(
            grasp_after_seat
            if task_id == 2
            else grasp_after_close
        )

        lift_center = pinch_center.copy()
        lift_center[2] += 0.110

        lift_ok, lift_error = move_to(
            lift_center,
            +1.0,
            0.18,
            "lift_grasped",
            120,
        )

        for _ in range(15):
            control_step(
                lift_center,
                +1.0,
                0.14,
                "lift_hold",
            )

        carried_target = calibrator.target_xyz(
            obs,
            target,
        ).copy()
        carry_offset = (
            carried_target - finger_center()
        )

        desired_target_above_plate = np.array(
            [
                plate_xyz[0],
                plate_xyz[1],
                max(
                    carried_target[2],
                    plate_xyz[2] + 0.150,
                ),
            ],
            dtype=np.float64,
        )
        above_plate_center = (
            desired_target_above_plate
            - carry_offset
        )

        carry_ok, carry_error = move_to(
            above_plate_center,
            +1.0,
            0.20,
            "carry_above_plate",
            240,
        )

        for _ in range(20):
            control_step(
                above_plate_center,
                +1.0,
                0.14,
                "carry_hold",
            )

        carried_target = calibrator.target_xyz(
            obs,
            target,
        ).copy()
        carry_offset = (
            carried_target - finger_center()
        )

        desired_target_placed = np.array(
            [
                plate_xyz[0],
                plate_xyz[1],
                plate_xyz[2] + 0.050,
            ],
            dtype=np.float64,
        )
        placed_center = (
            desired_target_placed - carry_offset
        )

        lower_ok, lower_error = move_to(
            placed_center,
            +1.0,
            0.14,
            "lower_onto_plate",
            140,
        )

        release_center = finger_center().copy()

        for _ in range(50):
            control_step(
                release_center,
                -1.0,
                0.12,
                "release_open",
            )

        grasp_released = not bool(
            inner._check_grasp(
                inner.robots[0].gripper,
                geometry["target"],
            )
        )

        retreat_center = finger_center().copy()
        retreat_center[2] += 0.100

        retreat_ok, retreat_error = move_to(
            retreat_center,
            -1.0,
            0.20,
            "retreat_up",
            120,
        )

        for _ in range(
            calibrator.TERMINAL_HOLD_STEPS
        ):
            raw_step(
                np.zeros(7, dtype=np.float32),
                "terminal_hold",
            )

        final_poses = validator.capture_poses(
            obs,
            "plate_1",
        )
        receiver_region_ok = bool(
            calibrator.goal_validator
            .receiver_requested_region_ok(
                inner,
                final_poses,
                row,
                "plate_1",
            )
        )

        positions = np.asarray(
            terminal_positions,
            dtype=np.float64,
        )
        quaternions = np.asarray(
            terminal_quaternions,
            dtype=np.float64,
        )

        if (
            len(positions)
            != calibrator.TERMINAL_HOLD_STEPS
        ):
            raise RuntimeError(
                "Terminal position trace is incomplete"
            )

        terminal_motion = float(
            np.max(
                np.linalg.norm(
                    np.diff(
                        positions[-6:],
                        axis=0,
                    ),
                    axis=1,
                )
            )
        )

        terminal_orientation = max(
            validator.quaternion_distance_deg(
                quaternions[index - 1],
                quaternions[index],
            )
            for index in range(
                len(quaternions) - 5,
                len(quaternions),
            )
        )

        terminal_relation_hold = bool(
            len(relation_history)
            >= calibrator.TERMINAL_HOLD_STEPS
            and all(
                relation_history[
                    -calibrator.TERMINAL_HOLD_STEPS:
                ]
            )
        )

        checks = {
            "approach_reached": bool(
                approach_ok
            ),
            "seat_pose_reached": bool(
                task_id != 2
                or seat_ok
                or seat_error
                <= calibrator.POSITION_TOLERANCE_M
            ),
            "grasp_before_transport": bool(
                grasp_before_transport
            ),
            "lift_reached": bool(lift_ok),
            "carry_reached": bool(carry_ok),
            "lower_pose_reached": bool(
                lower_ok
                or lower_error
                <= calibrator.POSITION_TOLERANCE_M
            ),
            "grasp_released": bool(
                grasp_released
            ),
            "retreat_reached": bool(
                retreat_ok
            ),
            "terminal_relation_hold": bool(
                terminal_relation_hold
            ),
            "receiver_region_preserved": bool(
                receiver_region_ok
            ),
            "terminal_motion_stable": bool(
                terminal_motion
                <= calibrator.MAX_FINAL_MOTION_M
            ),
            "terminal_orientation_stable": bool(
                terminal_orientation
                <= calibrator.MAX_FINAL_ORIENTATION_DEG
            ),
            "no_robot_distractor_contact": bool(
                metrics["distractor_steps"] == 0
            ),
            "no_robot_plate_contact": bool(
                metrics["robot_plate_steps"] == 0
            ),
            "no_unexpected_object_contact": bool(
                metrics["unexpected_steps"] == 0
            ),
            "within_action_horizon": bool(
                metrics["actions"]
                <= calibrator.MAX_ACTIVE_ACTIONS
            ),
        }

        failed_checks = [
            name
            for name, passed in checks.items()
            if not passed
        ]

        trajectory_path = OUTPUT / (
            f"task_{task_id:03d}_layout_1_trajectory.npz"
        )

        np.savez_compressed(
            trajectory_path,
            task_id=np.asarray(
                task_id,
                dtype=np.int64,
            ),
            layout_id=np.asarray(
                1,
                dtype=np.int64,
            ),
            seed=np.asarray(
                seed,
                dtype=np.int64,
            ),
            initial_state=np.asarray(
                initial_state,
                dtype=np.float64,
            ),
            actions=np.asarray(
                trace_actions,
                dtype=np.float32,
            ),
            states=np.asarray(
                trace_states,
                dtype=np.float64,
            ),
            phase=np.asarray(
                trace_phases,
                dtype=np.str_,
            ),
            left_contact=np.asarray(
                trace_left,
                dtype=np.bool_,
            ),
            right_contact=np.asarray(
                trace_right,
                dtype=np.bool_,
            ),
            grasp_proxy=np.asarray(
                trace_grasp,
                dtype=np.bool_,
            ),
            relation=np.asarray(
                trace_relation,
                dtype=np.bool_,
            ),
            done=np.asarray(
                trace_done,
                dtype=np.bool_,
            ),
            robot_distractor_contact=np.asarray(
                trace_distractor,
                dtype=np.bool_,
            ),
            robot_plate_contact=np.asarray(
                trace_robot_plate,
                dtype=np.bool_,
            ),
            unexpected_object_contact=np.asarray(
                trace_unexpected,
                dtype=np.bool_,
            ),
            target_xyz=np.asarray(
                trace_target_xyz,
                dtype=np.float64,
            ),
            eef_xyz=np.asarray(
                trace_eef_xyz,
                dtype=np.float64,
            ),
        )

        return {
            "task_id": task_id,
            "layout_id": 1,
            "spatial_region":
                row["spatial_region"],
            "seed": int(seed),
            "actions": int(metrics["actions"]),
            "trajectory_path": str(
                trajectory_path
            ),
            "grasp_before_transport": bool(
                grasp_before_transport
            ),
            "grasp_steps": int(
                metrics["grasp_steps"]
            ),
            "grasp_after_close_diagnostic": bool(
                grasp_after_close
            ),
            "grasp_after_seat": bool(
                grasp_after_seat
            ),
            "seat_final_error_m": float(
                seat_error
            ),
            "first_grasp_action": (
                int(grasp_events[0][0])
                if grasp_events
                else -1
            ),
            "first_grasp_phase": (
                grasp_events[0][1]
                if grasp_events
                else ""
            ),
            "last_grasp_action": (
                int(grasp_events[-1][0])
                if grasp_events
                else -1
            ),
            "last_grasp_phase": (
                grasp_events[-1][1]
                if grasp_events
                else ""
            ),
            "grasp_phase_steps_json": json.dumps(
                grasp_phase_steps,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "robot_distractor_contact_steps":
                int(metrics["distractor_steps"]),
            "robot_plate_contact_steps":
                int(metrics["robot_plate_steps"]),
            "unexpected_object_contact_steps":
                int(metrics["unexpected_steps"]),
            "approach_final_error_m":
                float(approach_error),
            "lift_final_error_m":
                float(lift_error),
            "carry_final_error_m":
                float(carry_error),
            "lower_final_error_m":
                float(lower_error),
            "retreat_final_error_m":
                float(retreat_error),
            "terminal_motion_m":
                float(terminal_motion),
            "terminal_orientation_deg":
                float(terminal_orientation),
            "failed_checks": json.dumps(
                failed_checks,
                separators=(",", ":"),
            ),
            "passed": bool(
                not failed_checks
            ),
            "error": "",
        }

    except Exception as exc:
        return {
            "task_id": task_id,
            "layout_id": 1,
            "spatial_region":
                row["spatial_region"],
            "seed": int(seed),
            "passed": False,
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }

    finally:
        env.close()


results = []

for row in selected_rows:
    rim_offset_x_m = 0.000
    close_steps = 45

    result = run_one(
        row,
        rim_offset_x_m,
        close_steps,
    )
    result["rim_offset_x_m"] = rim_offset_x_m
    result["close_steps"] = close_steps
    results.append(result)

    print(
        f"task={result['task_id']:02d} "
        f"region={result['spatial_region']} "
        f"actions={result.get('actions')} "
        f"grasp_before_transport="
        f"{result.get('grasp_before_transport')} "
        f"failed={result.get('failed_checks')} "
        f"passed={result['passed']} "
        f"error={result['error']}"
    )

fields = sorted(
    {
        key
        for result in results
        for key in result
    }
)

with SUMMARY.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=fields,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(results)

print("=" * 80)
print(
    "passed:",
    sum(bool(result["passed"]) for result in results),
    "/",
    len(results),
)
print("summary:", SUMMARY)
