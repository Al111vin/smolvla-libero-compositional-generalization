import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from scripts import calibrate_libero_36_push as calibrator
from scripts import validate_libero_36_envs as validator


ROOT = Path(
    "/tmp/libero36_gate5_task8_leg2_xp001"
)
TRACE_DIR = ROOT / "traces"
SUMMARY = ROOT / "summary.csv"

LEG1_TRACE = Path(
    "/tmp/"
    "libero36_gate5_task8_leg1_fast_i00200_h048/"
    "traces/"
    "candidate_01_yaw_+85.0_left_height_+0.048.npz"
)

TASK_ID = 8
LAYOUT_ID = 1

DIRECTION = np.array(
    [0.0, 1.0],
    dtype=np.float64,
)

STANDOFF_M = 0.060
PAD_HEIGHT_M = 0.043
PUSH_INCREMENT_M = 0.00225
PUSH_WAYPOINT_STEPS = 2
TERMINAL_HOLD_STEPS = 20

SETTINGS = (
    (-8.0, "left"),
)


def rotation_z(degrees):
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


def selected_contact(recorder, side):
    values = (
        recorder.left_contact
        if side == "left"
        else recorder.right_contact
    )
    return bool(values[-1])


def opposite_contact(recorder, side):
    values = (
        recorder.right_contact
        if side == "left"
        else recorder.left_contact
    )
    return bool(values[-1])


def staged_move(
    recorder,
    obs,
    geometry,
    side,
    desired_pad,
    desired_rotation,
):
    current_pad = calibrator.finger_positions(
        recorder.env.env,
        geometry,
    )[side]

    current_rotation = calibrator.eef_rotation(
        obs
    )

    # First-leg motion is -X. The selected pad ends on the
    # +X side of the bowl, so retreat in +X before lifting.
    escape_pad = current_pad.copy()
    escape_pad[:2] += np.array(
        [0.020, 0.0],
        dtype=np.float64,
    )

    obs, escape_ok, _ = calibrator.move_pad_to(
        recorder,
        obs,
        geometry,
        side,
        escape_pad,
        current_rotation,
        "transit_detach_lateral",
        calibrator.MAX_TRANSIT_STEPS,
        0.08,
    )

    if not escape_ok:
        return obs, False

    current_pad = calibrator.finger_positions(
        recorder.env.env,
        geometry,
    )[side]

    detach_high = current_pad.copy()
    detach_high[2] += 0.030

    obs, detach_ok, _ = calibrator.move_pad_to(
        recorder,
        obs,
        geometry,
        side,
        detach_high,
        current_rotation,
        "transit_detach_lift",
        calibrator.MAX_TRANSIT_STEPS,
        0.10,
    )

    if not detach_ok:
        return obs, False

    current_pad = calibrator.finger_positions(
        recorder.env.env,
        geometry,
    )[side]

    home_high = np.array(
        [
            current_pad[0],
            current_pad[1],
            desired_pad[2] + 0.12,
        ],
        dtype=np.float64,
    )

    obs, lift_ok, _ = calibrator.move_pad_to(
        recorder,
        obs,
        geometry,
        side,
        home_high,
        current_rotation,
        "transit_vertical_lift",
        calibrator.MAX_TRANSIT_STEPS,
        0.30,
    )

    if not lift_ok:
        return obs, False

    current_pad = calibrator.finger_positions(
        recorder.env.env,
        geometry,
    )[side]

    obs, rotate_ok, _ = calibrator.move_pad_to(
        recorder,
        obs,
        geometry,
        side,
        current_pad,
        desired_rotation,
        "transit_rotate_high",
        calibrator.MAX_TRANSIT_STEPS,
        0.20,
    )

    if not rotate_ok:
        return obs, False

    current_pad = calibrator.finger_positions(
        recorder.env.env,
        geometry,
    )[side]

    align_x = np.array(
        [
            desired_pad[0],
            current_pad[1],
            current_pad[2],
        ],
        dtype=np.float64,
    )

    obs, x_ok, _ = calibrator.move_pad_to(
        recorder,
        obs,
        geometry,
        side,
        align_x,
        desired_rotation,
        "precontact_align_x",
        calibrator.MAX_TRANSIT_STEPS,
        0.40,
    )

    if not x_ok:
        return obs, False

    current_pad = calibrator.finger_positions(
        recorder.env.env,
        geometry,
    )[side]

    align_y = np.array(
        [
            desired_pad[0],
            desired_pad[1],
            current_pad[2],
        ],
        dtype=np.float64,
    )

    obs, y_ok, _ = calibrator.move_pad_to(
        recorder,
        obs,
        geometry,
        side,
        align_y,
        desired_rotation,
        "precontact_align_y",
        calibrator.MAX_TRANSIT_STEPS,
        0.40,
    )

    if not y_ok:
        return obs, False

    obs, behind_ok, _ = calibrator.move_pad_to(
        recorder,
        obs,
        geometry,
        side,
        desired_pad,
        desired_rotation,
        "descend_behind",
        calibrator.MAX_TRANSIT_STEPS,
        0.20,
    )

    return obs, bool(behind_ok)


def run_trial(row, seed, yaw_degrees, side):
    target = calibrator.target_instance(row)

    env = validator.make_environment(
        Path(row["bddl_path"])
    )

    recorder = None

    try:
        settled = calibrator.settle_environment(
            env,
            row,
            seed,
        )
        obs = settled["obs"]

        geometry = calibrator.contact_geometry(
            env,
            target,
        )

        recorder = calibrator.AttemptRecorder(
            env,
            row,
            target,
            geometry,
            side,
            settled["stabilized_z"],
        )

        leg1 = np.load(LEG1_TRACE)

        for action in leg1["actions"]:
            obs = recorder.step(
                obs,
                np.asarray(
                    action,
                    dtype=np.float32,
                ),
                "leg1_replay",
            )

        replay_difference = float(
            np.max(
                np.abs(
                    np.asarray(
                        env.get_sim_state(),
                        dtype=np.float64,
                    )
                    - np.asarray(
                        leg1["states"][-1],
                        dtype=np.float64,
                    )
                )
            )
        )

        if (
            replay_difference
            > calibrator.STATE_REPLAY_ATOL
        ):
            raise RuntimeError(
                "Leg-1 replay mismatch: "
                f"{replay_difference}"
            )

        start_target = calibrator.target_xyz(
            obs,
            target,
        )

        desired_rotation = (
            rotation_z(yaw_degrees)
            @ settled["desired_rotation"]
        )

        behind_pad = np.array(
            [
                start_target[0],
                start_target[1] - STANDOFF_M,
                start_target[2] + PAD_HEIGHT_M,
            ],
            dtype=np.float64,
        )

        obs, behind_ok = staged_move(
            recorder,
            obs,
            geometry,
            side,
            behind_pad,
            desired_rotation,
        )

        contact_step = None
        contact_pad = behind_pad.copy()

        search_steps = int(
            math.ceil(
                calibrator.MAX_CONTACT_SEARCH_M
                / calibrator.CONTACT_SEARCH_INCREMENT_M
            )
        )

        if behind_ok:
            for _ in range(search_steps):
                contact_pad[:2] += (
                    DIRECTION
                    * calibrator.CONTACT_SEARCH_INCREMENT_M
                )

                obs, _, _ = calibrator.move_pad_to(
                    recorder,
                    obs,
                    geometry,
                    side,
                    contact_pad,
                    desired_rotation,
                    "contact_search",
                    4,
                    0.12,
                )

                if (
                    selected_contact(recorder, side)
                    and not opposite_contact(
                        recorder,
                        side,
                    )
                ):
                    contact_step = len(
                        recorder.actions
                    )
                    break

                if (
                    opposite_contact(recorder, side)
                    or recorder.grasp_proxy[-1]
                ):
                    break

        reached = False

        if contact_step is not None:
            for push_index in range(93):
                if push_index == 76:
                    contact_pad[0] += (
                        0.001
                    )

                contact_pad[:2] += (
                    DIRECTION * PUSH_INCREMENT_M
                )

                obs, _, _ = calibrator.move_pad_to(
                    recorder,
                    obs,
                    geometry,
                    side,
                    contact_pad,
                    desired_rotation,
                    "leg2_push",
                    PUSH_WAYPOINT_STEPS,
                    0.14,
                )

                relation, xy_ok = (
                    calibrator.target_region_status(
                        env,
                        obs,
                        row,
                    )
                )

                if relation and xy_ok:
                    reached = True
                    break

                if recorder.grasp_proxy[-1]:
                    break

        if reached:
            zero_action = np.zeros(
                7,
                dtype=np.float32,
            )

            for _ in range(
                TERMINAL_HOLD_STEPS
            ):
                obs = recorder.step(
                    obs,
                    zero_action,
                    "terminal_hold",
                )

        arrays = recorder.arrays()

        end_target = (
            arrays["target_xyz"][-1]
            if arrays["target_xyz"].size
            else start_target
        )

        selected_values = (
            arrays["left_contact"]
            if side == "left"
            else arrays["right_contact"]
        )

        opposite_values = (
            arrays["right_contact"]
            if side == "left"
            else arrays["left_contact"]
        )

        target_z = arrays["target_xyz"][:, 2]

        max_lift = float(
            max(
                0.0,
                np.max(target_z)
                - settled["stabilized_z"],
            )
        )

        hold_relation = bool(
            reached
            and len(arrays["relation"])
            >= TERMINAL_HOLD_STEPS
            and arrays["relation"][
                -TERMINAL_HOLD_STEPS:
            ].all()
        )

        hold_xy = bool(
            reached
            and len(arrays["xy_in_target"])
            >= TERMINAL_HOLD_STEPS
            and arrays["xy_in_target"][
                -TERMINAL_HOLD_STEPS:
            ].all()
        )

        hold_support = bool(
            reached
            and len(arrays["table_support"])
            >= TERMINAL_HOLD_STEPS
            and arrays["table_support"][
                -TERMINAL_HOLD_STEPS:
            ].all()
        )

        checks = {
            "leg1_replay_exact":
                replay_difference
                <= calibrator.STATE_REPLAY_ATOL,
            "behind_pose_reached":
                bool(behind_ok),
            "one_finger_contact_acquired":
                contact_step is not None,
            "target_region_reached":
                bool(reached),
            "terminal_relation_hold":
                hold_relation,
            "terminal_xy_hold":
                hold_xy,
            "terminal_support_hold":
                hold_support,
            "table_support_all":
                bool(arrays["table_support"].all()),
            "opposite_pad_never_contacts":
                bool(not opposite_values.any()),
            "no_grasp":
                bool(not arrays["grasp_proxy"].any()),
            "no_unexpected_object_contacts":
                bool(
                    not arrays[
                        "unexpected_object_contact"
                    ].any()
                ),
            "no_robot_distractor_contacts":
                bool(
                    not arrays[
                        "robot_distractor_contact"
                    ].any()
                ),
            "selected_fingerpad_only_target_contact":
                bool(
                    not arrays[
                        "nonselected_robot_target_contact"
                    ].any()
                ),
            "provisional_lift_check":
                max_lift
                <= calibrator.MAX_LIFT_PROVISIONAL_M,
        }

        failed = [
            name
            for name, passed in checks.items()
            if not passed
        ]

        result = {
            "task_id": TASK_ID,
            "layout_id": LAYOUT_ID,
            "yaw_degrees": yaw_degrees,
            "finger": side,
            "active_actions": len(
                arrays["actions"]
            ),
            "prefix_actions": len(
                leg1["actions"]
            ),
            "prefix_replay_state_max_abs_diff":
                replay_difference,
            "contact_step": (
                ""
                if contact_step is None
                else contact_step
            ),
            "start_leg2_x_m":
                float(start_target[0]),
            "start_leg2_y_m":
                float(start_target[1]),
            "end_x_m": float(end_target[0]),
            "end_y_m": float(end_target[1]),
            "max_lift_m": max_lift,
            "selected_contact_steps": int(
                selected_values.sum()
            ),
            "opposite_contact_steps": int(
                opposite_values.sum()
            ),
            "nonselected_contact_steps": int(
                arrays[
                    "nonselected_robot_target_contact"
                ].sum()
            ),
            "unexpected_object_contact_steps": int(
                arrays[
                    "unexpected_object_contact"
                ].sum()
            ),
            "robot_distractor_contact_steps": int(
                arrays[
                    "robot_distractor_contact"
                ].sum()
            ),
            "support_failure_steps": int(
                np.sum(~arrays["table_support"])
            ),
            "terminal_relation": bool(
                arrays["relation"][-1]
            ),
            "terminal_xy": bool(
                arrays["xy_in_target"][-1]
            ),
            "phase_counts_json": json.dumps(
                dict(
                    Counter(
                        arrays["phase"].astype(str)
                    )
                ),
                sort_keys=True,
                separators=(",", ":"),
            ),
            "failed_checks_json": json.dumps(
                failed,
                separators=(",", ":"),
            ),
            "success": not failed,
            "error": "",
        }

        return result, arrays

    except Exception as error:
        arrays = (
            recorder.arrays()
            if recorder is not None
            else calibrator.empty_attempt_arrays()
        )

        result = {
            "task_id": TASK_ID,
            "layout_id": LAYOUT_ID,
            "yaw_degrees": yaw_degrees,
            "finger": side,
            "active_actions": len(
                arrays["actions"]
            ),
            "failed_checks_json":
                '["exception"]',
            "success": False,
            "error":
                f"{type(error).__name__}: {error}",
        }

        return result, arrays

    finally:
        env.close()


if not LEG1_TRACE.is_file():
    raise FileNotFoundError(LEG1_TRACE)

rows = validator.read_layout_spec(
    Path("data/libero_36/layout_spec.csv")
)

matches = [
    row
    for row in rows
    if int(row["task_id"]) == TASK_ID
    and int(row["layout_id"]) == LAYOUT_ID
]

if len(matches) != 1:
    raise RuntimeError(
        f"Expected one Task 8 row, "
        f"found {len(matches)}"
    )

row = matches[0]

seed = validator.seed_for(
    row,
    calibrator.GATE3_CALIBRATION_RESET_INDEX,
    validator.FULL_SEED_BASE,
)

ROOT.mkdir(parents=True, exist_ok=True)
TRACE_DIR.mkdir(parents=True, exist_ok=True)

results = []

for index, (
    yaw_degrees,
    side,
) in enumerate(
    SETTINGS,
    start=1,
):
    result, arrays = run_trial(
        row,
        seed,
        yaw_degrees,
        side,
    )

    results.append(result)

    trace_path = TRACE_DIR / (
        f"candidate_{index:02d}_"
        f"yaw_{yaw_degrees:+05.1f}_"
        f"{side}.npz"
    )

    np.savez_compressed(
        trace_path,
        **arrays,
    )

    print(
        f"[{index:02d}/{len(SETTINGS):02d}] "
        f"yaw={yaw_degrees:+.1f} "
        f"actions={result.get('active_actions')} "
        f"end=({result.get('end_x_m')},"
        f"{result.get('end_y_m')}) "
        f"support_failures="
        f"{result.get('support_failure_steps')} "
        f"nonselected="
        f"{result.get('nonselected_contact_steps')} "
        f"opposite="
        f"{result.get('opposite_contact_steps')} "
        f"success={result['success']} "
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

print("=" * 100)
print(
    "successful second-leg candidates:",
    sum(
        bool(result["success"])
        for result in results
    ),
    "/",
    len(results),
)
print("summary:", SUMMARY)
