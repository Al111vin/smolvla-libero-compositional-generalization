from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scripts import calibrate_libero_36_push as calibrator
from scripts import validate_libero_36_envs as validator


ROOT = Path(
    "/tmp/libero36_gate5_mug_tasks9_11_formal"
)
LAYOUT_SPEC = Path(
    "data/libero_36/layout_spec.csv"
)
TASK_IDS = (9, 10, 11)
LAYOUT_ID = 1


def exact_match(left, right) -> bool:
    return bool(
        np.array_equal(
            np.asarray(left),
            np.asarray(right),
        )
    )


def replay_one(row: dict) -> dict:
    task_id = int(row["task_id"])
    seed = validator.seed_for(
        row,
        calibrator.GATE3_CALIBRATION_RESET_INDEX,
        validator.FULL_SEED_BASE,
    )
    target = calibrator.target_instance(row)

    trajectory_path = ROOT / (
        f"task_{task_id:03d}_layout_1_trajectory.npz"
    )
    replay_path = ROOT / (
        f"task_{task_id:03d}_layout_1_replay.npz"
    )
    summary_path = ROOT / (
        f"task_{task_id:03d}_layout_1_replay_summary.json"
    )

    result = {
        "task_id": task_id,
        "layout_id": LAYOUT_ID,
        "seed": int(seed),
        "trajectory_path": str(trajectory_path),
        "replay_path": str(replay_path),
        "summary_path": str(summary_path),
        "replay_passed": False,
        "error": "",
    }

    env = None

    try:
        if not trajectory_path.is_file():
            raise FileNotFoundError(trajectory_path)

        with np.load(
            trajectory_path,
            allow_pickle=False,
        ) as recorded:
            trajectory = {
                name: np.asarray(recorded[name]).copy()
                for name in recorded.files
            }

        actions = np.asarray(
            trajectory["actions"],
            dtype=np.float32,
        )
        expected_states = np.asarray(
            trajectory["states"],
            dtype=np.float64,
        )
        phases = np.asarray(
            trajectory["phase"],
        ).astype(str)

        if actions.ndim != 2 or actions.shape[1] != 7:
            raise RuntimeError(
                f"Invalid action shape: {actions.shape}"
            )
        if expected_states.shape[0] != actions.shape[0]:
            raise RuntimeError(
                "Recorded state/action lengths disagree"
            )
        if phases.shape != (actions.shape[0],):
            raise RuntimeError(
                "Recorded phase/action lengths disagree"
            )
        if not np.isfinite(actions).all():
            raise RuntimeError(
                "Recorded actions contain non-finite values"
            )
        if (
            np.any(actions < -1.0)
            or np.any(actions > 1.0)
        ):
            raise RuntimeError(
                "Recorded actions exceed [-1, 1]"
            )

        recorded_task = int(
            np.asarray(
                trajectory["task_id"]
            ).reshape(())
        )
        recorded_layout = int(
            np.asarray(
                trajectory["layout_id"]
            ).reshape(())
        )
        recorded_seed = int(
            np.asarray(
                trajectory["seed"]
            ).reshape(())
        )

        if recorded_task != task_id:
            raise RuntimeError(
                f"Task mismatch: {recorded_task} != {task_id}"
            )
        if recorded_layout != LAYOUT_ID:
            raise RuntimeError(
                "Layout mismatch: "
                f"{recorded_layout} != {LAYOUT_ID}"
            )
        if recorded_seed != seed:
            raise RuntimeError(
                f"Seed mismatch: {recorded_seed} != {seed}"
            )

        env = validator.make_environment(
            Path(row["bddl_path"])
        )
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

        replay_initial_state = np.asarray(
            env.get_sim_state(),
            dtype=np.float64,
        ).copy()
        recorded_initial_state = np.asarray(
            trajectory["initial_state"],
            dtype=np.float64,
        )

        if replay_initial_state.shape != (
            recorded_initial_state.shape
        ):
            raise RuntimeError(
                "Initial state shapes disagree"
            )

        initial_state_max_abs_diff = float(
            np.max(
                np.abs(
                    replay_initial_state
                    - recorded_initial_state
                )
            )
        )

        replay_states = []
        replay_left = []
        replay_right = []
        replay_grasp = []
        replay_relation = []
        replay_done = []
        replay_distractor = []
        replay_robot_plate = []
        replay_unexpected = []
        replay_target_xyz = []
        replay_eef_xyz = []

        for index, action in enumerate(actions):
            obs, _, done, _ = env.step(action)
            relation = bool(env.check_success())

            if bool(done) != relation:
                raise RuntimeError(
                    "Replay done flag disagrees with "
                    "goal predicate at action "
                    f"{index + 1}: "
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

            replay_states.append(
                np.asarray(
                    env.get_sim_state(),
                    dtype=np.float64,
                ).copy()
            )
            replay_left.append(bool(left))
            replay_right.append(bool(right))
            replay_grasp.append(bool(grasp))
            replay_relation.append(bool(relation))
            replay_done.append(bool(done))
            replay_distractor.append(bool(distractor))
            replay_robot_plate.append(bool(robot_plate))
            replay_unexpected.append(bool(unexpected))
            replay_target_xyz.append(
                calibrator.target_xyz(
                    obs,
                    target,
                ).copy()
            )
            replay_eef_xyz.append(
                calibrator.eef_xyz(obs).copy()
            )

        replay_arrays = {
            "states": np.asarray(
                replay_states,
                dtype=np.float64,
            ),
            "left_contact": np.asarray(
                replay_left,
                dtype=np.bool_,
            ),
            "right_contact": np.asarray(
                replay_right,
                dtype=np.bool_,
            ),
            "grasp_proxy": np.asarray(
                replay_grasp,
                dtype=np.bool_,
            ),
            "relation": np.asarray(
                replay_relation,
                dtype=np.bool_,
            ),
            "done": np.asarray(
                replay_done,
                dtype=np.bool_,
            ),
            "robot_distractor_contact": np.asarray(
                replay_distractor,
                dtype=np.bool_,
            ),
            "robot_plate_contact": np.asarray(
                replay_robot_plate,
                dtype=np.bool_,
            ),
            "unexpected_object_contact": np.asarray(
                replay_unexpected,
                dtype=np.bool_,
            ),
            "target_xyz": np.asarray(
                replay_target_xyz,
                dtype=np.float64,
            ),
            "eef_xyz": np.asarray(
                replay_eef_xyz,
                dtype=np.float64,
            ),
        }

        if replay_arrays["states"].shape != (
            expected_states.shape
        ):
            raise RuntimeError(
                "Replay state shape disagrees with recording"
            )

        trajectory_state_max_abs_diff = float(
            np.max(
                np.abs(
                    replay_arrays["states"]
                    - expected_states
                )
            )
        )

        trace_names = (
            "left_contact",
            "right_contact",
            "grasp_proxy",
            "relation",
            "done",
            "robot_distractor_contact",
            "robot_plate_contact",
            "unexpected_object_contact",
            "target_xyz",
            "eef_xyz",
        )

        trace_matches = {
            name: exact_match(
                replay_arrays[name],
                trajectory[name],
            )
            for name in trace_names
        }

        terminal_steps = (
            calibrator.TERMINAL_HOLD_STEPS
        )
        terminal_slice = slice(
            len(actions) - terminal_steps,
            len(actions),
        )

        terminal_phases_ok = bool(
            len(actions) >= terminal_steps
            and np.all(
                phases[terminal_slice]
                == "terminal_hold"
            )
        )
        terminal_actions_zero = bool(
            len(actions) >= terminal_steps
            and np.array_equal(
                actions[terminal_slice],
                np.zeros(
                    (terminal_steps, 7),
                    dtype=np.float32,
                ),
            )
        )
        terminal_relation_hold = bool(
            len(actions) >= terminal_steps
            and replay_arrays["relation"][
                terminal_slice
            ].all()
        )

        final_poses = validator.capture_poses(
            obs,
            "plate_1",
        )
        receiver_region_preserved = bool(
            calibrator.goal_validator
            .receiver_requested_region_ok(
                inner,
                final_poses,
                row,
                "plate_1",
            )
        )

        required_grasp_phase = (
            "close_rim_pinch"
            if task_id == 1
            else "seat_grasp"
        )
        required_phase_mask = (
            phases == required_grasp_phase
        )
        grasp_before_transport = bool(
            required_phase_mask.any()
            and replay_arrays["grasp_proxy"][
                required_phase_mask
            ].any()
        )

        checks = {
            "initial_state_exact": bool(
                initial_state_max_abs_diff == 0.0
            ),
            "trajectory_state_exact": bool(
                trajectory_state_max_abs_diff == 0.0
            ),
            "all_recorded_traces_exact": bool(
                all(trace_matches.values())
            ),
            "done_exactly_matches_relation": bool(
                exact_match(
                    replay_arrays["done"],
                    replay_arrays["relation"],
                )
            ),
            "goal_acquired": bool(
                replay_arrays["relation"].any()
            ),
            "grasp_before_transport": bool(
                grasp_before_transport
            ),
            "grasp_released_final": bool(
                not replay_arrays[
                    "grasp_proxy"
                ][-1]
            ),
            "no_robot_distractor_contact": bool(
                not replay_arrays[
                    "robot_distractor_contact"
                ].any()
            ),
            "no_robot_plate_contact": bool(
                not replay_arrays[
                    "robot_plate_contact"
                ].any()
            ),
            "no_unexpected_object_contact": bool(
                not replay_arrays[
                    "unexpected_object_contact"
                ].any()
            ),
            "terminal_phases_exact": bool(
                terminal_phases_ok
            ),
            "terminal_actions_zero": bool(
                terminal_actions_zero
            ),
            "terminal_relation_hold": bool(
                terminal_relation_hold
            ),
            "receiver_region_preserved": bool(
                receiver_region_preserved
            ),
            "within_action_horizon": bool(
                len(actions)
                <= calibrator.MAX_ACTIVE_ACTIONS
            ),
        }

        failed_checks = [
            name
            for name, passed in checks.items()
            if not passed
        ]

        np.savez_compressed(
            replay_path,
            task_id=np.asarray(
                task_id,
                dtype=np.int64,
            ),
            layout_id=np.asarray(
                LAYOUT_ID,
                dtype=np.int64,
            ),
            seed=np.asarray(
                seed,
                dtype=np.int64,
            ),
            initial_state=replay_initial_state,
            actions=actions.copy(),
            states=replay_arrays["states"],
            phase=phases.copy(),
            left_contact=replay_arrays[
                "left_contact"
            ],
            right_contact=replay_arrays[
                "right_contact"
            ],
            grasp_proxy=replay_arrays[
                "grasp_proxy"
            ],
            relation=replay_arrays["relation"],
            done=replay_arrays["done"],
            robot_distractor_contact=replay_arrays[
                "robot_distractor_contact"
            ],
            robot_plate_contact=replay_arrays[
                "robot_plate_contact"
            ],
            unexpected_object_contact=replay_arrays[
                "unexpected_object_contact"
            ],
            target_xyz=replay_arrays["target_xyz"],
            eef_xyz=replay_arrays["eef_xyz"],
        )

        result.update(
            {
                "actions": int(len(actions)),
                "initial_state_max_abs_diff":
                    initial_state_max_abs_diff,
                "trajectory_state_max_abs_diff":
                    trajectory_state_max_abs_diff,
                "trace_matches": trace_matches,
                "checks": checks,
                "failed_checks": failed_checks,
                "first_goal_action": (
                    int(
                        np.flatnonzero(
                            replay_arrays["relation"]
                        )[0]
                        + 1
                    )
                    if replay_arrays[
                        "relation"
                    ].any()
                    else None
                ),
                "done_action_count": int(
                    replay_arrays["done"].sum()
                ),
                "replay_passed": bool(
                    not failed_checks
                ),
            }
        )

    except Exception as exc:
        result["error"] = (
            f"{type(exc).__name__}: {exc}"
        )
        result["failed_checks"] = [
            "replay_exception"
        ]

    finally:
        if env is not None:
            env.close()

    summary_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return result


def main():
    layout_rows = validator.read_layout_spec(
        LAYOUT_SPEC
    )
    selected_rows = sorted(
        [
            row
            for row in layout_rows
            if row["task_id"] in TASK_IDS
            and row["layout_id"] == LAYOUT_ID
        ],
        key=lambda row: row["task_id"],
    )

    if [
        row["task_id"]
        for row in selected_rows
    ] != list(TASK_IDS):
        raise RuntimeError(
            "Expected tasks 1 and 2 on layout 1"
        )

    results = []

    for row in selected_rows:
        result = replay_one(row)
        results.append(result)

        print(
            f"task={result['task_id']:02d} "
            f"actions={result.get('actions')} "
            f"state_diff="
            f"{result.get('trajectory_state_max_abs_diff')} "
            f"failed={result.get('failed_checks')} "
            f"passed={result['replay_passed']} "
            f"error={result['error']}"
        )

    fields = [
        "task_id",
        "layout_id",
        "seed",
        "actions",
        "initial_state_max_abs_diff",
        "trajectory_state_max_abs_diff",
        "done_action_count",
        "first_goal_action",
        "replay_passed",
        "failed_checks_json",
        "error",
        "trajectory_path",
        "replay_path",
        "summary_path",
    ]

    csv_rows = []
    for result in results:
        csv_rows.append(
            {
                key: (
                    json.dumps(
                        result.get(
                            "failed_checks",
                            [],
                        ),
                        separators=(",", ":"),
                    )
                    if key == "failed_checks_json"
                    else result.get(key, "")
                )
                for key in fields
            }
        )

    aggregate_path = ROOT / "replay_summary.csv"
    with aggregate_path.open(
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
        writer.writerows(csv_rows)

    all_passed = bool(
        len(results) == len(TASK_IDS)
        and all(
            result["replay_passed"]
            for result in results
        )
    )

    manifest = {
        "protocol_version":
            "libero_36_proxy_tabletop_draft_v5",
        "gate": 5,
        "task_ids": list(TASK_IDS),
        "layout_id": LAYOUT_ID,
        "completed_replays": len(results),
        "passed_replays": sum(
            bool(result["replay_passed"])
            for result in results
        ),
        "all_replays_passed": all_passed,
        "summary_csv": str(aggregate_path),
    }

    manifest_path = ROOT / "replay_manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 80)
    print(
        "exact replay passed:",
        manifest["passed_replays"],
        "/",
        len(TASK_IDS),
    )
    print("summary:", aggregate_path)
    print("manifest:", manifest_path)

    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
