from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np
from robosuite.utils import transform_utils as transform


LANGUAGE = "push the white and yellow mug to the middle of the table"
TASK_ID = 16
LAYOUT_ID = 1
TERMINAL_HOLD_STEPS = 20

DEFAULT_PLANS = (
    {
        "seed": 521000,
        "push_scale": 0.6,
        "path": Path(
            "results/libero36_gate5_official_v6/"
            "task_016_push_to/original.npz"
        ),
        "origin": "official_gate5_v6",
    },
    {
        "seed": 521001,
        "push_scale": 0.6,
        "path": Path(
            "/tmp/libero36_task16_demo_pilot_seed_521001_tol060/"
            "attempts/task_16_layout_1_candidate_00.npz"
        ),
        "origin": "task16_demo_seed_pilot",
    },
    {
        "seed": 521002,
        "push_scale": 0.6,
        "path": Path(
            "/tmp/libero36_task16_demo_pilot_seed_521002_tol060/"
            "attempts/task_16_layout_1_candidate_00.npz"
        ),
        "origin": "task16_demo_seed_pilot",
    },
    {
        "seed": 521003,
        "push_scale": 0.9,
        "path": Path(
            "/tmp/libero36_task16_seed_screen_scale090/"
            "seed_521003/trajectory.npz"
        ),
        "origin": "task16_seed_screen_scale090",
    },
    {
        "seed": 521004,
        "push_scale": 1.2,
        "path": Path(
            "/tmp/libero36_task16_seed_screen_scale120/"
            "seed_521004/trajectory.npz"
        ),
        "origin": "task16_seed_screen_scale120",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay five validated Task-16 plans, capture observations, and "
            "write a LIBERO-compatible pilot HDF5 with strict QC evidence."
        )
    )
    parser.add_argument(
        "--controller",
        type=Path,
        default=Path("/tmp/run_mug_push_task16_horiz.py"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/demos/libero36_task16_5_success_pilot_v1"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_controller(controller_path: Path):
    controller_parent = str(controller_path.parent)
    if controller_parent not in sys.path:
        sys.path.insert(0, controller_parent)
    source = controller_path.read_text(encoding="utf-8")
    marker = "calibrator.main()"
    if source.count(marker) != 1:
        raise RuntimeError(
            f"Expected one {marker!r} call, found {source.count(marker)}"
        )
    source = source.replace(
        "calibrator.ACTIVE_GRIPPER_ACTION = -1.0",
        "calibrator.ACTIVE_GRIPPER_ACTION = 0.0",
        1,
    )
    source = source.replace(marker, "", 1)
    namespace: dict[str, object] = {
        "__file__": str(controller_path),
        "__name__": "task16_success_pilot_controller",
    }
    exec(compile(source, str(controller_path), "exec"), namespace)
    calibrator = namespace["calibrator"]
    calibrator.ACTIVE_GRIPPER_ACTION = 0.0
    calibrator.ORIENTATION_TOLERANCE_RAD = 0.060
    calibrator.reset_validator.SOURCE_SAMPLER_RADII_M[
        "white_yellow_mug"
    ] = 0.075
    return calibrator


def resolve_controller(path: Path) -> Path:
    if path.is_file():
        return path.resolve()
    bundled = Path(
        "data/demos/libero36_task16_5_success_pilot_v1/"
        "provenance/run_mug_push_task16_horiz.py"
    )
    if bundled.is_file():
        return bundled.resolve()
    raise FileNotFoundError(path)


def resolve_plans() -> list[dict]:
    bundled_root = Path(
        "data/demos/libero36_task16_5_success_pilot_v1/plans"
    )
    resolved = []
    for index, plan in enumerate(DEFAULT_PLANS):
        item = dict(plan)
        path = Path(item["path"])
        if not path.is_file():
            path = bundled_root / f"demo_{index}_seed_{item['seed']}.npz"
        if not path.is_file():
            raise FileNotFoundError(item["path"])
        item["path"] = path.resolve()
        resolved.append(item)
    return resolved


def layout_row(calibrator) -> dict:
    rows = calibrator.reset_validator.read_layout_spec(
        Path("data/libero_36/layout_spec.csv")
    )
    selected = [
        dict(row)
        for row in rows
        if int(row["task_id"]) == TASK_ID
        and int(row["layout_id"]) == LAYOUT_ID
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected Task {TASK_ID} layout {LAYOUT_ID}, found {len(selected)}"
        )
    return selected[0]


def create_numeric_dataset(group, name: str, shape, dtype):
    return group.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        compression="lzf",
        shuffle=True,
    )


def create_image_dataset(group, name: str, length: int):
    return group.create_dataset(
        name,
        shape=(length, 128, 128, 3),
        dtype=np.uint8,
        chunks=(1, 128, 128, 3),
        compression="lzf",
        shuffle=True,
    )


def replay_and_write(
    calibrator,
    row: dict,
    plan: dict,
    demo_group,
) -> dict:
    seed = int(plan["seed"])
    plan_path = Path(plan["path"])
    with np.load(plan_path, allow_pickle=True) as loaded:
        plan_arrays = {name: loaded[name] for name in loaded.files}

    actions = np.asarray(plan_arrays["actions"], dtype=np.float32)
    plan_states = np.asarray(plan_arrays["states"], dtype=np.float64)
    phases = np.asarray(plan_arrays["phase"]).astype(str)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Seed {seed}: invalid action shape {actions.shape}")
    if plan_states.shape[0] != actions.shape[0] + 1:
        raise ValueError(
            f"Seed {seed}: states/actions mismatch "
            f"{plan_states.shape[0]} != {actions.shape[0]} + 1"
        )
    if phases.shape != (actions.shape[0],):
        raise ValueError(f"Seed {seed}: invalid phase shape {phases.shape}")

    length = actions.shape[0]
    target_name = calibrator.target_instance(row)
    env = calibrator.reset_validator.make_environment(Path(row["bddl_path"]))
    try:
        settled = calibrator.settle_environment(env, row, seed)
        obs = settled["obs"]
        geometry = calibrator.contact_geometry(env, target_name)
        initial_state = np.asarray(
            env.get_sim_state(), dtype=np.float64
        ).copy()
        initial_diff = float(np.max(np.abs(initial_state - plan_states[0])))
        if initial_diff != 0.0:
            raise RuntimeError(
                f"Seed {seed}: initial state is not exact; max diff={initial_diff}"
            )

        obs_group = demo_group.create_group("obs")
        datasets = {
            "actions": create_numeric_dataset(
                demo_group, "actions", actions.shape, np.float32
            ),
            "states": create_numeric_dataset(
                demo_group, "states", (length, plan_states.shape[1]), np.float64
            ),
            "robot_states": create_numeric_dataset(
                demo_group, "robot_states", (length, 9), np.float64
            ),
            "rewards": create_numeric_dataset(
                demo_group, "rewards", (length,), np.float32
            ),
            "dones": create_numeric_dataset(
                demo_group, "dones", (length,), np.uint8
            ),
            "env_dones": create_numeric_dataset(
                demo_group, "env_dones", (length,), np.uint8
            ),
            "agentview": create_image_dataset(
                obs_group, "agentview_rgb", length
            ),
            "wrist": create_image_dataset(
                obs_group, "eye_in_hand_rgb", length
            ),
            "joint": create_numeric_dataset(
                obs_group, "joint_states", (length, 7), np.float64
            ),
            "ee_pos": create_numeric_dataset(
                obs_group, "ee_pos", (length, 3), np.float64
            ),
            "ee_ori": create_numeric_dataset(
                obs_group, "ee_ori", (length, 3), np.float64
            ),
            "ee_states": create_numeric_dataset(
                obs_group, "ee_states", (length, 6), np.float64
            ),
            "gripper": create_numeric_dataset(
                obs_group, "gripper_states", (length, 2), np.float64
            ),
        }
        string_dtype = h5py.string_dtype(encoding="utf-8")
        demo_group.create_dataset(
            "phase",
            data=np.asarray(phases, dtype=object),
            dtype=string_dtype,
        )

        traces = {
            "left_contact": [],
            "right_contact": [],
            "grasp_proxy": [],
            "table_support": [],
            "unexpected_object_contact": [],
            "robot_distractor_contact": [],
            "nonselected_robot_target_contact": [],
            "relation": [],
            "xy_in_target": [],
        }
        maximum_state_diff = 0.0
        rewards = np.empty(length, dtype=np.float32)
        env_dones = np.empty(length, dtype=np.uint8)

        for index, action in enumerate(actions):
            state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
            maximum_state_diff = max(
                maximum_state_diff,
                float(np.max(np.abs(state - plan_states[index]))),
            )
            agentview = np.asarray(obs["agentview_image"])
            wrist = np.asarray(obs["robot0_eye_in_hand_image"])
            joint = np.asarray(obs["robot0_joint_pos"], dtype=np.float64)
            ee_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
            ee_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
            ee_ori = np.asarray(
                transform.quat2axisangle(ee_quat), dtype=np.float64
            )
            gripper = np.asarray(
                obs["robot0_gripper_qpos"], dtype=np.float64
            )
            if agentview.shape != (128, 128, 3):
                raise ValueError(
                    f"Seed {seed} frame {index}: agentview {agentview.shape}"
                )
            if wrist.shape != (128, 128, 3):
                raise ValueError(
                    f"Seed {seed} frame {index}: wrist {wrist.shape}"
                )

            datasets["actions"][index] = action
            datasets["states"][index] = state
            datasets["joint"][index] = joint
            datasets["ee_pos"][index] = ee_pos
            datasets["ee_ori"][index] = ee_ori
            datasets["ee_states"][index] = np.concatenate([ee_pos, ee_ori])
            datasets["gripper"][index] = gripper
            datasets["robot_states"][index] = np.concatenate([joint, gripper])
            datasets["agentview"][index] = agentview.astype(np.uint8, copy=False)
            datasets["wrist"][index] = wrist.astype(np.uint8, copy=False)

            next_obs, reward, env_done, _ = env.step(action)
            rewards[index] = float(reward)
            env_dones[index] = int(bool(env_done))
            next_state = np.asarray(
                env.get_sim_state(), dtype=np.float64
            ).copy()
            maximum_state_diff = max(
                maximum_state_diff,
                float(np.max(np.abs(next_state - plan_states[index + 1]))),
            )

            left, right, grasp = calibrator.push_contact_record(env, geometry)
            support = calibrator.table_support(env.env, target_name)
            unexpected, distractor, nonselected = (
                calibrator.active_unexpected_contacts(
                    env, target_name, geometry, "left"
                )
            )
            relation, xy_ok = calibrator.target_region_status(env, next_obs, row)
            traces["left_contact"].append(bool(left))
            traces["right_contact"].append(bool(right))
            traces["grasp_proxy"].append(bool(grasp))
            traces["table_support"].append(bool(support))
            traces["unexpected_object_contact"].append(bool(unexpected))
            traces["robot_distractor_contact"].append(bool(distractor))
            traces["nonselected_robot_target_contact"].append(bool(nonselected))
            traces["relation"].append(bool(relation))
            traces["xy_in_target"].append(bool(xy_ok))
            obs = next_obs

        datasets["rewards"][:] = rewards
        datasets["env_dones"][:] = env_dones
        episode_dones = np.zeros(length, dtype=np.uint8)
        episode_dones[-1] = 1
        datasets["dones"][:] = episode_dones

        trace_match = True
        trace_group = demo_group.create_group("diagnostics")
        for key, values in traces.items():
            array = np.asarray(values, dtype=bool)
            trace_group.create_dataset(key, data=array, compression="lzf")
            if key in plan_arrays:
                trace_match = trace_match and bool(
                    np.array_equal(array, np.asarray(plan_arrays[key], dtype=bool))
                )

        terminal_relation = bool(
            np.asarray(traces["relation"], dtype=bool)[
                -TERMINAL_HOLD_STEPS:
            ].all()
        )
        terminal_xy = bool(
            np.asarray(traces["xy_in_target"], dtype=bool)[
                -TERMINAL_HOLD_STEPS:
            ].all()
        )
        support_all = bool(np.asarray(traces["table_support"], dtype=bool).all())
        no_grasp = not any(traces["grasp_proxy"])
        no_distractor = not any(traces["robot_distractor_contact"])
        no_unexpected = not any(traces["unexpected_object_contact"])
        actions_exact = bool(
            np.array_equal(np.asarray(datasets["actions"]), actions)
        )
        state_exact = maximum_state_diff == 0.0
        passed = bool(
            terminal_relation
            and terminal_xy
            and support_all
            and no_grasp
            and no_distractor
            and no_unexpected
            and actions_exact
            and state_exact
            and trace_match
        )

        demo_group.attrs["num_samples"] = length
        demo_group.attrs["init_state"] = initial_state
        demo_group.attrs["seed"] = seed
        demo_group.attrs["task_id"] = TASK_ID
        demo_group.attrs["layout_id"] = LAYOUT_ID
        demo_group.attrs["source_plan"] = str(plan_path)
        demo_group.attrs["source_plan_sha256"] = sha256(plan_path)
        demo_group.attrs["source_origin"] = str(plan["origin"])
        demo_group.attrs["push_scale"] = float(plan["push_scale"])
        demo_group.attrs["exact_state_replay"] = state_exact
        demo_group.attrs["exact_action_replay"] = actions_exact
        demo_group.attrs["diagnostic_trace_match"] = trace_match
        demo_group.attrs["terminal_relation"] = terminal_relation
        demo_group.attrs["terminal_xy"] = terminal_xy
        demo_group.attrs["support_all"] = support_all
        demo_group.attrs["passed"] = passed
        demo_group.attrs["auxiliary_target_contact_policy"] = (
            "official_v6_allows_non_grasping_auxiliary_robot_target_contact"
        )

        return {
            "seed": seed,
            "frames": length,
            "push_scale": float(plan["push_scale"]),
            "source_plan": str(plan_path),
            "source_sha256": sha256(plan_path),
            "initial_state_max_abs_diff": initial_diff,
            "trajectory_state_max_abs_diff": maximum_state_diff,
            "actions_exact": actions_exact,
            "trace_match": trace_match,
            "terminal_relation": terminal_relation,
            "terminal_xy": terminal_xy,
            "support_all": support_all,
            "grasp_steps": int(sum(traces["grasp_proxy"])),
            "robot_distractor_contact_steps": int(
                sum(traces["robot_distractor_contact"])
            ),
            "unexpected_object_contact_steps": int(
                sum(traces["unexpected_object_contact"])
            ),
            "auxiliary_robot_target_contact_steps": int(
                sum(traces["nonselected_robot_target_contact"])
            ),
            "reward_positive_steps": int(np.count_nonzero(rewards > 0)),
            "env_done_steps": int(env_dones.sum()),
            "passed": passed,
        }
    finally:
        env.close()


def validate_hdf5(path: Path) -> dict:
    failures: list[str] = []
    frames = 0
    initial_states = []
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        demo_names = sorted(data.keys())
        if len(demo_names) != 5:
            failures.append(f"episode_count={len(demo_names)}")
        for name in demo_names:
            demo = data[name]
            length = int(demo.attrs["num_samples"])
            frames += length
            initial_states.append(np.asarray(demo.attrs["init_state"]))
            expected_shapes = {
                "actions": (length, 7),
                "states": (length, 71),
                "robot_states": (length, 9),
                "obs/agentview_rgb": (length, 128, 128, 3),
                "obs/eye_in_hand_rgb": (length, 128, 128, 3),
                "obs/joint_states": (length, 7),
                "obs/ee_pos": (length, 3),
                "obs/ee_ori": (length, 3),
                "obs/gripper_states": (length, 2),
                "rewards": (length,),
                "dones": (length,),
            }
            for key, expected in expected_shapes.items():
                if key not in demo:
                    failures.append(f"{name}:missing:{key}")
                    continue
                if demo[key].shape != expected:
                    failures.append(
                        f"{name}:{key}:shape={demo[key].shape}:expected={expected}"
                    )
            if demo["obs/agentview_rgb"].dtype != np.uint8:
                failures.append(f"{name}:agentview_dtype")
            if demo["obs/eye_in_hand_rgb"].dtype != np.uint8:
                failures.append(f"{name}:wrist_dtype")
            for key in (
                "actions",
                "states",
                "robot_states",
                "obs/joint_states",
                "obs/ee_pos",
                "obs/ee_ori",
                "obs/gripper_states",
                "rewards",
            ):
                if not np.isfinite(np.asarray(demo[key])).all():
                    failures.append(f"{name}:{key}:nonfinite")
            actions = np.asarray(demo["actions"])
            if np.any(actions < -1.0) or np.any(actions > 1.0):
                failures.append(f"{name}:actions_out_of_range")
            dones = np.asarray(demo["dones"], dtype=np.uint8)
            if dones.sum() != 1 or dones[-1] != 1:
                failures.append(f"{name}:episode_incomplete")
            if not bool(demo.attrs.get("passed", False)):
                failures.append(f"{name}:pilot_checks_failed")

        for left in range(len(initial_states)):
            for right in range(left + 1, len(initial_states)):
                if np.array_equal(initial_states[left], initial_states[right]):
                    failures.append(
                        f"duplicate_initial_states:{demo_names[left]}:{demo_names[right]}"
                    )

        if int(data.attrs.get("num_demos", -1)) != len(demo_names):
            failures.append("num_demos_attribute")
        if int(data.attrs.get("total", -1)) != frames:
            failures.append("total_attribute")

    return {
        "passed": not failures,
        "failures": failures,
        "episodes": len(initial_states),
        "frames": frames,
        "file_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    controller_path = resolve_controller(args.controller)
    plans = resolve_plans()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_root}; use --overwrite"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    calibrator = load_controller(controller_path)
    row = layout_row(calibrator)

    final_hdf5 = output_root / "task_016_layout_1_5_success.hdf5"
    partial_hdf5 = output_root / "task_016_layout_1_5_success.partial.hdf5"
    results = []
    with h5py.File(partial_hdf5, "w") as handle:
        data = handle.create_group("data")
        data.attrs["bddl_file_name"] = str(row["bddl_path"])
        data.attrs["env_name"] = "Libero_Tabletop_Manipulation"
        data.attrs["macros_image_convention"] = "opengl"
        data.attrs["tag"] = "libero36-task16-success-pilot-v1"
        data.attrs["problem_info"] = json.dumps(
            {
                "problem_name": "libero_tabletop_manipulation",
                "domain_name": "robosuite",
                "language_instruction": LANGUAGE,
                "task_id": TASK_ID,
                "layout_id": LAYOUT_ID,
            },
            sort_keys=True,
        )
        data.attrs["env_args"] = json.dumps(
            {
                "camera_names": ["robot0_eye_in_hand", "agentview"],
                "camera_heights": 128,
                "camera_widths": 128,
                "control_freq": 20,
                "action_dimension": 7,
            },
            sort_keys=True,
        )

        for index, plan in enumerate(plans):
            demo = data.create_group(f"demo_{index}")
            result = replay_and_write(calibrator, row, plan, demo)
            results.append(result)
            print(
                f"[{index + 1}/5] seed={result['seed']} "
                f"frames={result['frames']} exact="
                f"{result['trajectory_state_max_abs_diff'] == 0.0} "
                f"passed={result['passed']}",
                flush=True,
            )
            if not result["passed"]:
                raise RuntimeError(f"Seed {result['seed']} failed replay/QC")

        data.attrs["num_demos"] = len(results)
        data.attrs["total"] = sum(item["frames"] for item in results)
        handle.flush()

    os.replace(partial_hdf5, final_hdf5)
    hdf5_qc = validate_hdf5(final_hdf5)
    if not hdf5_qc["passed"]:
        raise RuntimeError(f"HDF5 QC failed: {hdf5_qc['failures']}")

    plans_root = output_root / "plans"
    provenance_root = output_root / "provenance"
    plans_root.mkdir()
    provenance_root.mkdir()
    for index, plan in enumerate(plans):
        shutil.copy2(
            plan["path"],
            plans_root / f"demo_{index}_seed_{plan['seed']}.npz",
        )
    shutil.copy2(controller_path, provenance_root / controller_path.name)
    calibrator_path = Path(calibrator.__file__).resolve()
    shutil.copy2(calibrator_path, provenance_root / calibrator_path.name)

    qc_csv = output_root / "episode_qc.csv"
    with qc_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    manifest = {
        "schema_version": 1,
        "artifact_kind": "libero36_task16_success_demonstration_pilot",
        "task_id": TASK_ID,
        "layout_id": LAYOUT_ID,
        "language_instruction": LANGUAGE,
        "episodes": len(results),
        "total_frames": sum(item["frames"] for item in results),
        "seeds": [item["seed"] for item in results],
        "formal_gate5_evidence_modified": False,
        "auxiliary_target_contact_policy": (
            "official_v6_allows_non_grasping_auxiliary_robot_target_contact"
        ),
        "episodes_qc": results,
        "hdf5_qc": hdf5_qc,
        "files": {
            "hdf5": {
                "path": str(final_hdf5),
                "sha256": sha256(final_hdf5),
                "bytes": final_hdf5.stat().st_size,
            },
            "episode_qc_csv": {
                "path": str(qc_csv),
                "sha256": sha256(qc_csv),
            },
            "controller": {
                "path": str(provenance_root / controller_path.name),
                "sha256": sha256(provenance_root / controller_path.name),
            },
            "calibrator": {
                "path": str(provenance_root / calibrator_path.name),
                "sha256": sha256(provenance_root / calibrator_path.name),
            },
        },
        "passed": bool(
            hdf5_qc["passed"]
            and len(results) == 5
            and all(item["passed"] for item in results)
        ),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("=" * 88)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("=" * 88)
    print("TASK 16 FIVE-SUCCESS PILOT PASSED:", manifest["passed"])
    print("HDF5:", final_hdf5)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
