from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glx")

import h5py
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


SUITE_NAME = "libero_spatial"
DATASET_DIR = Path("datasets/libero/datasets/libero_spatial")
TARGET_BOWL = "akita_black_bowl_1"
TARGET_RAMEKIN = "glazed_rim_porcelain_ramekin_1"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit stability of LIBERO Spatial task-5 initial states."
        )
    )
    parser.add_argument(
        "--init-source",
        choices=["benchmark", "hdf5"],
        required=True,
    )
    parser.add_argument("--task-id", type=int, default=5)
    parser.add_argument("--hdf5-path")
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument(
        "--xy-threshold-m",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--z-threshold-m",
        type=float,
        default=0.005,
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--output")
    return parser.parse_args()


def sorted_demo_keys(data_group):
    return sorted(
        data_group.keys(),
        key=lambda name: int(name.replace("demo_", "")),
    )


def load_initial_states(args, suite, task):
    if args.init_source == "benchmark":
        states = suite.get_task_init_states(args.task_id)

        if states is None or len(states) == 0:
            raise RuntimeError("No benchmark initial states found.")

        labels = [
            f"benchmark_init_{index}"
            for index in range(len(states))
        ]

        return [
            np.asarray(state, dtype=np.float64)
            for state in states
        ], labels, None

    if args.hdf5_path:
        hdf5_path = Path(args.hdf5_path)
    else:
        hdf5_path = (
            DATASET_DIR
            / f"{Path(task.bddl_file).stem}_demo.hdf5"
        )

    if not hdf5_path.exists():
        raise FileNotFoundError(
            f"HDF5 file does not exist: {hdf5_path}"
        )

    with h5py.File(hdf5_path, "r") as hdf5_file:
        data_group = hdf5_file["data"]
        demo_keys = sorted_demo_keys(data_group)

        states = [
            np.asarray(
                data_group[demo_key]["states"][0],
                dtype=np.float64,
            )
            for demo_key in demo_keys
        ]

    return states, demo_keys, hdf5_path


def select_bowl_ramekin_pair(env):
    inner = env.env

    missing = [
        name
        for name in [TARGET_BOWL, TARGET_RAMEKIN]
        if name not in inner.obj_body_id
        or name not in inner.object_states_dict
    ]

    if missing:
        raise KeyError(
            f"Missing target objects in environment: {missing}"
        )

    return TARGET_BOWL, TARGET_RAMEKIN


def optional_image(obs, key):
    if key not in obs:
        return None

    return np.asarray(obs[key]).copy()


def snapshot(env, obs, bowl_name, ramekin_name):
    inner = env.env

    bowl = inner.sim.data.body_xpos[
        inner.obj_body_id[bowl_name]
    ].copy()

    ramekin = inner.sim.data.body_xpos[
        inner.obj_body_id[ramekin_name]
    ].copy()

    bowl_state = inner.object_states_dict[bowl_name]
    ramekin_state = inner.object_states_dict[ramekin_name]

    return {
        "bowl": bowl,
        "ramekin": ramekin,
        "contact": bool(
            bowl_state.check_contact(ramekin_state)
        ),
        "on": bool(
            ramekin_state.check_ontop(bowl_state)
        ),
        "agentview": optional_image(
            obs,
            "agentview_image",
        ),
        "wrist": optional_image(
            obs,
            "robot0_eye_in_hand_image",
        ),
    }


def image_mse(first, second):
    if first is None or second is None:
        return float("nan")

    first = first.astype(np.float32) / 255.0
    second = second.astype(np.float32) / 255.0

    return float(np.mean((first - second) ** 2))


def xy_norm(vector):
    return float(np.linalg.norm(vector[:2]))


def xyz_norm(vector):
    return float(np.linalg.norm(vector))


def make_row(
    args,
    index,
    label,
    bowl_key,
    ramekin_key,
    initial,
    step1,
    settled,
    executed_steps,
    done,
):
    initial_relative = (
        initial["bowl"] - initial["ramekin"]
    )
    step1_relative = (
        step1["bowl"] - step1["ramekin"]
    )
    settled_relative = (
        settled["bowl"] - settled["ramekin"]
    )

    step1_relative_change = (
        step1_relative - initial_relative
    )
    settled_relative_change = (
        settled_relative - initial_relative
    )
    step1_to_settled_relative_change = (
        settled_relative - step1_relative
    )

    row = {
        "init_source": args.init_source,
        "init_index": index,
        "state_label": label,
        "bowl_key": bowl_key,
        "ramekin_key": ramekin_key,
        "settle_steps_executed": executed_steps,
        "done_after_settle": bool(done),
        "contact_initial": initial["contact"],
        "contact_step1": step1["contact"],
        "contact_settled": settled["contact"],
        "on_initial": initial["on"],
        "on_step1": step1["on"],
        "on_settled": settled["on"],
        "initial_bowl_ramekin_xy_m": xy_norm(
            initial_relative
        ),
        "step1_bowl_ramekin_xy_m": xy_norm(
            step1_relative
        ),
        "settled_bowl_ramekin_xy_m": xy_norm(
            settled_relative
        ),
        "initial_bowl_above_ramekin_m": float(
            initial_relative[2]
        ),
        "step1_bowl_above_ramekin_m": float(
            step1_relative[2]
        ),
        "settled_bowl_above_ramekin_m": float(
            settled_relative[2]
        ),
        "bowl_xy_move_step1_m": xy_norm(
            step1["bowl"] - initial["bowl"]
        ),
        "bowl_xyz_move_step1_m": xyz_norm(
            step1["bowl"] - initial["bowl"]
        ),
        "bowl_xy_move_settled_m": xy_norm(
            settled["bowl"] - initial["bowl"]
        ),
        "bowl_xyz_move_settled_m": xyz_norm(
            settled["bowl"] - initial["bowl"]
        ),
        "ramekin_xy_move_step1_m": xy_norm(
            step1["ramekin"] - initial["ramekin"]
        ),
        "ramekin_xy_move_settled_m": xy_norm(
            settled["ramekin"] - initial["ramekin"]
        ),
        "relative_xy_change_step1_m": xy_norm(
            step1_relative_change
        ),
        "relative_z_change_step1_m": float(
            step1_relative_change[2]
        ),
        "relative_xy_change_settled_m": xy_norm(
            settled_relative_change
        ),
        "relative_z_change_settled_m": float(
            settled_relative_change[2]
        ),
        "relative_xy_change_step1_to_settled_m": xy_norm(
            step1_to_settled_relative_change
        ),
        "relative_z_change_step1_to_settled_m": float(
            step1_to_settled_relative_change[2]
        ),
        "agentview_mse_step1": image_mse(
            initial["agentview"],
            step1["agentview"],
        ),
        "agentview_mse_settled": image_mse(
            initial["agentview"],
            settled["agentview"],
        ),
        "wrist_mse_step1": image_mse(
            initial["wrist"],
            step1["wrist"],
        ),
        "wrist_mse_settled": image_mse(
            initial["wrist"],
            settled["wrist"],
        ),
    }

    row["unstable_step1"] = bool(
        not row["contact_initial"]
        or not row["contact_step1"]
        or not row["on_initial"]
        or not row["on_step1"]
        or row["relative_xy_change_step1_m"]
        > args.xy_threshold_m
        or abs(row["relative_z_change_step1_m"])
        > args.z_threshold_m
    )

    row["unstable_settled"] = bool(
        not row["contact_initial"]
        or not row["contact_step1"]
        or not row["contact_settled"]
        or not row["on_initial"]
        or not row["on_step1"]
        or not row["on_settled"]
        or row["relative_xy_change_settled_m"]
        > args.xy_threshold_m
        or abs(row["relative_z_change_settled_m"])
        > args.z_threshold_m
    )

    axes = ["x", "y", "z"]

    for prefix, state in [
        ("initial", initial),
        ("step1", step1),
        ("settled", settled),
    ]:
        for axis, value in zip(axes, state["bowl"]):
            row[f"{prefix}_bowl_{axis}_m"] = float(value)

        for axis, value in zip(axes, state["ramekin"]):
            row[f"{prefix}_ramekin_{axis}_m"] = float(value)

    return row


def print_metric(rows, field):
    values = np.asarray(
        [float(row[field]) for row in rows],
        dtype=np.float64,
    )

    print(
        f"{field}: "
        f"median={np.median(values) * 100:.3f} cm, "
        f"max={np.max(values) * 100:.3f} cm"
    )


def main():
    args = parse_args()

    if args.task_id != 5:
        raise ValueError(
            "This audit is designed specifically for task ID 5."
        )

    if args.settle_steps < 1:
        raise ValueError("settle-steps must be at least 1.")

    suite = benchmark.get_benchmark(SUITE_NAME)()
    task = suite.get_task(args.task_id)

    states, labels, hdf5_path = load_initial_states(
        args,
        suite,
        task,
    )

    if len(states) != 50:
        raise ValueError(
            f"Expected 50 initial states, got {len(states)}"
        )

    if args.max_states is not None:
        if args.max_states < 1:
            raise ValueError("max-states must be positive.")

        states = states[:args.max_states]
        labels = labels[:args.max_states]

    bddl_path = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=128,
        camera_widths=128,
    )

    rows = []
    action_low, _ = env.env.action_spec
    zero_action = np.zeros_like(
        action_low,
        dtype=np.float32,
    )

    try:
        for index, (state, label) in enumerate(
            zip(states, labels)
        ):
            np.random.seed(args.seed + index)

            obs = env.reset()
            updated_obs = env.set_init_state(state)

            if updated_obs is None:
                raise RuntimeError(
                    "set_init_state returned no observation."
                )

            obs = updated_obs

            bowl_key, ramekin_key = (
                select_bowl_ramekin_pair(env)
            )

            initial = snapshot(
                env,
                obs,
                bowl_key,
                ramekin_key,
            )

            obs, _, done, _ = env.step(zero_action)
            step1 = snapshot(
                env,
                obs,
                bowl_key,
                ramekin_key,
            )

            executed_steps = 1

            while (
                executed_steps < args.settle_steps
                and not done
            ):
                obs, _, done, _ = env.step(zero_action)
                executed_steps += 1

            settled = snapshot(
                env,
                obs,
                bowl_key,
                ramekin_key,
            )

            row = make_row(
                args,
                index,
                label,
                bowl_key,
                ramekin_key,
                initial,
                step1,
                settled,
                executed_steps,
                done,
            )
            rows.append(row)

            print(
                f"[{index + 1:02d}/{len(states):02d}] "
                f"{label}: "
                f"relative xy drift="
                f"{row['relative_xy_change_settled_m'] * 100:.3f} cm, "
                f"relative dz="
                f"{row['relative_z_change_settled_m'] * 100:.3f} cm, "
                f"contact={row['contact_settled']}, "
                f"on={row['on_settled']}, "
                f"unstable={row['unstable_settled']}"
            )
    finally:
        close = getattr(env, "close", None)

        if callable(close):
            close()

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(
            "results/"
            f"audit_libero_spatial_task5_"
            f"{args.init_source}_init_stability.csv"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    unstable_step1 = [
        int(row["init_index"])
        for row in rows
        if row["unstable_step1"]
    ]
    unstable_settled = [
        int(row["init_index"])
        for row in rows
        if row["unstable_settled"]
    ]

    print("=" * 80)
    print("Task:", task.language)
    print("Init source:", args.init_source)

    if hdf5_path is not None:
        print("HDF5:", hdf5_path)

    print("States audited:", len(rows))
    print(
        "Thresholds:",
        f"xy={args.xy_threshold_m * 100:.3f} cm,",
        f"z={args.z_threshold_m * 100:.3f} cm",
    )

    for field in [
        "relative_xy_change_step1_m",
        "relative_xy_change_settled_m",
        "relative_z_change_step1_m",
        "relative_z_change_settled_m",
        "relative_xy_change_step1_to_settled_m",
        "relative_z_change_step1_to_settled_m",
        "bowl_xyz_move_settled_m",
    ]:
        print_metric(rows, field)

    for stage in ["initial", "step1", "settled"]:
        contact_count = sum(
            bool(row[f"contact_{stage}"])
            for row in rows
        )
        on_count = sum(
            bool(row[f"on_{stage}"])
            for row in rows
        )
        print(
            f"{stage}: contact={contact_count}/{len(rows)}, "
            f"On={on_count}/{len(rows)}"
        )

    print(
        "Unstable after step 1:",
        f"{len(unstable_step1)}/{len(rows)}",
        unstable_step1,
    )
    print(
        f"Unstable after {args.settle_steps} steps:",
        f"{len(unstable_settled)}/{len(rows)}",
        unstable_settled,
    )
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
