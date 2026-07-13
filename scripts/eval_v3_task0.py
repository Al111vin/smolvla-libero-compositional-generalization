from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glx")

import h5py
import numpy as np
import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


SUITE_NAME = "libero_spatial"

DEFAULT_TASK0_HDF5 = Path(
    "datasets/libero/datasets/libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_"
    "and_place_it_on_the_plate_demo.hdf5"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a V3 SmolVLA checkpoint on LIBERO task 0."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint pretrained_model directory.",
    )
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument(
        "--init-source",
        choices=["benchmark", "hdf5"],
        default="benchmark",
    )
    parser.add_argument("--init-index", type=int, default=0)
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument(
        "--hdf5-path",
        default=str(DEFAULT_TASK0_HDF5),
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--wait-steps",
        type=int,
        default=None,
        help=(
            "Defaults to 10 for benchmark initialization "
            "and 0 for HDF5 initialization."
        ),
    )
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def quat_xyzw_to_axis_angle(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q)

    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)

    q = q / norm
    x, y, z, w = q

    w = np.clip(w, -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    scale = np.sqrt(max(1.0 - w * w, 0.0))

    if scale < 1e-8:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = np.array([x, y, z]) / scale

    return (axis * angle).astype(np.float32)


def image_to_tensor(image):
    tensor = torch.as_tensor(image).permute(2, 0, 1).float()

    if tensor.max() > 1:
        tensor = tensor / 255.0

    return tensor


def observation_to_frame(obs, language):
    joint = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)
    ee_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    ee_ori = quat_xyzw_to_axis_angle(obs["robot0_eef_quat"])
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)

    state = np.concatenate(
        [joint, ee_pos, ee_ori, gripper],
        axis=0,
    ).astype(np.float32)

    if state.shape != (15,):
        raise ValueError(
            f"Expected state shape (15,), got {state.shape}"
        )

    return {
        "observation.images.agentview": image_to_tensor(
            obs["agentview_image"]
        ),
        "observation.images.wrist": image_to_tensor(
            obs["robot0_eye_in_hand_image"]
        ),
        "observation.state": torch.from_numpy(state),
        "task": language,
    }


def action_to_numpy(action):
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()

    action = np.asarray(action, dtype=np.float32)
    action = np.squeeze(action)

    if action.shape != (7,):
        raise ValueError(
            f"Expected action shape (7,), got {action.shape}"
        )

    return action


def checkpoint_label(checkpoint):
    checkpoint = Path(checkpoint)

    if checkpoint.name == "pretrained_model":
        return checkpoint.parent.name

    return checkpoint.name


def run_episode(args):
    checkpoint = Path(args.checkpoint)

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint}"
        )

    device = torch.device(args.device)

    print("Loading checkpoint:", checkpoint)

    policy = SmolVLAPolicy.from_pretrained(
        str(checkpoint)
    ).to(device).eval()

    if args.n_action_steps is not None:
        if not 1 <= args.n_action_steps <= policy.config.chunk_size:
            raise ValueError(
                "n-action-steps must be between 1 and "
                f"{policy.config.chunk_size}"
            )

        policy.config.n_action_steps = args.n_action_steps

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {
                "device": str(device),
            }
        },
    )

    suite = benchmark.get_benchmark(SUITE_NAME)()
    task = suite.get_task(args.task_id)

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

    if args.init_source == "benchmark":
        init_states = suite.get_task_init_states(
            args.task_id
        )

        if init_states is None or len(init_states) == 0:
            raise RuntimeError(
                "No LIBERO initial states available."
            )

        if not 0 <= args.init_index < len(init_states):
            raise IndexError(
                f"init-index {args.init_index} is invalid; "
                f"available range is "
                f"0..{len(init_states) - 1}"
            )

        initial_state = np.asarray(
            init_states[args.init_index],
            dtype=np.float64,
        )
        initialization_index = args.init_index
        initialization_label = (
            f"benchmark_init{args.init_index}"
        )
        default_wait_steps = 10

    else:
        hdf5_path = Path(args.hdf5_path)

        if not hdf5_path.exists():
            raise FileNotFoundError(
                f"HDF5 dataset does not exist: {hdf5_path}"
            )

        demo_key = f"demo_{args.demo_index}"

        with h5py.File(hdf5_path, "r") as file:
            if demo_key not in file["data"]:
                raise KeyError(
                    f"{demo_key} is not present in {hdf5_path}"
                )

            initial_state = np.asarray(
                file[f"data/{demo_key}/states"][0],
                dtype=np.float64,
            )

        initialization_index = args.demo_index
        initialization_label = (
            f"hdf5_demo{args.demo_index}"
        )
        default_wait_steps = 0

    effective_wait_steps = (
        default_wait_steps
        if args.wait_steps is None
        else args.wait_steps
    )

    if effective_wait_steps < 0:
        raise ValueError("wait-steps cannot be negative")

    obs = env.reset()
    updated_obs = env.set_init_state(initial_state)

    if updated_obs is not None:
        obs = updated_obs

    zero_action = np.zeros(7, dtype=np.float32)

    for _ in range(effective_wait_steps):
        obs, _, done, _ = env.step(zero_action)

        if done:
            break

    seed = args.seed + initialization_index
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    policy.reset()

    total_reward = 0.0
    info = {}
    action_rows = []

    for step in range(args.max_steps):
        frame = observation_to_frame(obs, task.language)
        batch = preprocess(frame)

        with torch.inference_mode():
            raw_action = policy.select_action(batch)
            processed_action = postprocess(raw_action)

        raw_action_np = action_to_numpy(raw_action)
        processed_action_np = action_to_numpy(
            processed_action
        )

        # LIBERO OSC actions are expected in [-1, 1].
        applied_action = np.clip(
            processed_action_np,
            -1.0,
            1.0,
        )

        obs, reward, done, info = env.step(
            applied_action
        )
        total_reward += float(reward)

        row = {
            "step": step,
            "reward": float(reward),
        }

        for index in range(7):
            row[f"raw_action_{index}"] = float(
                raw_action_np[index]
            )
            row[f"processed_action_{index}"] = float(
                processed_action_np[index]
            )
            row[f"applied_action_{index}"] = float(
                applied_action[index]
            )

        action_rows.append(row)

        if done or reward > 0:
            break

    steps = step + 1
    success = bool(
        info.get(
            "success",
            info.get("is_success", False),
        )
    ) or total_reward > 0

    env.close()

    label = checkpoint_label(checkpoint)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    run_label = (
        f"eval_v3_task{args.task_id}_{label}_"
        f"{initialization_label}_"
        f"n{policy.config.n_action_steps}_seed{seed}"
    )

    summary_path = results_dir / f"{run_label}_summary.csv"
    actions_path = results_dir / f"{run_label}_actions.csv"

    summary_fields = [
        "suite",
        "task_id",
        "init_source",
        "init_index",
        "demo_index",
        "language",
        "success",
        "total_reward",
        "steps",
        "wait_steps",
        "n_action_steps",
        "seed",
        "checkpoint",
    ]

    summary_row = {
        "suite": SUITE_NAME,
        "task_id": args.task_id,
        "init_source": args.init_source,
        "init_index": (
            args.init_index
            if args.init_source == "benchmark"
            else ""
        ),
        "demo_index": (
            args.demo_index
            if args.init_source == "hdf5"
            else ""
        ),
        "language": task.language,
        "success": success,
        "total_reward": total_reward,
        "steps": steps,
        "wait_steps": effective_wait_steps,
        "n_action_steps": policy.config.n_action_steps,
        "seed": seed,
        "checkpoint": str(checkpoint),
    }

    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=summary_fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(summary_row)

    action_fields = ["step", "reward"]

    for prefix in [
        "raw_action",
        "processed_action",
        "applied_action",
    ]:
        action_fields.extend(
            f"{prefix}_{index}"
            for index in range(7)
        )

    with actions_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=action_fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(action_rows)

    print("=" * 80)
    print("Checkpoint:", checkpoint)
    print("Task:", task.language)
    print("Init source:", args.init_source)
    print("Initialization:", initialization_label)
    print("Wait steps:", effective_wait_steps)
    print("Action steps:", policy.config.n_action_steps)
    print("Seed:", seed)
    print("Success:", success)
    print("Total reward:", total_reward)
    print("Steps:", steps)
    print("Summary:", summary_path)
    print("Actions:", actions_path)


def main():
    args = parse_args()
    run_episode(args)


if __name__ == "__main__":
    main()