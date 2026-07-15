from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import re
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glx")

import numpy as np
import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from eval_v3_task0 import action_to_numpy, observation_to_frame


SUITE_NAME = "libero_spatial"
PROTOCOL_VERSION = "v4_loco_official_v1"
OFFICIAL_MAX_STEPS = 280
OFFICIAL_WAIT_STEPS = 10
OFFICIAL_DUMMY_ACTION = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
    dtype=np.float32,
)

SUMMARY_FIELDS = [
    "protocol_version",
    "git_commit",
    "evaluator_sha256",
    "evaluation_mode",
    "model_label",
    "heldout_task_id",
    "evaluation_role",
    "suite",
    "task_id",
    "init_source",
    "init_index",
    "language",
    "success",
    "steps",
    "first_success_step",
    "total_reward",
    "max_steps",
    "wait_steps",
    "n_action_steps",
    "env_seed",
    "policy_seed",
    "checkpoint_label",
    "checkpoint_sha256",
    "checkpoint",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a frozen V4 SmolVLA LOCO model on fixed "
            "LIBERO-Spatial benchmark initial states."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--mode",
        choices=["sanity", "formal"],
        required=True,
    )
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--heldout-task-id", type=int, required=True)
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--init-indices",
        type=int,
        nargs="+",
        default=list(range(50)),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=OFFICIAL_MAX_STEPS,
    )
    parser.add_argument(
        "--wait-steps",
        type=int,
        default=OFFICIAL_WAIT_STEPS,
    )
    parser.add_argument("--n-action-steps", type=int, default=25)
    parser.add_argument("--base-env-seed", type=int, default=1000)
    parser.add_argument("--base-policy-seed", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--save-actions", action="store_true")
    return parser.parse_args()


def checkpoint_label(checkpoint: Path) -> str:
    if checkpoint.name == "pretrained_model":
        return checkpoint.parent.name
    return checkpoint.name


def checkpoint_fingerprint(checkpoint: Path) -> str:
    paths = sorted(
        path for path in checkpoint.rglob("*") if path.is_file()
    )
    if not paths:
        raise FileNotFoundError(
            f"No checkpoint files found in {checkpoint}"
        )

    digest = hashlib.sha256()
    for path in paths:
        digest.update(
            str(path.relative_to(checkpoint)).encode("utf-8")
        )
        with path.open("rb") as file:
            while chunk := file.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def current_git_commit() -> str:
    root = Path(__file__).resolve().parent.parent
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def evaluator_fingerprint(require_protocol: bool) -> str:
    script_path = Path(__file__).resolve()
    adapter_path = script_path.with_name("eval_v3_task0.py")
    protocol_path = (
        script_path.parent.parent
        / "results"
        / "V4_LOCO_EVAL_PROTOCOL.md"
    )
    paths = [script_path, adapter_path]
    if require_protocol:
        paths.append(protocol_path)

    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required evaluator dependency is missing: {path}"
            )
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def require_clean_formal_files():
    root = Path(__file__).resolve().parent.parent
    relative_paths = [
        "scripts/eval_v4_loco.py",
        "scripts/eval_v3_task0.py",
        "results/V4_LOCO_EVAL_PROTOCOL.md",
    ]

    for relative_path in relative_paths:
        subprocess.run(
            [
                "git",
                "ls-files",
                "--error-unmatch",
                relative_path,
            ],
            cwd=root,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        for cached in [False, True]:
            command = ["git", "diff", "--quiet"]
            if cached:
                command.append("--cached")
            command.extend(["--", relative_path])
            subprocess.run(command, cwd=root, check=True)


def validate_args(args, checkpoint: Path):
    if not checkpoint.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint}"
        )

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.model_label):
        raise ValueError(
            "model-label may contain only letters, digits, '.', '_' and '-'"
        )

    if len(set(args.task_ids)) != len(args.task_ids):
        raise ValueError("task-ids contains duplicates")

    if len(set(args.init_indices)) != len(args.init_indices):
        raise ValueError("init-indices contains duplicates")

    if any(index < 0 for index in args.init_indices):
        raise ValueError("init-indices cannot be negative")

    if args.max_steps != OFFICIAL_MAX_STEPS:
        raise ValueError(
            f"Formal protocol requires max-steps={OFFICIAL_MAX_STEPS}"
        )

    if args.wait_steps != OFFICIAL_WAIT_STEPS:
        raise ValueError(
            f"Formal protocol requires wait-steps={OFFICIAL_WAIT_STEPS}"
        )

    if args.n_action_steps != 25:
        raise ValueError("Formal protocol requires n-action-steps=25")

    if args.base_env_seed != 1000:
        raise ValueError("Formal protocol requires base-env-seed=1000")

    if args.base_policy_seed != 2000:
        raise ValueError("Formal protocol requires base-policy-seed=2000")

    expected_heldout = {
        "task3_holdout": 3,
        "task6_holdout": 6,
    }
    if args.model_label not in expected_heldout:
        raise ValueError(
            "model-label must be task3_holdout or task6_holdout"
        )
    if args.heldout_task_id != expected_heldout[args.model_label]:
        raise ValueError(
            "heldout-task-id does not match model-label"
        )

    if args.mode == "formal":
        checkpoint_parts = checkpoint.resolve().parts
        expected_run_component = (
            f"smolvla_v4_loco_task{args.heldout_task_id}_90k_run1"
        )
        if checkpoint.parent.name != "090000":
            raise ValueError(
                "Formal mode requires the preselected 090000 checkpoint"
            )
        if expected_run_component not in checkpoint_parts:
            raise ValueError(
                "Formal checkpoint path does not match model-label: "
                f"expected path component {expected_run_component!r}"
            )
        if set(args.task_ids) != {3, 6} or len(args.task_ids) != 2:
            raise ValueError(
                "Formal mode requires exactly task-ids 3 and 6"
            )
        if args.init_indices != list(range(50)):
            raise ValueError(
                "Formal mode requires benchmark init indices 0..49"
            )
        if not args.save_actions:
            raise ValueError(
                "Formal mode requires --save-actions"
            )
        require_clean_formal_files()
    elif args.heldout_task_id in args.task_ids:
        raise ValueError(
            "Sanity mode cannot evaluate the held-out target task"
        )


def seed_environment(env, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if hasattr(env, "seed"):
        env.seed(seed)


def seed_policy(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_completed(
    summary_path: Path,
    expected: dict[str, str],
    heldout_task_id: int,
    base_env_seed: int,
    base_policy_seed: int,
    actions_dir: Path,
    require_actions: bool,
):
    completed = set()
    if not summary_path.exists():
        return completed

    with summary_path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != SUMMARY_FIELDS:
            raise ValueError(
                "Existing summary header does not match evaluator"
            )
        rows = list(reader)

    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(
                "Existing summary contains a torn or malformed row"
            )
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(
                    f"Existing summary protocol mismatch for {key}: "
                    f"{row.get(key)!r} != {value!r}"
                )

        pair = (int(row["task_id"]), int(row["init_index"]))
        task_id, init_index = pair
        expected_role = (
            "heldout"
            if task_id == heldout_task_id
            else "seen_control"
        )
        dynamic_expected = {
            "heldout_task_id": str(heldout_task_id),
            "evaluation_role": expected_role,
            "env_seed": str(base_env_seed + init_index),
            "policy_seed": str(base_policy_seed + init_index),
        }
        for key, value in dynamic_expected.items():
            if row.get(key) != value:
                raise ValueError(
                    f"Existing summary mismatch for {key}: "
                    f"{row.get(key)!r} != {value!r}"
                )

        if pair in completed:
            raise ValueError(
                f"Duplicate completed rollout in summary: {pair}"
            )

        if require_actions:
            action_path = (
                actions_dir
                / f"task{task_id}_init{init_index:02d}.npz"
            )
            if not action_path.is_file():
                raise FileNotFoundError(
                    f"Missing action trace for completed rollout: {pair}"
                )
            with np.load(action_path) as trace:
                required_keys = {
                    "raw_actions",
                    "processed_actions",
                    "applied_actions",
                    "rewards",
                    "official_success",
                    "done",
                }
                if set(trace.files) != required_keys:
                    raise ValueError(
                        f"Malformed action trace for rollout: {pair}"
                    )
                if len(trace["rewards"]) != int(row["steps"]):
                    raise ValueError(
                        f"Action trace length mismatch for rollout: {pair}"
                    )
        completed.add(pair)
    return completed


def append_summary(summary_path: Path, row: dict):
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=SUMMARY_FIELDS,
            lineterminator="\n",
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        file.flush()
        os.fsync(file.fileno())


def save_action_trace(path: Path, trace: dict[str, list]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    arrays = {
        "raw_actions": np.asarray(trace["raw_actions"], dtype=np.float32),
        "processed_actions": np.asarray(
            trace["processed_actions"], dtype=np.float32
        ),
        "applied_actions": np.asarray(
            trace["applied_actions"], dtype=np.float32
        ),
        "rewards": np.asarray(trace["rewards"], dtype=np.float32),
        "official_success": np.asarray(
            trace["official_success"], dtype=np.bool_
        ),
        "done": np.asarray(trace["done"], dtype=np.bool_),
    }

    with temporary_path.open("wb") as file:
        np.savez_compressed(file, **arrays)
    temporary_path.replace(path)


def make_environment(task):
    bddl_path = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    return OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=128,
        camera_widths=128,
    )


def initialize_episode(env, initial_state, env_seed: int, wait_steps: int):
    seed_environment(env, env_seed)
    obs = env.reset()
    updated_obs = env.set_init_state(
        np.asarray(initial_state, dtype=np.float64)
    )
    if updated_obs is not None:
        obs = updated_obs

    for wait_index in range(wait_steps):
        obs, _, done, _ = env.step(OFFICIAL_DUMMY_ACTION)
        if done:
            raise RuntimeError(
                f"Environment ended during stabilization step {wait_index}"
            )

    if bool(env.check_success()):
        raise RuntimeError(
            "Benchmark initialization already satisfies the task goal"
        )
    return obs


def run_episode(
    env,
    task,
    initial_state,
    policy,
    preprocess,
    postprocess,
    env_seed: int,
    policy_seed: int,
    max_steps: int,
    wait_steps: int,
):
    obs = initialize_episode(
        env,
        initial_state,
        env_seed,
        wait_steps,
    )

    seed_policy(policy_seed)
    policy.reset()

    trace = {
        "raw_actions": [],
        "processed_actions": [],
        "applied_actions": [],
        "rewards": [],
        "official_success": [],
        "done": [],
    }

    total_reward = 0.0
    success = False
    first_success_step = None

    for step_index in range(max_steps):
        frame = observation_to_frame(obs, task.language)
        batch = preprocess(frame)

        with torch.inference_mode():
            raw_action = policy.select_action(batch)
            processed_action = postprocess(raw_action)

        raw_action_np = action_to_numpy(raw_action)
        processed_action_np = action_to_numpy(processed_action)
        applied_action = np.clip(
            processed_action_np,
            -1.0,
            1.0,
        )

        obs, reward, done, _ = env.step(applied_action)
        official_success = bool(env.check_success())
        total_reward += float(reward)

        trace["raw_actions"].append(raw_action_np)
        trace["processed_actions"].append(processed_action_np)
        trace["applied_actions"].append(applied_action)
        trace["rewards"].append(float(reward))
        trace["official_success"].append(official_success)
        trace["done"].append(bool(done))

        if official_success:
            success = True
            first_success_step = step_index + 1
            break

        if done:
            break

    return {
        "success": success,
        "steps": len(trace["rewards"]),
        "first_success_step": first_success_step,
        "total_reward": total_reward,
        "trace": trace,
    }


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    validate_args(args, checkpoint)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / (
        f"{args.model_label}_{args.mode}_summary.csv"
    )
    actions_dir = results_dir / (
        f"{args.model_label}_{args.mode}_actions"
    )

    git_commit = current_git_commit()
    if args.mode == "formal" and git_commit == "unknown":
        raise RuntimeError(
            "Formal evaluation requires a Git commit"
        )
    evaluator_sha256 = evaluator_fingerprint(
        require_protocol=args.mode == "formal"
    )
    print("Hashing checkpoint weights...")
    checkpoint_sha256 = checkpoint_fingerprint(checkpoint)
    label = checkpoint_label(checkpoint)

    expected_resume_fields = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": git_commit,
        "evaluator_sha256": evaluator_sha256,
        "evaluation_mode": args.mode,
        "model_label": args.model_label,
        "suite": SUITE_NAME,
        "init_source": "benchmark",
        "max_steps": str(args.max_steps),
        "wait_steps": str(args.wait_steps),
        "n_action_steps": str(args.n_action_steps),
        "checkpoint_label": label,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint": str(checkpoint),
    }
    completed = read_completed(
        summary_path,
        expected_resume_fields,
        heldout_task_id=args.heldout_task_id,
        base_env_seed=args.base_env_seed,
        base_policy_seed=args.base_policy_seed,
        actions_dir=actions_dir,
        require_actions=args.save_actions,
    )

    device = torch.device(args.device)
    print("Loading checkpoint:", checkpoint)
    policy = SmolVLAPolicy.from_pretrained(
        str(checkpoint)
    ).to(device).eval()

    expected_input_shapes = {
        "observation.images.agentview": (3, 128, 128),
        "observation.images.wrist": (3, 128, 128),
        "observation.state": (15,),
    }
    actual_input_shapes = {
        name: tuple(feature.shape)
        for name, feature in policy.config.input_features.items()
    }
    actual_output_shapes = {
        name: tuple(feature.shape)
        for name, feature in policy.config.output_features.items()
    }
    if actual_input_shapes != expected_input_shapes:
        raise ValueError(
            "Checkpoint input features do not match the V4 protocol: "
            f"{actual_input_shapes}"
        )
    if actual_output_shapes != {"action": (7,)}:
        raise ValueError(
            "Checkpoint output features do not match the V4 protocol: "
            f"{actual_output_shapes}"
        )
    if policy.config.chunk_size != 50:
        raise ValueError(
            "Checkpoint chunk size does not match the V4 protocol: "
            f"{policy.config.chunk_size}"
        )

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
            "device_processor": {"device": str(device)}
        },
    )

    suite = benchmark.get_benchmark(SUITE_NAME)()
    requested_pairs = [
        (task_id, init_index)
        for task_id in args.task_ids
        for init_index in args.init_indices
    ]

    print("=" * 80)
    print("Protocol:", PROTOCOL_VERSION)
    print("Mode:", args.mode)
    print("Git commit:", git_commit)
    print("Model:", args.model_label)
    print("Held-out task:", args.heldout_task_id)
    print("Task IDs:", args.task_ids)
    print("Init indices:", args.init_indices)
    print("Max steps:", args.max_steps)
    print("Wait steps:", args.wait_steps)
    print("Action steps:", args.n_action_steps)
    print("Already complete:", len(completed))
    print("Requested rollouts:", len(requested_pairs))

    completed_this_run = 0
    for task_id in args.task_ids:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        if init_states is None or len(init_states) == 0:
            raise RuntimeError(
                f"No benchmark initial states for task {task_id}"
            )

        invalid_indices = [
            index
            for index in args.init_indices
            if index >= len(init_states)
        ]
        if invalid_indices:
            raise IndexError(
                f"Task {task_id} invalid init indices: {invalid_indices}"
            )

        env = make_environment(task)
        try:
            for init_index in args.init_indices:
                pair = (task_id, init_index)
                if pair in completed:
                    print(f"[skip] task={task_id} init={init_index}")
                    continue

                env_seed = args.base_env_seed + init_index
                policy_seed = args.base_policy_seed + init_index
                result = run_episode(
                    env=env,
                    task=task,
                    initial_state=init_states[init_index],
                    policy=policy,
                    preprocess=preprocess,
                    postprocess=postprocess,
                    env_seed=env_seed,
                    policy_seed=policy_seed,
                    max_steps=args.max_steps,
                    wait_steps=args.wait_steps,
                )

                if args.save_actions:
                    action_path = (
                        actions_dir
                        / f"task{task_id}_init{init_index:02d}.npz"
                    )
                    save_action_trace(action_path, result["trace"])

                role = (
                    "heldout"
                    if task_id == args.heldout_task_id
                    else "seen_control"
                )
                row = {
                    "protocol_version": PROTOCOL_VERSION,
                    "git_commit": git_commit,
                    "evaluator_sha256": evaluator_sha256,
                    "evaluation_mode": args.mode,
                    "model_label": args.model_label,
                    "heldout_task_id": args.heldout_task_id,
                    "evaluation_role": role,
                    "suite": SUITE_NAME,
                    "task_id": task_id,
                    "init_source": "benchmark",
                    "init_index": init_index,
                    "language": task.language,
                    "success": result["success"],
                    "steps": result["steps"],
                    "first_success_step": (
                        ""
                        if result["first_success_step"] is None
                        else result["first_success_step"]
                    ),
                    "total_reward": result["total_reward"],
                    "max_steps": args.max_steps,
                    "wait_steps": args.wait_steps,
                    "n_action_steps": args.n_action_steps,
                    "env_seed": env_seed,
                    "policy_seed": policy_seed,
                    "checkpoint_label": label,
                    "checkpoint_sha256": checkpoint_sha256,
                    "checkpoint": str(checkpoint),
                }
                append_summary(summary_path, row)
                completed.add(pair)
                completed_this_run += 1

                print(
                    f"[{len(completed):03d}/{len(requested_pairs):03d}] "
                    f"task={task_id} init={init_index:02d} "
                    f"role={role} success={result['success']} "
                    f"steps={result['steps']}"
                )
        finally:
            env.close()

    print("=" * 80)
    print("Evaluation complete")
    print("Completed this run:", completed_this_run)
    print("Total summary rows:", len(completed))
    print("Summary:", summary_path)
    if args.save_actions:
        print("Actions:", actions_dir)


if __name__ == "__main__":
    main()
