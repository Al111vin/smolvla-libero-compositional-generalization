from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glx")

import imageio.v2 as imageio
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


SUITE_NAME = "libero_spatial"
FORMAL_PROTOCOL_VERSION = "v4_loco_official_v1"
DIAGNOSTIC_PROTOCOL_VERSION = "v4_loco_diagnostic_v1"
MAX_STEPS = 280
WAIT_STEPS = 10
N_ACTION_STEPS = 25
BASE_ENV_SEED = 1000
BASE_POLICY_SEED = 2000
VIDEO_FPS = 20
LIFT_THRESHOLD_M = 0.03
APPROACH_THRESHOLD_M = 0.05
NEAR_PLATE_XY_THRESHOLD_M = 0.08
REWARD_ATOL = 1e-7
VIDEO_FRAME_SHAPE = (128, 256, 3)
DEFAULT_SELECTION_PATH = (
    "data/diagnostics/v4_loco_case_selection.csv"
)
DEFAULT_FORMAL_ROLLOUTS_PATH = (
    "results/eval_v4_loco_formal_rollouts.csv"
)
DEFAULT_FORMAL_MANIFEST_PATH = (
    "results/eval_v4_loco_manifest.json"
)
DEFAULT_DIAGNOSTIC_RESULTS_DIR = "results/v4_loco_diagnostics"
OFFICIAL_DUMMY_ACTION = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
    dtype=np.float32,
)

MODEL_SPECS = {
    "task3_holdout": {
        "heldout_task_id": 3,
        "seen_task_id": 6,
    },
    "task6_holdout": {
        "heldout_task_id": 6,
        "seen_task_id": 3,
    },
}

TRACE_KEYS = {
    "raw_actions",
    "processed_actions",
    "applied_actions",
    "rewards",
    "official_success",
    "done",
}

SELECTION_FIELDS = {
    "task_id",
    "init_index",
    "stratum",
    "seen_model",
    "heldout_model",
    "selection_seed",
    "diagnostic_only",
}

STEP_FIELDS = (
    [
        "step",
        "source_reward",
        "replay_reward",
        "reward_abs_diff",
        "source_done",
        "replay_done",
        "source_official_success",
        "replay_official_success",
    ]
    + [f"raw_action_{index}" for index in range(7)]
    + [f"processed_action_{index}" for index in range(7)]
    + [f"applied_action_{index}" for index in range(7)]
    + [f"robot_joint_{index}" for index in range(7)]
    + [
        "eef_x",
        "eef_y",
        "eef_z",
        "gripper_qpos_0",
        "gripper_qpos_1",
        "gripper_qvel_0",
        "gripper_qvel_1",
        "bowl_x",
        "bowl_y",
        "bowl_z",
        "cookies_x",
        "cookies_y",
        "cookies_z",
        "plate_x",
        "plate_y",
        "plate_z",
        "eef_to_bowl_distance_m",
        "bowl_to_cookies_xy_distance_m",
        "bowl_to_cookies_distance_m",
        "bowl_to_plate_xy_distance_m",
        "bowl_to_plate_distance_m",
        "bowl_lift_m",
        "bowl_lifted",
        "eef_within_5cm_of_bowl",
        "bowl_within_8cm_xy_of_plate",
        "left_fingerpad_contact",
        "right_fingerpad_contact",
        "grasp_contact_proxy",
        "bowl_contact_cookies",
        "bowl_contact_plate",
        "bowl_ontop_plate",
    ]
)

SUMMARY_FIELDS = [
    "diagnostic_protocol_version",
    "git_commit",
    "evaluator_sha256",
    "evaluation_mode",
    "model_label",
    "heldout_task_id",
    "evaluation_role",
    "suite",
    "task_id",
    "init_index",
    "language",
    "stratum",
    "selection_seed",
    "selection_sha256",
    "formal_rollouts_sha256",
    "formal_manifest_sha256",
    "formal_protocol_version",
    "formal_git_commit",
    "formal_evaluator_sha256",
    "expected_formal_success",
    "expected_formal_steps",
    "replay_success",
    "success_matches_formal",
    "replay_steps",
    "steps_match_formal",
    "terminal_flags_match_formal",
    "max_reward_abs_diff",
    "first_replay_success_step",
    "replay_total_reward",
    "max_steps",
    "wait_steps",
    "n_action_steps",
    "env_seed",
    "policy_seed",
    "checkpoint_label",
    "checkpoint_sha256",
    "checkpoint",
    "source_action_trace_bundle_sha256",
    "source_action_trace_sha256",
    "source_action_trace",
    "bowl_collision_geom_count",
    "bowl_collision_geoms",
    "initial_bowl_z",
    "initial_eef_to_bowl_distance_m",
    "initial_bowl_to_plate_xy_distance_m",
    "initial_bowl_to_plate_distance_m",
    "ever_approached_bowl",
    "first_approach_step",
    "ever_near_plate",
    "first_near_plate_step",
    "ever_grasp_contact_proxy",
    "first_grasp_contact_step",
    "ever_lifted",
    "first_lift_step",
    "max_bowl_lift_m",
    "ever_contact_cookies",
    "first_contact_cookies_step",
    "ever_contact_plate",
    "first_contact_plate_step",
    "ever_ontop_plate",
    "first_ontop_plate_step",
    "min_eef_to_bowl_distance_m",
    "min_bowl_to_plate_xy_distance_m",
    "min_bowl_to_plate_distance_m",
    "final_eef_x",
    "final_eef_y",
    "final_eef_z",
    "final_bowl_x",
    "final_bowl_y",
    "final_bowl_z",
    "final_cookies_x",
    "final_cookies_y",
    "final_cookies_z",
    "final_plate_x",
    "final_plate_y",
    "final_plate_z",
    "step_csv_rows",
    "video_frames",
    "step_csv_sha256",
    "video_sha256",
    "step_csv",
    "video",
    "diagnostic_only",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay the exact saved V4 formal action traces on selected "
            "LIBERO-Spatial cases while recording state, contact, grasp, "
            "and video diagnostics. These post-hoc replays never replace "
            "the frozen formal success rates."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "diagnostic"],
        required=True,
    )
    parser.add_argument(
        "--model-label",
        choices=["all", *sorted(MODEL_SPECS)],
        default="all",
    )
    parser.add_argument(
        "--selection-csv",
        default=DEFAULT_SELECTION_PATH,
    )
    parser.add_argument(
        "--formal-rollouts-csv",
        default=DEFAULT_FORMAL_ROLLOUTS_PATH,
    )
    parser.add_argument(
        "--formal-manifest",
        default=DEFAULT_FORMAL_MANIFEST_PATH,
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_DIAGNOSTIC_RESULTS_DIR,
    )
    parser.add_argument("--smoke-task-id", type=int)
    parser.add_argument("--smoke-init-index", type=int)
    return parser.parse_args()


def parse_bool(value, label: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid boolean for {label}: {value!r}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file_set(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(
            str(path.relative_to(root)).encode("utf-8")
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


def evaluator_fingerprint() -> str:
    root = Path(__file__).resolve().parent.parent
    paths = [
        Path(__file__).resolve(),
        root / "results/V4_LOCO_EVAL_PROTOCOL.md",
        root / "results/V4_LOCO_DIAGNOSTIC_PROTOCOL.md",
    ]

    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing diagnostic dependency: {path}"
            )
        digest.update(
            str(path.relative_to(root)).encode("utf-8")
        )
        digest.update(path.read_bytes())
    return digest.hexdigest()


def require_tracked_clean_diagnostic_files():
    root = Path(__file__).resolve().parent.parent
    relative_paths = [
        "scripts/eval_v4_loco_diagnostics.py",
        "results/V4_LOCO_EVAL_PROTOCOL.md",
        "results/V4_LOCO_DIAGNOSTIC_PROTOCOL.md",
        "data/diagnostics/v4_loco_case_selection.csv",
        "results/eval_v4_loco_formal_rollouts.csv",
        "results/eval_v4_loco_manifest.json",
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


def validate_args(args):
    smoke_values = [
        args.smoke_task_id is not None,
        args.smoke_init_index is not None,
    ]
    if any(smoke_values) and not all(smoke_values):
        raise ValueError(
            "--smoke-task-id and --smoke-init-index must be used together"
        )
    if args.mode == "diagnostic" and any(smoke_values):
        raise ValueError(
            "Smoke case arguments are only valid in smoke mode"
        )
    if args.mode == "diagnostic":
        if args.model_label != "all":
            raise ValueError(
                "Full diagnostic mode requires --model-label all"
            )
        frozen_paths = {
            "selection CSV": (
                args.selection_csv,
                DEFAULT_SELECTION_PATH,
            ),
            "formal rollout CSV": (
                args.formal_rollouts_csv,
                DEFAULT_FORMAL_ROLLOUTS_PATH,
            ),
            "formal manifest": (
                args.formal_manifest,
                DEFAULT_FORMAL_MANIFEST_PATH,
            ),
            "diagnostic results directory": (
                args.results_dir,
                DEFAULT_DIAGNOSTIC_RESULTS_DIR,
            ),
        }
        for label, (actual, expected) in frozen_paths.items():
            if (
                resolve_repo_path(actual).resolve()
                != resolve_repo_path(expected).resolve()
            ):
                raise ValueError(
                    f"Full diagnostic mode requires the frozen "
                    f"{label}: {expected}"
                )
        require_tracked_clean_diagnostic_files()


def read_selection(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Selection CSV not found: {path}")

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("Selection CSV has no header")
        if set(reader.fieldnames) != SELECTION_FIELDS:
            raise ValueError(
                "Selection CSV fields do not match the diagnostic protocol"
            )
        rows = list(reader)

    if len(rows) != 20:
        raise ValueError(
            f"Expected 20 diagnostic cases, found {len(rows)}"
        )

    seen_pairs = set()
    counts = {}
    expected_models = {
        3: ("task6_holdout", "task3_holdout"),
        6: ("task3_holdout", "task6_holdout"),
    }
    normalized = []

    for row in rows:
        task_id = int(row["task_id"])
        init_index = int(row["init_index"])
        pair = (task_id, init_index)
        if pair in seen_pairs:
            raise ValueError(f"Duplicate selected case: {pair}")
        seen_pairs.add(pair)

        if task_id not in expected_models:
            raise ValueError(f"Unexpected selected task: {task_id}")
        if not 0 <= init_index < 50:
            raise ValueError(f"Invalid selected init index: {pair}")
        if row["stratum"] not in {
            "seen_success_heldout_failure",
            "both_failure",
        }:
            raise ValueError(
                f"Unexpected selection stratum: {row['stratum']}"
            )

        expected_seen, expected_heldout = expected_models[task_id]
        if (
            row["seen_model"] != expected_seen
            or row["heldout_model"] != expected_heldout
        ):
            raise ValueError(f"Model pairing mismatch for case: {pair}")
        if not parse_bool(
            row["diagnostic_only"],
            "selection diagnostic_only",
        ):
            raise ValueError(
                f"Selection case is not diagnostic-only: {pair}"
            )

        expected_seed = 20260716 + task_id
        selection_seed = int(row["selection_seed"])
        if selection_seed != expected_seed:
            raise ValueError(
                f"Selection seed mismatch for {pair}: "
                f"{selection_seed} != {expected_seed}"
            )

        count_key = (task_id, row["stratum"])
        counts[count_key] = counts.get(count_key, 0) + 1
        normalized.append(
            {
                **row,
                "task_id": task_id,
                "init_index": init_index,
                "selection_seed": selection_seed,
            }
        )

    expected_counts = {
        (task_id, stratum): 5
        for task_id in [3, 6]
        for stratum in [
            "seen_success_heldout_failure",
            "both_failure",
        ]
    }
    if counts != expected_counts:
        raise ValueError(
            f"Unexpected selection stratum counts: {counts}"
        )

    return sorted(
        normalized,
        key=lambda row: (
            row["task_id"],
            row["stratum"],
            row["init_index"],
        ),
    )


def selected_models(args) -> list[str]:
    if args.model_label != "all":
        return [args.model_label]
    if args.mode == "smoke":
        return ["task3_holdout"]
    return sorted(MODEL_SPECS)


def selected_cases(args, rows: list[dict]) -> list[dict]:
    if args.mode == "diagnostic":
        return rows
    if args.smoke_task_id is None:
        return rows[:1]

    matches = [
        row
        for row in rows
        if row["task_id"] == args.smoke_task_id
        and row["init_index"] == args.smoke_init_index
    ]
    if len(matches) != 1:
        raise ValueError(
            "Requested smoke case is not in the frozen selection: "
            f"task={args.smoke_task_id}, init={args.smoke_init_index}"
        )
    return matches


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root = Path(__file__).resolve().parent.parent
    return root / candidate


def load_manifest(
    path: Path,
    formal_rollouts_path: Path,
) -> tuple[dict, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Formal manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != FORMAL_PROTOCOL_VERSION:
        raise ValueError("Formal manifest protocol mismatch")
    models = manifest.get("models")
    if not isinstance(models, dict) or set(models) != set(MODEL_SPECS):
        raise ValueError("Formal manifest model section mismatch")

    rollout_protocol = manifest.get("rollout_protocol")
    expected_rollout_protocol = {
        "suite": SUITE_NAME,
        "tasks": [3, 6],
        "init_indices": list(range(50)),
        "max_steps": MAX_STEPS,
        "wait_steps": WAIT_STEPS,
        "stabilization_action": OFFICIAL_DUMMY_ACTION.tolist(),
        "n_action_steps": N_ACTION_STEPS,
        "base_env_seed": BASE_ENV_SEED,
        "base_policy_seed": BASE_POLICY_SEED,
        "success_criterion": "env.check_success()",
    }
    if rollout_protocol != expected_rollout_protocol:
        raise ValueError("Formal manifest rollout protocol mismatch")

    protocol_document = manifest.get("protocol_document")
    if not isinstance(protocol_document, dict):
        raise ValueError("Formal manifest protocol document is missing")
    protocol_path = resolve_repo_path(protocol_document.get("path", ""))
    if not protocol_path.is_file():
        raise FileNotFoundError(
            f"Frozen formal protocol document is missing: {protocol_path}"
        )
    if sha256_file(protocol_path) != protocol_document.get("sha256"):
        raise ValueError("Frozen formal protocol document hash mismatch")

    outputs = manifest.get("outputs")
    rollout_output = (
        outputs.get("rollouts")
        if isinstance(outputs, dict)
        else None
    )
    if not isinstance(rollout_output, dict):
        raise ValueError("Formal manifest rollout output is missing")
    recorded_rollout_path = resolve_repo_path(
        rollout_output.get("path", "")
    )
    if recorded_rollout_path.resolve() != formal_rollouts_path.resolve():
        raise ValueError(
            "Formal rollout CSV path does not match its manifest"
        )
    if sha256_file(formal_rollouts_path) != rollout_output.get("sha256"):
        raise ValueError(
            "Formal rollout CSV hash does not match its manifest"
        )

    for model_label, spec in MODEL_SPECS.items():
        recorded = models[model_label]
        if int(recorded.get("heldout_task_id", -1)) != int(
            spec["heldout_task_id"]
        ):
            raise ValueError(
                f"Formal manifest held-out task mismatch: {model_label}"
            )
        if int(recorded.get("action_trace_count", -1)) != 100:
            raise ValueError(
                f"Formal manifest trace count mismatch: {model_label}"
            )
        for field in [
            "checkpoint",
            "checkpoint_sha256",
            "action_trace_bundle_sha256",
        ]:
            if not recorded.get(field):
                raise ValueError(
                    f"Formal manifest model field is missing: "
                    f"{model_label}.{field}"
                )
    return manifest, sha256_file(path)


def validate_source_trace(
    trace_path: Path,
    expected_steps: int,
    expected_success: bool,
) -> dict[str, np.ndarray]:
    if not trace_path.is_file():
        raise FileNotFoundError(
            f"Missing frozen formal action trace: {trace_path}"
        )

    with np.load(trace_path) as trace:
        if set(trace.files) != TRACE_KEYS:
            raise ValueError(
                f"Unexpected formal trace fields: {trace_path}"
            )
        arrays = {
            key: np.asarray(trace[key]).copy()
            for key in TRACE_KEYS
        }

    for key in [
        "raw_actions",
        "processed_actions",
        "applied_actions",
    ]:
        if arrays[key].shape != (expected_steps, 7):
            raise ValueError(
                f"Unexpected {key} shape in {trace_path}: "
                f"{arrays[key].shape}"
            )
    for key in ["rewards", "official_success", "done"]:
        if arrays[key].shape != (expected_steps,):
            raise ValueError(
                f"Unexpected {key} shape in {trace_path}: "
                f"{arrays[key].shape}"
            )

    source_success = bool(np.any(arrays["official_success"]))
    if source_success != expected_success:
        raise ValueError(
            f"Formal trace success mismatch: {trace_path}"
        )
    if expected_success and not bool(
        arrays["official_success"][-1]
    ):
        raise ValueError(
            f"Successful formal trace does not end on success: {trace_path}"
        )
    return arrays


def read_formal_rollouts(
    path: Path,
    manifest: dict,
) -> tuple[dict[tuple[str, int, int], dict], str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Formal rollout CSV not found: {path}"
        )

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])

    required_fields = {
        "protocol_version",
        "git_commit",
        "evaluator_sha256",
        "model_label",
        "heldout_task_id",
        "evaluation_role",
        "suite",
        "task_id",
        "init_index",
        "language",
        "success",
        "steps",
        "max_steps",
        "wait_steps",
        "n_action_steps",
        "env_seed",
        "policy_seed",
        "checkpoint_label",
        "checkpoint_sha256",
        "checkpoint",
        "action_trace",
    }
    if not required_fields <= fieldnames:
        raise ValueError(
            "Formal rollout CSV is missing required fields: "
            f"{sorted(required_fields - fieldnames)}"
        )
    if len(rows) != 200:
        raise ValueError(
            f"Expected 200 frozen formal rollouts, found {len(rows)}"
        )

    lookup = {}
    trace_paths_by_model = {}
    formal_git_commits = set()
    formal_evaluator_hashes = set()
    for row in rows:
        model_label = row["model_label"]
        if model_label not in MODEL_SPECS:
            raise ValueError(
                f"Unexpected formal model: {model_label}"
            )
        spec = MODEL_SPECS[model_label]
        task_id = int(row["task_id"])
        init_index = int(row["init_index"])
        key = (model_label, task_id, init_index)
        if key in lookup:
            raise ValueError(f"Duplicate formal rollout: {key}")

        role = (
            "heldout"
            if task_id == spec["heldout_task_id"]
            else "seen_control"
        )
        fixed = {
            "protocol_version": FORMAL_PROTOCOL_VERSION,
            "heldout_task_id": str(spec["heldout_task_id"]),
            "evaluation_role": role,
            "suite": SUITE_NAME,
            "max_steps": str(MAX_STEPS),
            "wait_steps": str(WAIT_STEPS),
            "n_action_steps": str(N_ACTION_STEPS),
            "env_seed": str(BASE_ENV_SEED + init_index),
            "policy_seed": str(BASE_POLICY_SEED + init_index),
        }
        for field, expected in fixed.items():
            if row[field] != expected:
                raise ValueError(
                    f"Formal rollout mismatch for {key}, {field}: "
                    f"{row[field]!r} != {expected!r}"
                )
        if int(row["checkpoint_label"]) != 90000:
            raise ValueError(
                f"Formal rollout is not checkpoint 090000: {key}"
            )

        success = parse_bool(row["success"], "formal success")
        steps = int(row["steps"])
        recorded_model = manifest["models"][model_label]
        if row["checkpoint"] != recorded_model["checkpoint"]:
            raise ValueError(
                f"Formal checkpoint path mismatch for {key}"
            )
        if (
            row["checkpoint_sha256"]
            != recorded_model["checkpoint_sha256"]
        ):
            raise ValueError(
                f"Formal checkpoint hash mismatch for {key}"
            )

        formal_git_commits.add(row["git_commit"])
        formal_evaluator_hashes.add(row["evaluator_sha256"])
        trace_path = resolve_repo_path(row["action_trace"])
        arrays = validate_source_trace(
            trace_path,
            steps,
            success,
        )

        lookup[key] = {
            "success": success,
            "steps": steps,
            "role": role,
            "language": row["language"],
            "formal_git_commit": row["git_commit"],
            "formal_evaluator_sha256": row["evaluator_sha256"],
            "checkpoint_label": row["checkpoint_label"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "checkpoint": row["checkpoint"],
            "trace_path": trace_path,
            "trace_sha256": sha256_file(trace_path),
            "trace": arrays,
        }
        trace_paths_by_model.setdefault(
            model_label,
            [],
        ).append(trace_path)

    expected_keys = {
        (model_label, task_id, init_index)
        for model_label in MODEL_SPECS
        for task_id in [3, 6]
        for init_index in range(50)
    }
    if set(lookup) != expected_keys:
        raise ValueError("Formal rollout matrix is incomplete")
    if formal_git_commits != {manifest.get("evaluation_commit")}:
        raise ValueError(
            "Formal rollout Git commit does not match its manifest"
        )
    if formal_evaluator_hashes != {manifest.get("evaluator_sha256")}:
        raise ValueError(
            "Formal evaluator hash does not match its manifest"
        )

    for model_label, trace_paths in trace_paths_by_model.items():
        if len(trace_paths) != 100:
            raise ValueError(
                f"Expected 100 traces for {model_label}"
            )
        roots = {path.parent for path in trace_paths}
        if len(roots) != 1:
            raise ValueError(
                f"Mixed trace directories for {model_label}"
            )
        root = next(iter(roots))
        bundle_sha256 = sha256_file_set(trace_paths, root)
        recorded = manifest["models"][model_label]
        if int(recorded["action_trace_count"]) != 100:
            raise ValueError(
                f"Manifest trace count mismatch for {model_label}"
            )
        if (
            bundle_sha256
            != recorded["action_trace_bundle_sha256"]
        ):
            raise ValueError(
                f"Formal action trace bundle hash mismatch for "
                f"{model_label}"
            )
        for key, value in lookup.items():
            if key[0] == model_label:
                value["trace_bundle_sha256"] = bundle_sha256

    return lookup, sha256_file(path)


def validate_selection_outcomes(
    selection: list[dict],
    formal_lookup: dict[tuple[str, int, int], dict],
):
    for case in selection:
        task_id = case["task_id"]
        init_index = case["init_index"]
        seen = formal_lookup[
            (case["seen_model"], task_id, init_index)
        ]["success"]
        heldout = formal_lookup[
            (case["heldout_model"], task_id, init_index)
        ]["success"]

        if case["stratum"] == "seen_success_heldout_failure":
            expected = (True, False)
        elif case["stratum"] == "both_failure":
            expected = (False, False)
        else:
            raise ValueError(
                f"Unexpected diagnostic stratum: {case['stratum']}"
            )

        if (seen, heldout) != expected:
            raise ValueError(
                "Diagnostic selection stratum does not match frozen "
                f"formal outcomes for task={task_id}, init={init_index}: "
                f"seen={seen}, heldout={heldout}, "
                f"stratum={case['stratum']}"
            )


def seed_environment(env, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if hasattr(env, "seed"):
        env.seed(seed)


def seed_replay_policy_phase(seed: int):
    """Match the Python/NumPy policy-phase seeds from the formal run."""
    random.seed(seed)
    np.random.seed(seed)


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


def initialize_episode(env, initial_state, env_seed: int):
    seed_environment(env, env_seed)
    obs = env.reset()
    updated_obs = env.set_init_state(
        np.asarray(initial_state, dtype=np.float64)
    )
    if updated_obs is not None:
        obs = updated_obs

    for wait_index in range(WAIT_STEPS):
        obs, _, done, _ = env.step(OFFICIAL_DUMMY_ACTION)
        if done:
            raise RuntimeError(
                "Environment ended during stabilization step "
                f"{wait_index}"
            )

    if bool(env.check_success()):
        raise RuntimeError(
            "Benchmark initialization already satisfies the task goal"
        )
    return obs


def descendant_body_ids(model, root_body_id: int) -> set[int]:
    parents = np.asarray(model.body_parentid, dtype=np.int64)
    descendants = {int(root_body_id)}
    changed = True
    while changed:
        changed = False
        for body_id, parent_id in enumerate(parents):
            if (
                body_id not in descendants
                and int(parent_id) in descendants
            ):
                descendants.add(body_id)
                changed = True
    return descendants


def collision_geom_names_for_root_body(
    env,
    object_name: str,
) -> list[str]:
    inner = env.env
    if object_name not in inner.obj_body_id:
        raise KeyError(
            f"Object root body is unavailable: {object_name}"
        )

    model = inner.sim.model
    root_body_id = int(inner.obj_body_id[object_name])
    body_ids = descendant_body_ids(model, root_body_id)
    geom_body_ids = np.asarray(model.geom_bodyid, dtype=np.int64)
    geom_contype = np.asarray(model.geom_contype, dtype=np.int64)
    geom_conaffinity = np.asarray(
        model.geom_conaffinity,
        dtype=np.int64,
    )

    collision_names = []
    for geom_id, body_id in enumerate(geom_body_ids):
        if int(body_id) not in body_ids:
            continue
        name = model.geom_id2name(geom_id)
        if not name:
            continue
        if (
            int(geom_contype[geom_id]) != 0
            or int(geom_conaffinity[geom_id]) != 0
        ):
            collision_names.append(name)

    if not collision_names:
        raise RuntimeError(
            f"No named collision geoms found below root body "
            f"{object_name!r}"
        )
    return sorted(set(collision_names))


def validate_geom_names(model, names: list[str], label: str):
    if not names:
        raise ValueError(f"No geom names configured for {label}")
    for name in names:
        try:
            geom_id = int(model.geom_name2id(name))
        except Exception as error:
            raise ValueError(
                f"Could not resolve geom {name!r} for {label}"
            ) from error
        if geom_id < 0:
            raise ValueError(
                f"Could not resolve geom {name!r} for {label}"
            )


def contact_geometry(env) -> dict[str, list[str]]:
    inner = env.env
    gripper = inner.robots[0].gripper
    important = gripper.important_geoms
    left = list(important.get("left_fingerpad", []))
    right = list(important.get("right_fingerpad", []))
    bowl = collision_geom_names_for_root_body(
        env,
        "akita_black_bowl_1",
    )

    validate_geom_names(inner.sim.model, left, "left fingerpad")
    validate_geom_names(inner.sim.model, right, "right fingerpad")
    validate_geom_names(inner.sim.model, bowl, "target bowl")
    return {
        "left_fingerpad": left,
        "right_fingerpad": right,
        "target_bowl": bowl,
    }


def as_uint8_rgb(image) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"Expected HWC RGB image, got shape {array.shape}"
        )
    if np.issubdtype(array.dtype, np.floating):
        if float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.rint(array)
    return np.clip(array, 0, 255).astype(np.uint8)


def side_by_side_frame(obs) -> np.ndarray:
    agent = as_uint8_rgb(obs["agentview_image"])
    wrist = as_uint8_rgb(obs["robot0_eye_in_hand_image"])
    if agent.shape != wrist.shape:
        raise ValueError(
            "Agent and wrist images have different shapes: "
            f"{agent.shape} != {wrist.shape}"
        )
    return np.concatenate([agent, wrist], axis=1)


def vector(obs, key: str, expected_size: int) -> np.ndarray:
    value = np.asarray(obs[key], dtype=np.float64).reshape(-1)
    if value.shape != (expected_size,):
        raise ValueError(
            f"Expected {key} shape ({expected_size},), got {value.shape}"
        )
    return value


def make_step_row(
    env,
    obs,
    step_index: int,
    source_trace: dict[str, np.ndarray],
    replay_reward: float,
    replay_done: bool,
    replay_success: bool,
    initial_bowl_z: float,
    contact_geoms: dict[str, list[str]],
) -> dict:
    inner = env.env
    bowl_state = inner.object_states_dict["akita_black_bowl_1"]
    cookies_state = inner.object_states_dict["cookies_1"]
    plate_state = inner.object_states_dict["plate_1"]

    joint = vector(obs, "robot0_joint_pos", 7)
    eef = vector(obs, "robot0_eef_pos", 3)
    gripper_qpos = vector(obs, "robot0_gripper_qpos", 2)
    gripper_qvel = vector(obs, "robot0_gripper_qvel", 2)
    bowl = vector(obs, "akita_black_bowl_1_pos", 3)
    cookies = vector(obs, "cookies_1_pos", 3)
    plate = vector(obs, "plate_1_pos", 3)

    raw_action = source_trace["raw_actions"][step_index]
    processed_action = source_trace["processed_actions"][step_index]
    applied_action = source_trace["applied_actions"][step_index]
    source_reward = float(source_trace["rewards"][step_index])
    bowl_lift = float(bowl[2] - initial_bowl_z)
    eef_to_bowl = float(np.linalg.norm(eef - bowl))
    bowl_to_plate_xy = float(
        np.linalg.norm(bowl[:2] - plate[:2])
    )
    bowl_to_plate = float(np.linalg.norm(bowl - plate))
    left_contact = bool(
        inner.check_contact(
            contact_geoms["left_fingerpad"],
            contact_geoms["target_bowl"],
        )
    )
    right_contact = bool(
        inner.check_contact(
            contact_geoms["right_fingerpad"],
            contact_geoms["target_bowl"],
        )
    )
    grasp_contact_proxy = bool(
        inner._check_grasp(
            inner.robots[0].gripper,
            contact_geoms["target_bowl"],
        )
    )
    if grasp_contact_proxy != (left_contact and right_contact):
        raise RuntimeError(
            "Robosuite grasp proxy disagrees with bilateral "
            f"fingerpad contact at action step {step_index + 1}"
        )
    bowl_ontop_plate = bool(
        plate_state.check_ontop(bowl_state)
    )
    if bowl_ontop_plate != bool(replay_success):
        raise RuntimeError(
            "Manual bowl-on-plate predicate disagrees with official "
            f"success at action step {step_index + 1}"
        )

    row = {
        "step": step_index + 1,
        "source_reward": source_reward,
        "replay_reward": float(replay_reward),
        "reward_abs_diff": abs(source_reward - float(replay_reward)),
        "source_done": bool(source_trace["done"][step_index]),
        "replay_done": bool(replay_done),
        "source_official_success": bool(
            source_trace["official_success"][step_index]
        ),
        "replay_official_success": bool(replay_success),
        "eef_x": eef[0],
        "eef_y": eef[1],
        "eef_z": eef[2],
        "gripper_qpos_0": gripper_qpos[0],
        "gripper_qpos_1": gripper_qpos[1],
        "gripper_qvel_0": gripper_qvel[0],
        "gripper_qvel_1": gripper_qvel[1],
        "bowl_x": bowl[0],
        "bowl_y": bowl[1],
        "bowl_z": bowl[2],
        "cookies_x": cookies[0],
        "cookies_y": cookies[1],
        "cookies_z": cookies[2],
        "plate_x": plate[0],
        "plate_y": plate[1],
        "plate_z": plate[2],
        "eef_to_bowl_distance_m": eef_to_bowl,
        "bowl_to_cookies_xy_distance_m": np.linalg.norm(
            bowl[:2] - cookies[:2]
        ),
        "bowl_to_cookies_distance_m": np.linalg.norm(bowl - cookies),
        "bowl_to_plate_xy_distance_m": bowl_to_plate_xy,
        "bowl_to_plate_distance_m": bowl_to_plate,
        "bowl_lift_m": bowl_lift,
        "bowl_lifted": bowl_lift >= LIFT_THRESHOLD_M,
        "eef_within_5cm_of_bowl": (
            eef_to_bowl <= APPROACH_THRESHOLD_M
        ),
        "bowl_within_8cm_xy_of_plate": (
            bowl_to_plate_xy <= NEAR_PLATE_XY_THRESHOLD_M
        ),
        "left_fingerpad_contact": left_contact,
        "right_fingerpad_contact": right_contact,
        "grasp_contact_proxy": grasp_contact_proxy,
        "bowl_contact_cookies": bool(
            bowl_state.check_contact(cookies_state)
        ),
        "bowl_contact_plate": bool(
            bowl_state.check_contact(plate_state)
        ),
        "bowl_ontop_plate": bowl_ontop_plate,
    }

    for prefix, action in [
        ("raw_action", raw_action),
        ("processed_action", processed_action),
        ("applied_action", applied_action),
    ]:
        for index, value in enumerate(action):
            row[f"{prefix}_{index}"] = float(value)
    for index, value in enumerate(joint):
        row[f"robot_joint_{index}"] = float(value)
    return row


def fsync_file(path: Path):
    with path.open("rb") as file:
        os.fsync(file.fileno())


def write_video(path: Path, frames: list[np.ndarray]):
    if not frames:
        raise ValueError("Cannot encode an empty diagnostic video")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(
        f"{path.stem}.tmp{path.suffix}"
    )
    temporary_path.unlink(missing_ok=True)

    writer = None
    try:
        writer = imageio.get_writer(
            temporary_path,
            fps=VIDEO_FPS,
            codec="libx264",
            quality=7,
            macro_block_size=None,
        )
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        writer = None
        temporary_path.replace(path)
        fsync_file(path)
    except BaseException:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise


def replay_episode(
    env,
    initial_state,
    env_seed: int,
    policy_seed: int,
    source_trace: dict[str, np.ndarray],
    video_path: Path,
) -> dict:
    obs = initialize_episode(env, initial_state, env_seed)
    seed_replay_policy_phase(policy_seed)
    contact_geoms = contact_geometry(env)
    initial_eef = vector(obs, "robot0_eef_pos", 3)
    initial_bowl = vector(obs, "akita_black_bowl_1_pos", 3)
    initial_plate = vector(obs, "plate_1_pos", 3)
    initial_bowl_z = float(initial_bowl[2])
    initial_eef_to_bowl = float(
        np.linalg.norm(initial_eef - initial_bowl)
    )
    initial_bowl_to_plate_xy = float(
        np.linalg.norm(initial_bowl[:2] - initial_plate[:2])
    )
    initial_bowl_to_plate = float(
        np.linalg.norm(initial_bowl - initial_plate)
    )

    video_frames = [side_by_side_frame(obs)]
    step_rows = []
    replay_total_reward = 0.0
    first_replay_success_step = None

    for step_index, applied_action in enumerate(
        source_trace["applied_actions"]
    ):
        obs, reward, done, _ = env.step(
            np.asarray(applied_action, dtype=np.float32)
        )
        replay_success = bool(env.check_success())
        source_reward = float(source_trace["rewards"][step_index])
        source_done = bool(source_trace["done"][step_index])
        source_success = bool(
            source_trace["official_success"][step_index]
        )
        reward_abs_diff = abs(float(reward) - source_reward)

        if reward_abs_diff > REWARD_ATOL:
            raise RuntimeError(
                "Replay reward diverged from the frozen formal trace "
                f"at action step {step_index + 1}: "
                f"{float(reward)} != {source_reward}"
            )
        if bool(done) != source_done:
            raise RuntimeError(
                "Replay done flag diverged from the frozen formal trace "
                f"at action step {step_index + 1}"
            )
        if replay_success != source_success:
            raise RuntimeError(
                "Replay official success diverged from the frozen "
                f"formal trace at action step {step_index + 1}"
            )

        replay_total_reward += float(reward)
        if replay_success and first_replay_success_step is None:
            first_replay_success_step = step_index + 1

        step_rows.append(
            make_step_row(
                env=env,
                obs=obs,
                step_index=step_index,
                source_trace=source_trace,
                replay_reward=float(reward),
                replay_done=bool(done),
                replay_success=replay_success,
                initial_bowl_z=initial_bowl_z,
                contact_geoms=contact_geoms,
            )
        )
        video_frames.append(side_by_side_frame(obs))

    replay_success = any(
        bool(row["replay_official_success"]) for row in step_rows
    )
    expected_steps = len(source_trace["rewards"])
    expected_success = bool(
        np.any(source_trace["official_success"])
    )
    if len(step_rows) != expected_steps:
        raise RuntimeError(
            "Replay step count diverged from the frozen formal trace: "
            f"{len(step_rows)} != {expected_steps}"
        )
    if replay_success != expected_success:
        raise RuntimeError(
            "Replay final success diverged from the frozen formal trace"
        )
    if len(video_frames) != expected_steps + 1:
        raise RuntimeError(
            "Diagnostic video frame count is not N+1 before encoding"
        )

    video_path.unlink(missing_ok=True)
    write_video(video_path, video_frames)

    return {
        "success": replay_success,
        "steps": len(step_rows),
        "first_success_step": first_replay_success_step,
        "total_reward": replay_total_reward,
        "terminal_flags_match": True,
        "max_reward_abs_diff": max(
            float(row["reward_abs_diff"]) for row in step_rows
        ),
        "step_rows": step_rows,
        "video_frames": len(video_frames),
        "initial_bowl_z": initial_bowl_z,
        "initial_eef_to_bowl_distance_m": initial_eef_to_bowl,
        "initial_bowl_to_plate_xy_distance_m": (
            initial_bowl_to_plate_xy
        ),
        "initial_bowl_to_plate_distance_m": (
            initial_bowl_to_plate
        ),
        "bowl_collision_geoms": contact_geoms["target_bowl"],
    }


def write_step_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.unlink(missing_ok=True)

    with gzip.open(
        temporary_path,
        "wt",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=STEP_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    temporary_path.replace(path)
    fsync_file(path)


def validate_step_csv(
    path: Path,
    expected_rows: int,
    expected_sha256: str,
) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing diagnostic step CSV: {path}"
        )
    if sha256_file(path) != expected_sha256:
        raise ValueError(
            f"Diagnostic step CSV hash mismatch: {path}"
        )

    with gzip.open(
        path,
        "rt",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != STEP_FIELDS:
            raise ValueError(
                f"Diagnostic step CSV header mismatch: {path}"
            )
        rows = list(reader)
    if len(rows) != expected_rows:
        raise ValueError(
            f"Diagnostic step CSV row mismatch: {path}"
        )
    return rows


def video_frame_count(path: Path) -> int:
    reader = imageio.get_reader(path)
    try:
        first = reader.get_data(0)
        if first.ndim != 3 or first.shape[2] != 3:
            raise ValueError(
                f"Unexpected diagnostic video frame shape: {first.shape}"
            )
        try:
            return int(reader.count_frames())
        except Exception:
            pass
    finally:
        reader.close()

    reader = imageio.get_reader(path)
    try:
        return sum(1 for _ in reader)
    finally:
        reader.close()


def validate_video(
    path: Path,
    expected_frames: int,
    expected_sha256: str,
):
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Missing or empty diagnostic video: {path}"
        )
    if sha256_file(path) != expected_sha256:
        raise ValueError(
            f"Diagnostic video hash mismatch: {path}"
        )
    reader = imageio.get_reader(path)
    try:
        first = reader.get_data(0)
        if tuple(first.shape) != VIDEO_FRAME_SHAPE:
            raise ValueError(
                f"Diagnostic video shape mismatch: {path}: "
                f"{first.shape} != {VIDEO_FRAME_SHAPE}"
            )
        fps = float(reader.get_meta_data().get("fps", 0.0))
        if abs(fps - VIDEO_FPS) > 0.01:
            raise ValueError(
                f"Diagnostic video FPS mismatch: {path}: "
                f"{fps} != {VIDEO_FPS}"
            )
    finally:
        reader.close()
    actual_frames = video_frame_count(path)
    if actual_frames != expected_frames:
        raise ValueError(
            f"Diagnostic video frame mismatch: {path}: "
            f"{actual_frames} != {expected_frames}"
        )


def append_summary(path: Path, row: dict):
    existing_rows = []
    if path.exists():
        with path.open(
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != SUMMARY_FIELDS:
                raise ValueError(
                    "Existing diagnostic summary header mismatch"
                )
            existing_rows = list(reader)

    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.unlink(missing_ok=True)
    with temporary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=SUMMARY_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerow(row)
        file.flush()
        os.fsync(file.fileno())
    temporary_path.replace(path)
    fsync_file(path)


def first_true_step(step_rows: list[dict], field: str):
    for row in step_rows:
        if bool(row[field]):
            return int(row["step"])
    return None


def optional_number(value):
    return "" if value is None else value


def summary_metrics(result: dict) -> dict:
    rows = result["step_rows"]
    if not rows:
        raise ValueError("Diagnostic replay produced no action steps")
    final = rows[-1]
    initial_approached = (
        result["initial_eef_to_bowl_distance_m"]
        <= APPROACH_THRESHOLD_M
    )
    initial_near_plate = (
        result["initial_bowl_to_plate_xy_distance_m"]
        <= NEAR_PLATE_XY_THRESHOLD_M
    )

    return {
        "initial_bowl_z": result["initial_bowl_z"],
        "initial_eef_to_bowl_distance_m": (
            result["initial_eef_to_bowl_distance_m"]
        ),
        "initial_bowl_to_plate_xy_distance_m": (
            result["initial_bowl_to_plate_xy_distance_m"]
        ),
        "initial_bowl_to_plate_distance_m": (
            result["initial_bowl_to_plate_distance_m"]
        ),
        "ever_approached_bowl": (
            initial_approached
            or any(
                bool(row["eef_within_5cm_of_bowl"])
                for row in rows
            )
        ),
        "first_approach_step": (
            0
            if initial_approached
            else optional_number(
                first_true_step(
                    rows,
                    "eef_within_5cm_of_bowl",
                )
            )
        ),
        "ever_near_plate": (
            initial_near_plate
            or any(
                bool(row["bowl_within_8cm_xy_of_plate"])
                for row in rows
            )
        ),
        "first_near_plate_step": (
            0
            if initial_near_plate
            else optional_number(
                first_true_step(
                    rows,
                    "bowl_within_8cm_xy_of_plate",
                )
            )
        ),
        "ever_grasp_contact_proxy": any(
            bool(row["grasp_contact_proxy"]) for row in rows
        ),
        "first_grasp_contact_step": optional_number(
            first_true_step(rows, "grasp_contact_proxy")
        ),
        "ever_lifted": any(bool(row["bowl_lifted"]) for row in rows),
        "first_lift_step": optional_number(
            first_true_step(rows, "bowl_lifted")
        ),
        "max_bowl_lift_m": max(
            float(row["bowl_lift_m"]) for row in rows
        ),
        "ever_contact_cookies": any(
            bool(row["bowl_contact_cookies"]) for row in rows
        ),
        "first_contact_cookies_step": optional_number(
            first_true_step(rows, "bowl_contact_cookies")
        ),
        "ever_contact_plate": any(
            bool(row["bowl_contact_plate"]) for row in rows
        ),
        "first_contact_plate_step": optional_number(
            first_true_step(rows, "bowl_contact_plate")
        ),
        "ever_ontop_plate": any(
            bool(row["bowl_ontop_plate"]) for row in rows
        ),
        "first_ontop_plate_step": optional_number(
            first_true_step(rows, "bowl_ontop_plate")
        ),
        "min_eef_to_bowl_distance_m": min(
            [
                result["initial_eef_to_bowl_distance_m"],
                *[
                    float(row["eef_to_bowl_distance_m"])
                    for row in rows
                ],
            ]
        ),
        "min_bowl_to_plate_xy_distance_m": min(
            [
                result["initial_bowl_to_plate_xy_distance_m"],
                *[
                    float(row["bowl_to_plate_xy_distance_m"])
                    for row in rows
                ],
            ]
        ),
        "min_bowl_to_plate_distance_m": min(
            [
                result["initial_bowl_to_plate_distance_m"],
                *[
                    float(row["bowl_to_plate_distance_m"])
                    for row in rows
                ],
            ]
        ),
        "final_eef_x": final["eef_x"],
        "final_eef_y": final["eef_y"],
        "final_eef_z": final["eef_z"],
        "final_bowl_x": final["bowl_x"],
        "final_bowl_y": final["bowl_y"],
        "final_bowl_z": final["bowl_z"],
        "final_cookies_x": final["cookies_x"],
        "final_cookies_y": final["cookies_y"],
        "final_cookies_z": final["cookies_z"],
        "final_plate_x": final["plate_x"],
        "final_plate_y": final["plate_y"],
        "final_plate_z": final["plate_z"],
    }


def read_completed(
    summary_path: Path,
    expected_fixed: dict[str, str],
    formal_lookup: dict[tuple[str, int, int], dict],
    selection_lookup: dict[tuple[int, int], dict],
) -> set[tuple[str, int, int]]:
    completed = set()
    if not summary_path.exists():
        return completed

    with summary_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != SUMMARY_FIELDS:
            raise ValueError(
                "Existing diagnostic summary header mismatch"
            )
        rows = list(reader)

    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(
                "Existing diagnostic summary contains a malformed row"
            )
        for field, expected in expected_fixed.items():
            if row[field] != expected:
                raise ValueError(
                    f"Existing diagnostic summary mismatch for {field}: "
                    f"{row[field]!r} != {expected!r}"
                )

        model_label = row["model_label"]
        task_id = int(row["task_id"])
        init_index = int(row["init_index"])
        key = (model_label, task_id, init_index)
        pair = (task_id, init_index)
        if key in completed:
            raise ValueError(
                f"Duplicate diagnostic summary case: {key}"
            )
        if key not in formal_lookup:
            raise ValueError(
                f"Diagnostic summary has no formal outcome: {key}"
            )
        if pair not in selection_lookup:
            raise ValueError(
                f"Diagnostic summary case is outside selection: {key}"
            )

        formal = formal_lookup[key]
        selection = selection_lookup[pair]
        dynamic_expected = {
            "heldout_task_id": str(
                MODEL_SPECS[model_label]["heldout_task_id"]
            ),
            "evaluation_role": formal["role"],
            "stratum": selection["stratum"],
            "selection_seed": str(selection["selection_seed"]),
            "formal_git_commit": formal["formal_git_commit"],
            "formal_evaluator_sha256": (
                formal["formal_evaluator_sha256"]
            ),
            "expected_formal_success": str(formal["success"]),
            "expected_formal_steps": str(formal["steps"]),
            "env_seed": str(BASE_ENV_SEED + init_index),
            "policy_seed": str(BASE_POLICY_SEED + init_index),
            "checkpoint_sha256": formal["checkpoint_sha256"],
            "source_action_trace_bundle_sha256": (
                formal["trace_bundle_sha256"]
            ),
            "source_action_trace_sha256": formal["trace_sha256"],
            "source_action_trace": str(formal["trace_path"]),
        }
        for field, expected in dynamic_expected.items():
            if row[field] != expected:
                raise ValueError(
                    f"Existing diagnostic case mismatch for {field}: "
                    f"{row[field]!r} != {expected!r}"
                )

        replay_steps = int(row["replay_steps"])
        if int(row["step_csv_rows"]) != replay_steps:
            raise ValueError(
                f"Diagnostic step count mismatch: {key}"
            )
        if int(row["video_frames"]) != replay_steps + 1:
            raise ValueError(
                f"Diagnostic video count mismatch: {key}"
            )

        step_rows = validate_step_csv(
            Path(row["step_csv"]),
            replay_steps,
            row["step_csv_sha256"],
        )
        validate_video(
            Path(row["video"]),
            int(row["video_frames"]),
            row["video_sha256"],
        )

        replay_success = any(
            parse_bool(
                step_row["replay_official_success"],
                "step replay success",
            )
            for step_row in step_rows
        )
        if parse_bool(
            row["replay_success"],
            "replay success",
        ) != replay_success:
            raise ValueError(
                f"Replay success does not match step CSV: {key}"
            )
        success_matches = parse_bool(
            row["success_matches_formal"],
            "success match",
        )
        if success_matches != (
            replay_success == formal["success"]
        ):
            raise ValueError(
                f"Formal success match flag is inconsistent: {key}"
            )
        steps_match = parse_bool(
            row["steps_match_formal"],
            "steps match",
        )
        if steps_match != (replay_steps == formal["steps"]):
            raise ValueError(
                f"Formal step match flag is inconsistent: {key}"
            )

        terminal_flags_match = (
            replay_steps == formal["steps"]
            and all(
                parse_bool(
                    step_row["source_done"],
                    "source done",
                )
                == parse_bool(
                    step_row["replay_done"],
                    "replay done",
                )
                and parse_bool(
                    step_row["source_official_success"],
                    "source success",
                )
                == parse_bool(
                    step_row["replay_official_success"],
                    "replay success",
                )
                for step_row in step_rows
            )
        )
        recorded_terminal_match = parse_bool(
            row["terminal_flags_match_formal"],
            "terminal flags match",
        )
        if recorded_terminal_match != terminal_flags_match:
            raise ValueError(
                f"Terminal match flag is inconsistent: {key}"
            )
        max_reward_abs_diff = max(
            float(step_row["reward_abs_diff"])
            for step_row in step_rows
        )
        if abs(
            float(row["max_reward_abs_diff"])
            - max_reward_abs_diff
        ) > 1e-12:
            raise ValueError(
                f"Recorded reward difference is inconsistent: {key}"
            )
        if max_reward_abs_diff > REWARD_ATOL:
            raise ValueError(
                f"Completed replay reward parity failed: {key}"
            )
        if not (
            success_matches
            and steps_match
            and recorded_terminal_match
        ):
            raise ValueError(
                f"Completed replay does not match formal trace: {key}"
            )
        for step_row in step_rows:
            left = parse_bool(
                step_row["left_fingerpad_contact"],
                "left fingerpad contact",
            )
            right = parse_bool(
                step_row["right_fingerpad_contact"],
                "right fingerpad contact",
            )
            grasp = parse_bool(
                step_row["grasp_contact_proxy"],
                "grasp contact proxy",
            )
            if grasp != (left and right):
                raise ValueError(
                    f"Completed replay grasp proxy is inconsistent: {key}"
                )
            if parse_bool(
                step_row["bowl_ontop_plate"],
                "bowl on plate",
            ) != parse_bool(
                step_row["replay_official_success"],
                "replay official success",
            ):
                raise ValueError(
                    f"Completed replay goal predicate is inconsistent: "
                    f"{key}"
                )
        completed.add(key)
    return completed


def main():
    args = parse_args()
    validate_args(args)

    selection_path = resolve_repo_path(args.selection_csv)
    formal_path = resolve_repo_path(args.formal_rollouts_csv)
    manifest_path = resolve_repo_path(args.formal_manifest)
    selection = read_selection(selection_path)
    cases = selected_cases(args, selection)
    models = selected_models(args)
    selection_lookup = {
        (row["task_id"], row["init_index"]): row
        for row in selection
    }

    manifest, formal_manifest_sha256 = load_manifest(
        manifest_path,
        formal_path,
    )
    formal_lookup, formal_rollouts_sha256 = read_formal_rollouts(
        formal_path,
        manifest,
    )
    validate_selection_outcomes(selection, formal_lookup)

    results_dir = resolve_repo_path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        results_dir
        / f"v4_loco_{args.mode}_diagnostic_summary.csv"
    )
    steps_dir = (
        results_dir / f"v4_loco_{args.mode}_diagnostic_steps"
    )
    videos_dir = (
        results_dir / f"v4_loco_{args.mode}_diagnostic_videos"
    )

    git_commit = current_git_commit()
    if args.mode == "diagnostic" and git_commit == "unknown":
        raise RuntimeError(
            "Full diagnostics require a Git commit"
        )
    evaluator_sha256 = evaluator_fingerprint()
    selection_sha256 = sha256_file(selection_path)
    expected_fixed = {
        "diagnostic_protocol_version": DIAGNOSTIC_PROTOCOL_VERSION,
        "git_commit": git_commit,
        "evaluator_sha256": evaluator_sha256,
        "evaluation_mode": args.mode,
        "suite": SUITE_NAME,
        "selection_sha256": selection_sha256,
        "formal_rollouts_sha256": formal_rollouts_sha256,
        "formal_manifest_sha256": formal_manifest_sha256,
        "formal_protocol_version": FORMAL_PROTOCOL_VERSION,
        "max_steps": str(MAX_STEPS),
        "wait_steps": str(WAIT_STEPS),
        "n_action_steps": str(N_ACTION_STEPS),
        "diagnostic_only": "True",
    }
    completed = read_completed(
        summary_path,
        expected_fixed,
        formal_lookup,
        selection_lookup,
    )

    requested = {
        (model_label, case["task_id"], case["init_index"])
        for model_label in models
        for case in cases
    }
    if not completed <= requested:
        raise ValueError(
            "Existing diagnostic summary contains cases outside this run"
        )

    suite = benchmark.get_benchmark(SUITE_NAME)()
    tasks = {
        task_id: suite.get_task(task_id)
        for task_id in sorted({case["task_id"] for case in cases})
    }
    init_states = {
        task_id: suite.get_task_init_states(task_id)
        for task_id in tasks
    }

    jobs = [
        (model_label, case)
        for model_label in models
        for case in cases
    ]
    print("=" * 80)
    print("Protocol:", DIAGNOSTIC_PROTOCOL_VERSION)
    print("Mode:", args.mode)
    print("Git commit:", git_commit)
    print("Models:", models)
    print("Selected states:", len(cases))
    print("Replay jobs:", len(jobs))
    print("Max steps:", MAX_STEPS)
    print("Wait steps:", WAIT_STEPS)
    print("Action steps:", N_ACTION_STEPS)
    print("Already complete:", len(completed))
    print(
        "Action source: exact NPZ traces from the frozen formal run"
    )
    print(
        "NOTE: diagnostic replays do not replace formal success rates."
    )

    environments = {}
    completed_this_run = 0
    try:
        for job_index, (model_label, case) in enumerate(
            jobs,
            start=1,
        ):
            task_id = case["task_id"]
            init_index = case["init_index"]
            key = (model_label, task_id, init_index)
            if key in completed:
                print(
                    f"[skip] model={model_label} "
                    f"task={task_id} init={init_index:02d}"
                )
                continue

            if task_id not in environments:
                environments[task_id] = make_environment(tasks[task_id])
            env = environments[task_id]
            states = init_states[task_id]
            if states is None or not 0 <= init_index < len(states):
                raise IndexError(
                    f"Invalid benchmark state for selected case: {key}"
                )

            formal = formal_lookup[key]
            stem = f"task{task_id}_init{init_index:02d}"
            step_csv_path = (
                steps_dir
                / model_label
                / f"{stem}.csv.gz"
            )
            video_path = (
                videos_dir
                / model_label
                / f"{stem}.mp4"
            )
            env_seed = BASE_ENV_SEED + init_index
            policy_seed = BASE_POLICY_SEED + init_index

            result = replay_episode(
                env=env,
                initial_state=states[init_index],
                env_seed=env_seed,
                policy_seed=policy_seed,
                source_trace=formal["trace"],
                video_path=video_path,
            )
            success_matches = (
                result["success"] == formal["success"]
            )
            steps_match = result["steps"] == formal["steps"]
            parity_valid = (
                success_matches
                and steps_match
                and result["terminal_flags_match"]
                and result["max_reward_abs_diff"] <= REWARD_ATOL
            )
            if not parity_valid:
                video_path.unlink(missing_ok=True)
                raise RuntimeError(
                    "Diagnostic replay failed formal parity for "
                    f"model={model_label}, task={task_id}, "
                    f"init={init_index}"
                )

            write_step_csv(step_csv_path, result["step_rows"])

            step_csv_sha256 = sha256_file(step_csv_path)
            video_sha256 = sha256_file(video_path)
            validate_step_csv(
                step_csv_path,
                result["steps"],
                step_csv_sha256,
            )
            validate_video(
                video_path,
                result["video_frames"],
                video_sha256,
            )

            metrics = summary_metrics(result)
            spec = MODEL_SPECS[model_label]

            row = {
                **expected_fixed,
                "model_label": model_label,
                "heldout_task_id": spec["heldout_task_id"],
                "evaluation_role": formal["role"],
                "task_id": task_id,
                "init_index": init_index,
                "language": formal["language"],
                "stratum": case["stratum"],
                "selection_seed": case["selection_seed"],
                "formal_git_commit": formal["formal_git_commit"],
                "formal_evaluator_sha256": (
                    formal["formal_evaluator_sha256"]
                ),
                "expected_formal_success": formal["success"],
                "expected_formal_steps": formal["steps"],
                "replay_success": result["success"],
                "success_matches_formal": success_matches,
                "replay_steps": result["steps"],
                "steps_match_formal": steps_match,
                "terminal_flags_match_formal": (
                    result["terminal_flags_match"]
                ),
                "max_reward_abs_diff": result["max_reward_abs_diff"],
                "first_replay_success_step": optional_number(
                    result["first_success_step"]
                ),
                "replay_total_reward": result["total_reward"],
                "env_seed": env_seed,
                "policy_seed": policy_seed,
                "checkpoint_label": formal["checkpoint_label"],
                "checkpoint_sha256": formal["checkpoint_sha256"],
                "checkpoint": formal["checkpoint"],
                "source_action_trace_bundle_sha256": (
                    formal["trace_bundle_sha256"]
                ),
                "source_action_trace_sha256": formal["trace_sha256"],
                "source_action_trace": str(formal["trace_path"]),
                "bowl_collision_geom_count": len(
                    result["bowl_collision_geoms"]
                ),
                "bowl_collision_geoms": "|".join(
                    result["bowl_collision_geoms"]
                ),
                **metrics,
                "step_csv_rows": result["steps"],
                "video_frames": result["video_frames"],
                "step_csv_sha256": step_csv_sha256,
                "video_sha256": video_sha256,
                "step_csv": str(step_csv_path.resolve()),
                "video": str(video_path.resolve()),
            }
            append_summary(summary_path, row)
            completed.add(key)
            completed_this_run += 1

            print(
                f"[{job_index:02d}/{len(jobs):02d}] "
                f"model={model_label} task={task_id} "
                f"init={init_index:02d} role={formal['role']} "
                f"expected={formal['success']} "
                f"replay={result['success']} "
                f"match={success_matches} steps={result['steps']} "
                f"terminal_match={result['terminal_flags_match']} "
                f"approach={metrics['ever_approached_bowl']} "
                f"grasp={metrics['ever_grasp_contact_proxy']} "
                f"lift={metrics['ever_lifted']} "
                f"plate={metrics['ever_contact_plate']}"
            )
    finally:
        for env in environments.values():
            env.close()

    if args.mode == "diagnostic":
        if len(requested) != 40 or len(completed) != 40:
            raise RuntimeError(
                "Full diagnostic mode did not complete the frozen "
                f"40 replays: requested={len(requested)}, "
                f"completed={len(completed)}"
            )

    print("=" * 80)
    print("Diagnostic replay complete")
    print("Completed this run:", completed_this_run)
    print("Total summary rows:", len(completed))
    print("Summary:", summary_path)
    print("Step CSVs:", steps_dir)
    print("Videos:", videos_dir)
    print(
        "Reminder: these selected post-hoc replays do not replace "
        "the frozen 200-rollout formal evaluation."
    )


if __name__ == "__main__":
    main()
