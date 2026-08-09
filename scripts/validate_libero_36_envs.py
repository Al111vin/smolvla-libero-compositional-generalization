from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import os
import random
import subprocess
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glx")

import numpy as np
from robosuite.utils.errors import RandomizationError

from libero.libero.envs import OffScreenRenderEnv

try:
    import libero_36_camera as camera_config
except ModuleNotFoundError:
    from scripts import libero_36_camera as camera_config


PROTOCOL_VERSION = "libero_36_proxy_tabletop_draft_v5"
BENCHMARK_NAME = "libero_registered_object_36_proxy_tabletop"
WORKSPACE_FIXTURE = "main_table"
DEFAULT_LAYOUT_SPEC = Path("data/libero_36/layout_spec.csv")
DEFAULT_OUTPUT = Path("results/libero_36_reset_audit.csv")
FULL_RESETS_PER_LAYOUT = 5
FULL_SETTLE_STEPS = 20
FULL_SEED_BASE = 360000
FULL_MAX_RESET_ATTEMPTS = 100
FULL_MAX_MANIPULABLE_HORIZONTAL_DRIFT_M = 0.02
FULL_MAX_RECEIVER_HORIZONTAL_DRIFT_M = 0.02
FULL_MAX_ORIENTATION_DRIFT_DEG = 15.0
FULL_FINAL_WINDOW_STEPS = 5
FULL_MAX_FINAL_WINDOW_MOTION_M = 0.001
FULL_MAX_FINAL_WINDOW_ORIENTATION_DEG = 1.0

OBJECT_TYPES = [
    "akita_black_bowl",
    "white_yellow_mug",
    "alphabet_soup",
    "cream_cheese",
]
MANIPULABLES = [f"{name}_1" for name in OBJECT_TYPES]
DISPLAY_NAMES = {
    "akita_black_bowl": "akita black bowl",
    "white_yellow_mug": "yellow and white mug",
    "alphabet_soup": "alphabet soup",
    "cream_cheese": "cream cheese box",
}
SOURCE_SAMPLER_RADII_M = {
    "akita_black_bowl": 0.025,
    "white_yellow_mug": 0.025,
    "alphabet_soup": 0.025,
    "cream_cheese": 0.030,
}

SOURCE_SLOT_RANGES = {
    0: (0.04, -0.244, 0.11, -0.182),
    1: (0.04, -0.102, 0.11, -0.040),
    2: (0.04, 0.040, 0.11, 0.102),
    3: (0.04, 0.182, 0.11, 0.244),
}
TARGET_REGION_RANGES = {
    "left": (-0.20, -0.28, -0.04, -0.14),
    "middle": (-0.20, -0.07, -0.04, 0.07),
    "right": (-0.20, 0.14, -0.04, 0.28),
}
TARGET_REGION_RGBA = (0.10, 0.45, 1.00, 0.25)

LAYOUT_SPEC_FIELDS = [
    "protocol_version",
    "benchmark_name",
    "task_id",
    "layout_id",
    "tuple",
    "object",
    "skill",
    "spatial_region",
    "language",
    "bddl_path",
    "bddl_sha256",
    "object_to_slot_json",
    "goal_expression",
    "goal_predicates_json",
    "receiver_region",
    "source_slot_ranges_json",
    "target_region_ranges_json",
    camera_config.OBSERVATION_CAMERA_SPEC_FIELD,
]

OUTPUT_FIELDS = [
    "protocol_version",
    "benchmark_name",
    "mode",
    "task_id",
    "layout_id",
    "reset_index",
    "seed",
    "reset_attempts",
    "tuple",
    "object",
    "skill",
    "spatial_region",
    "bddl_path",
    "bddl_sha256",
    "object_to_slot_json",
    "frozen_camera_configuration_ok",
    "agentview_camera_record_json",
    "preview_artifacts_json",
    "parsed_semantics_ok",
    "initial_goal_false",
    "goal_became_true_at_step",
    "settled_goal_false",
    "initial_source_regions_ok",
    "initial_source_official_predicates_ok",
    "initial_source_xy_bounds_ok",
    "initial_source_region_details_json",
    "settled_source_regions_ok",
    "settled_source_official_predicates_ok",
    "settled_source_xy_bounds_ok",
    "settled_source_region_details_json",
    "all_settle_source_regions_ok",
    "all_settle_source_official_predicates_ok",
    "all_settle_source_xy_bounds_ok",
    "final_window_source_regions_ok",
    "final_window_source_official_predicates_ok",
    "final_window_source_xy_bounds_ok",
    "initial_receiver_region_ok",
    "initial_receiver_official_predicate_ok",
    "initial_receiver_xy_bounds_ok",
    "initial_receiver_region_details_json",
    "settled_receiver_region_ok",
    "settled_receiver_official_predicate_ok",
    "settled_receiver_xy_bounds_ok",
    "settled_receiver_region_details_json",
    "all_settle_receiver_region_ok",
    "all_settle_receiver_official_predicate_ok",
    "all_settle_receiver_xy_bounds_ok",
    "final_window_receiver_region_ok",
    "final_window_receiver_official_predicate_ok",
    "final_window_receiver_xy_bounds_ok",
    "initial_images_ok",
    "settled_images_ok",
    "initial_finite_poses",
    "settled_finite_poses",
    "initial_unexpected_contact_count",
    "initial_unexpected_contact_pairs_json",
    "settled_unexpected_contact_count",
    "settled_unexpected_contact_pairs_json",
    "ever_unexpected_contact_count",
    "ever_unexpected_contact_pairs_json",
    "initial_poses_json",
    "settled_poses_json",
    "per_object_drift_m_json",
    "max_manipulable_drift_m",
    "per_object_horizontal_drift_m_json",
    "max_manipulable_horizontal_drift_m",
    "per_object_vertical_displacement_m_json",
    "per_object_orientation_drift_deg_json",
    "max_manipulable_orientation_drift_deg",
    "receiver_drift_m",
    "receiver_horizontal_drift_m",
    "receiver_vertical_displacement_m",
    "receiver_orientation_drift_deg",
    "max_final_window_motion_m",
    "max_final_window_orientation_deg",
    "determinism_checked",
    "determinism_initial_state_max_abs",
    "determinism_settled_state_max_abs",
    "deterministic",
    "failed_gate_checks_json",
    "passed",
    "error",
]


class ResetAttemptsExceeded(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Parse, reset, passively settle, and validate the generated "
            "LIBERO 36 registered-object proxy benchmark."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        required=True,
        help=(
            "smoke permits filters and short checks; full freezes the "
            "36x4x5=720-reset protocol gate."
        ),
    )
    parser.add_argument(
        "--layout-spec",
        type=Path,
        default=DEFAULT_LAYOUT_SPEC,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--task-ids", type=int, nargs="*")
    parser.add_argument("--layout-ids", type=int, nargs="*")
    parser.add_argument("--resets-per-layout", type=int)
    parser.add_argument("--settle-steps", type=int)
    parser.add_argument(
        "--seed-base",
        type=int,
        default=FULL_SEED_BASE,
    )
    parser.add_argument(
        "--max-reset-attempts",
        type=int,
        default=FULL_MAX_RESET_ATTEMPTS,
    )
    parser.add_argument(
        "--max-manipulable-horizontal-drift-m",
        "--max-manipulable-drift-m",
        dest="max_manipulable_horizontal_drift_m",
        type=float,
        default=FULL_MAX_MANIPULABLE_HORIZONTAL_DRIFT_M,
        help=(
            "Maximum raw-reset to settled XY drift. The legacy option "
            "name is accepted as an alias, but no 3D drift gate is used."
        ),
    )
    parser.add_argument(
        "--max-receiver-horizontal-drift-m",
        "--max-receiver-drift-m",
        dest="max_receiver_horizontal_drift_m",
        type=float,
        default=FULL_MAX_RECEIVER_HORIZONTAL_DRIFT_M,
        help=(
            "Maximum raw-reset to settled receiver XY drift. Vertical "
            "spawn settling remains a recorded diagnostic."
        ),
    )
    parser.add_argument(
        "--max-orientation-drift-deg",
        type=float,
        default=FULL_MAX_ORIENTATION_DRIFT_DEG,
    )
    parser.add_argument(
        "--max-final-window-motion-m",
        type=float,
        default=FULL_MAX_FINAL_WINDOW_MOTION_M,
    )
    parser.add_argument(
        "--max-final-window-orientation-deg",
        type=float,
        default=FULL_MAX_FINAL_WINDOW_ORIENTATION_DEG,
    )
    parser.add_argument("--check-determinism", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def normalize_run_config(args):
    if args.mode == "full":
        if args.task_ids is not None or args.layout_ids is not None:
            raise ValueError(
                "Full mode does not permit task/layout filters"
            )
        if (
            args.resets_per_layout is not None
            and args.resets_per_layout != FULL_RESETS_PER_LAYOUT
        ):
            raise ValueError(
                "Full mode requires --resets-per-layout=5"
            )
        if (
            args.settle_steps is not None
            and args.settle_steps != FULL_SETTLE_STEPS
        ):
            raise ValueError("Full mode requires --settle-steps=20")
        args.resets_per_layout = FULL_RESETS_PER_LAYOUT
        args.settle_steps = FULL_SETTLE_STEPS
        args.check_determinism = True
        frozen = {
            "seed_base": FULL_SEED_BASE,
            "max_reset_attempts": FULL_MAX_RESET_ATTEMPTS,
            "max_manipulable_horizontal_drift_m": (
                FULL_MAX_MANIPULABLE_HORIZONTAL_DRIFT_M
            ),
            "max_receiver_horizontal_drift_m": (
                FULL_MAX_RECEIVER_HORIZONTAL_DRIFT_M
            ),
            "max_orientation_drift_deg": (
                FULL_MAX_ORIENTATION_DRIFT_DEG
            ),
            "max_final_window_motion_m": (
                FULL_MAX_FINAL_WINDOW_MOTION_M
            ),
            "max_final_window_orientation_deg": (
                FULL_MAX_FINAL_WINDOW_ORIENTATION_DEG
            ),
        }
        changed = {
            name: (getattr(args, name), expected)
            for name, expected in frozen.items()
            if getattr(args, name) != expected
        }
        if changed:
            raise ValueError(
                f"Full mode parameters are frozen: {changed}"
            )
    else:
        if args.resets_per_layout is None:
            args.resets_per_layout = 1
        if args.settle_steps is None:
            args.settle_steps = 20

    if args.resets_per_layout <= 0:
        raise ValueError("--resets-per-layout must be positive")
    if args.settle_steps < 0:
        raise ValueError("--settle-steps cannot be negative")
    if args.max_reset_attempts <= 0:
        raise ValueError("--max-reset-attempts must be positive")
    for name in [
        "max_manipulable_horizontal_drift_m",
        "max_receiver_horizontal_drift_m",
        "max_orientation_drift_deg",
        "max_final_window_motion_m",
        "max_final_window_orientation_deg",
    ]:
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root() / candidate


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root()), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def parse_bool(value, label: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid boolean for {label}: {value!r}")


def normalized_range_mapping(value: str) -> dict[str, tuple[float, ...]]:
    parsed = json.loads(value)
    return {
        str(key): tuple(float(item) for item in bounds)
        for key, bounds in parsed.items()
    }


def expected_task_semantics(row: dict) -> dict:
    object_type = row["object"]
    skill = row["skill"]
    region = row["spatial_region"]
    if object_type not in OBJECT_TYPES:
        raise ValueError(f"Unknown task object: {object_type}")
    if skill not in {"put_on_top", "put_inside", "push_to"}:
        raise ValueError(f"Unknown task skill: {skill}")
    if region not in TARGET_REGION_RANGES:
        raise ValueError(f"Unknown task region: {region}")

    target = f"{object_type}_1"
    receiver_region = f"{WORKSPACE_FIXTURE}_target_{region}_region"
    display = DISPLAY_NAMES[object_type]
    if skill == "put_on_top":
        language = (
            f"Pick up the {display} and place it on the "
            f"plate in the {region} region"
        )
        goal_predicates = [
            ["on", target, "plate_1"],
        ]
        goal_expression = f"(And (On {target} plate_1))"
    elif skill == "put_inside":
        language = (
            f"Pick up the {display} and place it inside the "
            f"basket in the {region} region"
        )
        goal_predicates = [
            ["in", target, "basket_1_contain_region"],
        ]
        goal_expression = (
            f"(And (In {target} basket_1_contain_region))"
        )
    else:
        language = (
            f"Push the {display} into the "
            f"{region} target region"
        )
        goal_predicates = [
            ["on", target, receiver_region],
        ]
        goal_expression = (
            f"(And (On {target} {receiver_region}))"
        )

    return {
        "tuple": f"{object_type}*{skill}*{region}",
        "language": language,
        "receiver_region": receiver_region,
        "goal_predicates": goal_predicates,
        "goal_expression": goal_expression,
    }


def read_layout_spec(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Layout spec not found: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    fields = reader.fieldnames or []
    if fields != LAYOUT_SPEC_FIELDS:
        raise ValueError(
            "Layout spec schema changed: "
            f"expected={LAYOUT_SPEC_FIELDS}, actual={fields}"
        )
    if len(rows) != 144:
        raise ValueError(
            f"Expected 144 task-layout rows, found {len(rows)}"
        )

    normalized = []
    keys = set()
    for row in rows:
        if row["protocol_version"] != PROTOCOL_VERSION:
            raise ValueError("Layout spec protocol mismatch")
        if row["benchmark_name"] != BENCHMARK_NAME:
            raise ValueError("Layout spec benchmark mismatch")
        camera_config.validate_frozen_camera_spec_row(row)

        task_id = int(row["task_id"])
        layout_id = int(row["layout_id"])
        key = (task_id, layout_id)
        if key in keys:
            raise ValueError(f"Duplicate task-layout row: {key}")
        keys.add(key)
        if not 0 <= task_id < 36:
            raise ValueError(f"Invalid task ID: {task_id}")
        if not 0 <= layout_id < 4:
            raise ValueError(f"Invalid layout ID: {layout_id}")

        bddl_path = resolve_repo_path(row["bddl_path"])
        if not bddl_path.is_file():
            raise FileNotFoundError(
                f"Generated BDDL is missing: {bddl_path}"
            )
        if sha256_file(bddl_path) != row["bddl_sha256"]:
            raise ValueError(
                f"Generated BDDL hash mismatch: {bddl_path}"
            )

        mapping = json.loads(row["object_to_slot_json"])
        if set(mapping) != set(OBJECT_TYPES):
            raise ValueError(f"Invalid source layout mapping: {key}")
        if sorted(int(value) for value in mapping.values()) != [0, 1, 2, 3]:
            raise ValueError(f"Source layout is not a permutation: {key}")

        source_ranges = normalized_range_mapping(
            row["source_slot_ranges_json"]
        )
        expected_source_ranges = {
            str(key): tuple(value)
            for key, value in SOURCE_SLOT_RANGES.items()
        }
        if source_ranges != expected_source_ranges:
            raise ValueError(f"Source ranges changed for row: {key}")

        target_ranges = normalized_range_mapping(
            row["target_region_ranges_json"]
        )
        expected_target_ranges = {
            key: tuple(value)
            for key, value in TARGET_REGION_RANGES.items()
        }
        if target_ranges != expected_target_ranges:
            raise ValueError(f"Target ranges changed for row: {key}")

        expected_semantics = expected_task_semantics(row)
        for field in [
            "tuple",
            "language",
            "receiver_region",
            "goal_expression",
        ]:
            if row[field] != expected_semantics[field]:
                raise ValueError(
                    f"Independent task semantics mismatch for {key}, "
                    f"field={field}: {row[field]!r} != "
                    f"{expected_semantics[field]!r}"
                )
        manifest_goal = json.loads(row["goal_predicates_json"])
        if normalize_predicates(manifest_goal) != normalize_predicates(
            expected_semantics["goal_predicates"]
        ):
            raise ValueError(
                f"Independent goal predicate mismatch for {key}"
            )

        normalized.append(
            {
                **row,
                "task_id": task_id,
                "layout_id": layout_id,
                "bddl_path": bddl_path,
                "mapping": {
                    name: int(value)
                    for name, value in mapping.items()
                },
                "goal_predicates": json.loads(
                    row["goal_predicates_json"]
                ),
                "expected_semantics": expected_semantics,
            }
        )

    expected_keys = {
        (task_id, layout_id)
        for task_id in range(36)
        for layout_id in range(4)
    }
    if keys != expected_keys:
        raise ValueError("Task-layout keys are not exactly 36 x 4")

    by_task = {
        task_id: sorted(
            [
                row for row in normalized
                if row["task_id"] == task_id
            ],
            key=lambda row: row["layout_id"],
        )
        for task_id in range(36)
    }
    expected_factors = list(
        itertools.product(
            OBJECT_TYPES,
            ["put_on_top", "put_inside", "push_to"],
            ["left", "middle", "right"],
        )
    )
    for task_id, expected_factor in enumerate(expected_factors):
        task_rows = by_task[task_id]
        if len(task_rows) != 4:
            raise ValueError(
                f"Task {task_id} does not have four layouts"
            )
        actual_factors = {
            (
                row["object"],
                row["skill"],
                row["spatial_region"],
            )
            for row in task_rows
        }
        if actual_factors != {expected_factor}:
            raise ValueError(
                f"Task {task_id} factor mismatch: {actual_factors}"
            )
        for field in [
            "tuple",
            "language",
            "goal_expression",
            "goal_predicates_json",
            "receiver_region",
        ]:
            if len({row[field] for row in task_rows}) != 1:
                raise ValueError(
                    f"Task {task_id} changes {field} across layouts"
                )
        for object_type in OBJECT_TYPES:
            slots = sorted(
                row["mapping"][object_type]
                for row in task_rows
            )
            if slots != [0, 1, 2, 3]:
                raise ValueError(
                    f"Task {task_id} has unbalanced layouts for "
                    f"{object_type}: {slots}"
                )
    return normalized


def seed_environment(env, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if hasattr(env, "seed"):
        env.seed(seed)


def make_environment(path: Path):
    return OffScreenRenderEnv(
        bddl_file_name=str(path),
        camera_heights=camera_config.CAMERA_HEIGHT,
        camera_widths=camera_config.CAMERA_WIDTH,
        horizon=1000,
    )


def safe_reset(env, seed: int, max_attempts: int):
    seed_environment(env, seed)
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            env.env.reset()
            observation, camera_record = (
                camera_config.apply_frozen_agentview_camera(env)
            )
            return observation, attempt, camera_record
        except RandomizationError as error:
            last_error = error
    raise ResetAttemptsExceeded(
        f"Failed to sample a valid reset in {max_attempts} attempts: "
        f"{last_error}"
    )


def observation_pose(obs, instance: str) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(
        obs[f"{instance}_pos"],
        dtype=np.float64,
    ).reshape(-1)
    quaternion = np.asarray(
        obs[f"{instance}_quat"],
        dtype=np.float64,
    ).reshape(-1)
    if position.shape != (3,):
        raise ValueError(
            f"Unexpected position shape for {instance}: {position.shape}"
        )
    if quaternion.shape != (4,):
        raise ValueError(
            f"Unexpected quaternion shape for {instance}: "
            f"{quaternion.shape}"
        )
    quaternion_norm = float(np.linalg.norm(quaternion))
    if quaternion_norm <= 1e-12:
        raise ValueError(f"Zero quaternion for {instance}")
    return position, quaternion / quaternion_norm


def capture_poses(
    obs,
    receiver_instance: str | None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    names = list(MANIPULABLES)
    if receiver_instance is not None:
        names.append(receiver_instance)
    return {
        name: observation_pose(obs, name)
        for name in names
    }


def poses_are_finite(
    poses: dict[str, tuple[np.ndarray, np.ndarray]],
) -> bool:
    return all(
        np.isfinite(position).all()
        and np.isfinite(quaternion).all()
        for position, quaternion in poses.values()
    )


def poses_to_json(
    poses: dict[str, tuple[np.ndarray, np.ndarray]],
) -> str:
    payload = {
        name: {
            "position": [
                float(value)
                for value in position
            ],
            "quaternion_xyzw": [
                float(value)
                for value in quaternion
            ],
        }
        for name, (position, quaternion) in poses.items()
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )


def images_valid(obs) -> bool:
    for key in ["agentview_image", "robot0_eye_in_hand_image"]:
        image = np.asarray(obs[key])
        if image.shape != (128, 128, 3):
            return False
        if image.dtype != np.uint8:
            return False
        if int(image.min()) < 0 or int(image.max()) > 255:
            return False
        if float(image.astype(np.float32).std()) < 1.0:
            return False
    return True


def state_on_region(inner, object_name: str, region_name: str) -> bool:
    object_state = inner.object_states_dict[object_name]
    region_state = inner.object_states_dict[region_name]
    return bool(region_state.check_ontop(object_state))


def xy_in_bounds(
    position: np.ndarray,
    bounds: tuple[float, float, float, float],
    tolerance: float = 1e-6,
) -> bool:
    x1, y1, x2, y2 = bounds
    return bool(
        x1 - tolerance <= position[0] <= x2 + tolerance
        and y1 - tolerance <= position[1] <= y2 + tolerance
    )


def source_region_details(
    inner,
    poses: dict[str, tuple[np.ndarray, np.ndarray]],
    mapping: dict[str, int],
) -> dict[str, dict]:
    details = {}
    for object_type, slot_id in mapping.items():
        instance = f"{object_type}_1"
        region = (
            f"{WORKSPACE_FIXTURE}_source_slot_{slot_id}_region"
        )
        official_on = state_on_region(inner, instance, region)
        position = poses[instance][0]
        xy_ok = xy_in_bounds(
            poses[instance][0],
            SOURCE_SLOT_RANGES[slot_id],
        )
        details[instance] = {
            "region": region,
            "bounds": list(SOURCE_SLOT_RANGES[slot_id]),
            "xy": [float(position[0]), float(position[1])],
            "official_on": official_on,
            "xy_in_bounds": xy_ok,
            "combined": bool(official_on and xy_ok),
        }
    return details


def region_detail_checks(
    details: dict[str, dict],
) -> tuple[bool, bool, bool]:
    official = all(
        bool(detail["official_on"])
        for detail in details.values()
    )
    xy_ok = all(
        bool(detail["xy_in_bounds"])
        for detail in details.values()
    )
    return official, xy_ok, bool(official and xy_ok)


def details_to_json(details: dict[str, dict]) -> str:
    return json.dumps(
        details,
        sort_keys=True,
        separators=(",", ":"),
    )


def receiver_for_skill(skill: str) -> str | None:
    if skill == "put_on_top":
        return "plate_1"
    if skill == "put_inside":
        return "basket_1"
    if skill == "push_to":
        return None
    raise ValueError(f"Unknown skill: {skill}")


def receiver_region_details(
    inner,
    poses: dict[str, tuple[np.ndarray, np.ndarray]],
    receiver: str | None,
    region: str,
) -> dict[str, dict]:
    if receiver is None:
        return {}
    prefix = f"{WORKSPACE_FIXTURE}_target_"
    if not region.startswith(prefix) or not region.endswith("_region"):
        raise ValueError(f"Invalid receiver region name: {region}")
    region_name = region.removeprefix(prefix).removesuffix("_region")
    if region_name not in TARGET_REGION_RANGES:
        raise ValueError(f"Unknown receiver region: {region}")
    official_on = state_on_region(inner, receiver, region)
    position = poses[receiver][0]
    xy_ok = xy_in_bounds(
        position,
        TARGET_REGION_RANGES[region_name],
    )
    return {
        receiver: {
            "region": region,
            "bounds": list(TARGET_REGION_RANGES[region_name]),
            "xy": [float(position[0]), float(position[1])],
            "official_on": official_on,
            "xy_in_bounds": xy_ok,
            "combined": bool(official_on and xy_ok),
        }
    }


def contact_pairs(inner, receiver: str | None) -> list[list[str]]:
    names = list(MANIPULABLES)
    if receiver is not None:
        names.append(receiver)
    pairs = []
    for left, right in combinations(names, 2):
        if inner.object_states_dict[left].check_contact(
            inner.object_states_dict[right]
        ):
            pairs.append([left, right])
    return pairs


def quaternion_distance_deg(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    dot = float(np.clip(abs(np.dot(left, right)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def pose_drift(
    initial: dict[str, tuple[np.ndarray, np.ndarray]],
    settled: dict[str, tuple[np.ndarray, np.ndarray]],
    names: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    position = {
        name: float(
            np.linalg.norm(settled[name][0] - initial[name][0])
        )
        for name in names
    }
    orientation = {
        name: quaternion_distance_deg(
            initial[name][1],
            settled[name][1],
        )
        for name in names
    }
    return position, orientation


def horizontal_pose_drift(
    initial: dict[str, tuple[np.ndarray, np.ndarray]],
    settled: dict[str, tuple[np.ndarray, np.ndarray]],
    names: list[str],
) -> dict[str, float]:
    return {
        name: float(
            np.linalg.norm(
                settled[name][0][:2] - initial[name][0][:2]
            )
        )
        for name in names
    }


def vertical_pose_displacement(
    initial: dict[str, tuple[np.ndarray, np.ndarray]],
    settled: dict[str, tuple[np.ndarray, np.ndarray]],
    names: list[str],
) -> dict[str, float]:
    return {
        name: float(settled[name][0][2] - initial[name][0][2])
        for name in names
    }


def max_pose_motion(
    previous: dict[str, tuple[np.ndarray, np.ndarray]],
    current: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    translation = max(
        float(np.linalg.norm(current[name][0] - previous[name][0]))
        for name in current
    )
    orientation = max(
        quaternion_distance_deg(
            previous[name][1],
            current[name][1],
        )
        for name in current
    )
    return translation, orientation


def normalize_predicates(predicates) -> list[tuple[str, ...]]:
    return sorted(
        tuple(str(token).strip().lower() for token in predicate)
        for predicate in predicates
    )


def normalized_text(value) -> str:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    return " ".join(str(value).strip().lower().split())


def validate_region_definitions(parsed, skill: str):
    regions = parsed["regions"]
    expected_names = {
        f"{WORKSPACE_FIXTURE}_source_slot_{slot_id}_region"
        for slot_id in SOURCE_SLOT_RANGES
    } | {
        f"{WORKSPACE_FIXTURE}_target_{region}_region"
        for region in TARGET_REGION_RANGES
    }
    if skill == "put_inside":
        expected_names.add("basket_1_contain_region")
    if set(regions) != expected_names:
        raise ValueError(
            f"Parsed region names mismatch: {set(regions)}"
        )

    for slot_id, expected in SOURCE_SLOT_RANGES.items():
        name = (
            f"{WORKSPACE_FIXTURE}_source_slot_{slot_id}_region"
        )
        if regions[name]["target"] != WORKSPACE_FIXTURE:
            raise ValueError(f"Invalid source region target: {name}")
        actual = tuple(float(value) for value in regions[name]["ranges"][0])
        if actual != tuple(expected):
            raise ValueError(
                f"Parsed source range mismatch for {name}: {actual}"
            )
    for region, expected in TARGET_REGION_RANGES.items():
        name = f"{WORKSPACE_FIXTURE}_target_{region}_region"
        if regions[name]["target"] != WORKSPACE_FIXTURE:
            raise ValueError(f"Invalid target region target: {name}")
        actual = tuple(float(value) for value in regions[name]["ranges"][0])
        if actual != tuple(expected):
            raise ValueError(
                f"Parsed target range mismatch for {name}: {actual}"
            )
        rgba = tuple(float(value) for value in regions[name]["rgba"])
        if rgba != TARGET_REGION_RGBA:
            raise ValueError(
                f"Parsed target RGBA mismatch for {name}: {rgba}"
            )
    if skill == "put_inside":
        contain = regions["basket_1_contain_region"]
        if contain["target"] != "basket_1" or contain["ranges"]:
            raise ValueError("Invalid basket contain-region definition")
    centers_y = [
        np.mean(TARGET_REGION_RANGES[name][1::2])
        for name in ["left", "middle", "right"]
    ]
    if not centers_y[0] < centers_y[1] < centers_y[2]:
        raise ValueError("World-y target region order is invalid")


def validate_parsed_semantics(inner, row: dict):
    parsed = inner.parsed_problem
    semantics = expected_task_semantics(row)
    if normalized_text(parsed["language_instruction"]) != normalized_text(
        semantics["language"]
    ):
        raise ValueError("Parsed language does not match layout spec")

    actual_goal = normalize_predicates(parsed["goal_state"])
    expected_goal = normalize_predicates(
        semantics["goal_predicates"]
    )
    if actual_goal != expected_goal:
        raise ValueError(
            f"Parsed goal mismatch: {actual_goal} != {expected_goal}"
        )

    actual_objects = {
        instance
        for instances in parsed["objects"].values()
        for instance in instances
    }
    expected_objects = set(MANIPULABLES)
    receiver = receiver_for_skill(row["skill"])
    if receiver is not None:
        expected_objects.add(receiver)
    if actual_objects != expected_objects:
        raise ValueError(
            f"Parsed objects mismatch: {actual_objects} != "
            f"{expected_objects}"
        )

    actual_fixtures = {
        instance
        for instances in parsed["fixtures"].values()
        for instance in instances
    }
    if actual_fixtures != {WORKSPACE_FIXTURE}:
        raise ValueError(
            f"Parsed fixtures mismatch: {actual_fixtures}"
        )

    target = f"{row['object']}_1"
    expected_interest = {target}
    if row["skill"] == "put_on_top":
        expected_interest.add("plate_1")
    elif row["skill"] == "put_inside":
        expected_interest.add("basket_1")
    else:
        expected_interest.add(semantics["receiver_region"])
    if set(parsed["obj_of_interest"]) != expected_interest:
        raise ValueError("Parsed objects of interest mismatch")

    validate_region_definitions(parsed, row["skill"])


def settle_environment(
    env,
    obs,
    row: dict,
    receiver: str | None,
    settle_steps: int,
) -> dict:
    current_poses = capture_poses(obs, receiver)
    goal_became_true_at_step = None
    settled_obs = obs
    all_source_official_ok = True
    all_source_xy_ok = True
    all_receiver_official_ok = True
    all_receiver_xy_ok = True
    source_official_history = []
    source_xy_history = []
    receiver_official_history = []
    receiver_xy_history = []
    ever_contacts = set()
    step_translation = []
    step_orientation = []
    action = np.zeros(int(env.env.action_dim), dtype=np.float32)
    for step_index in range(1, settle_steps + 1):
        settled_obs, _, done, _ = env.step(action)
        next_poses = capture_poses(settled_obs, receiver)
        source_details = source_region_details(
            env.env,
            next_poses,
            row["mapping"],
        )
        source_official, source_xy, _ = region_detail_checks(
            source_details
        )
        receiver_details = receiver_region_details(
            env.env,
            next_poses,
            receiver,
            row["receiver_region"],
        )
        receiver_official, receiver_xy, _ = region_detail_checks(
            receiver_details
        )
        all_source_official_ok = bool(
            all_source_official_ok and source_official
        )
        all_source_xy_ok = bool(all_source_xy_ok and source_xy)
        all_receiver_official_ok = bool(
            all_receiver_official_ok and receiver_official
        )
        all_receiver_xy_ok = bool(
            all_receiver_xy_ok and receiver_xy
        )
        source_official_history.append(source_official)
        source_xy_history.append(source_xy)
        receiver_official_history.append(receiver_official)
        receiver_xy_history.append(receiver_xy)
        ever_contacts.update(
            tuple(pair)
            for pair in contact_pairs(env.env, receiver)
        )
        translation, orientation = max_pose_motion(
            current_poses,
            next_poses,
        )
        step_translation.append(translation)
        step_orientation.append(orientation)
        current_poses = next_poses
        if (done or env.check_success()) and goal_became_true_at_step is None:
            goal_became_true_at_step = step_index

    window_size = min(FULL_FINAL_WINDOW_STEPS, len(step_translation))
    if window_size:
        max_window_translation = max(
            step_translation[-window_size:]
        )
        max_window_orientation = max(
            step_orientation[-window_size:]
        )
    else:
        max_window_translation = 0.0
        max_window_orientation = 0.0

    def final_window_all(history: list[bool]) -> bool:
        if not history:
            return True
        return all(history[-FULL_FINAL_WINDOW_STEPS:])

    final_source_official_ok = final_window_all(
        source_official_history
    )
    final_source_xy_ok = final_window_all(source_xy_history)
    final_receiver_official_ok = final_window_all(
        receiver_official_history
    )
    final_receiver_xy_ok = final_window_all(receiver_xy_history)

    return {
        "settled_obs": settled_obs,
        "goal_became_true_at_step": goal_became_true_at_step,
        "all_source_official_ok": all_source_official_ok,
        "all_source_xy_ok": all_source_xy_ok,
        "all_receiver_official_ok": all_receiver_official_ok,
        "all_receiver_xy_ok": all_receiver_xy_ok,
        "final_source_official_ok": final_source_official_ok,
        "final_source_xy_ok": final_source_xy_ok,
        "final_receiver_official_ok": final_receiver_official_ok,
        "final_receiver_xy_ok": final_receiver_xy_ok,
        "ever_contacts": [
            list(pair)
            for pair in sorted(ever_contacts)
        ],
        "max_final_window_motion_m": max_window_translation,
        "max_final_window_orientation_deg": (
            max_window_orientation
        ),
    }


def save_previews(
    preview_dir: Path,
    row: dict,
    initial_obs,
    settled_obs,
):
    from PIL import Image

    preview_dir.mkdir(parents=True, exist_ok=True)
    stem = f"task_{row['task_id']:03d}_layout_{row['layout_id']}"
    artifacts = {}
    for stage, obs in [
        ("initial", initial_obs),
        ("settled", settled_obs),
    ]:
        for camera, key in [
            ("agentview", "agentview_image"),
            ("wrist", "robot0_eye_in_hand_image"),
        ]:
            path = preview_dir / f"{stem}_{stage}_{camera}.png"
            Image.fromarray(np.asarray(obs[key])).save(path)
            artifacts[f"{stage}_{camera}"] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }
    return artifacts


def max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left - right)))


def reset_once(
    env,
    row: dict,
    reset_index: int,
    seed: int,
    args,
    preview_dir: Path | None,
) -> dict:
    receiver = receiver_for_skill(row["skill"])
    initial_obs, reset_attempts, camera_record = safe_reset(
        env,
        seed,
        args.max_reset_attempts,
    )
    validate_parsed_semantics(env.env, row)
    initial_sim_state = np.asarray(
        env.get_sim_state(),
        dtype=np.float64,
    ).copy()
    initial_poses = capture_poses(initial_obs, receiver)
    initial_goal_false = not bool(env.check_success())
    initial_source_details = source_region_details(
        env.env,
        initial_poses,
        row["mapping"],
    )
    (
        initial_source_official_ok,
        initial_source_xy_ok,
        initial_source_combined_ok,
    ) = region_detail_checks(initial_source_details)
    initial_receiver_details = receiver_region_details(
        env.env,
        initial_poses,
        receiver,
        row["receiver_region"],
    )
    (
        initial_receiver_official_ok,
        initial_receiver_xy_ok,
        _,
    ) = region_detail_checks(initial_receiver_details)
    initial_contacts = contact_pairs(env.env, receiver)

    settle_result = settle_environment(
        env,
        initial_obs,
        row,
        receiver,
        args.settle_steps,
    )
    settled_obs = settle_result["settled_obs"]
    goal_true_step = settle_result["goal_became_true_at_step"]
    settled_sim_state = np.asarray(
        env.get_sim_state(),
        dtype=np.float64,
    ).copy()
    settled_poses = capture_poses(settled_obs, receiver)
    settled_goal_false = not bool(env.check_success())
    settled_source_details = source_region_details(
        env.env,
        settled_poses,
        row["mapping"],
    )
    (
        settled_source_official_ok,
        settled_source_xy_ok,
        settled_source_combined_ok,
    ) = region_detail_checks(settled_source_details)
    settled_receiver_details = receiver_region_details(
        env.env,
        settled_poses,
        receiver,
        row["receiver_region"],
    )
    (
        settled_receiver_official_ok,
        settled_receiver_xy_ok,
        _,
    ) = region_detail_checks(settled_receiver_details)
    settled_contacts = contact_pairs(env.env, receiver)

    position_drift, orientation_drift = pose_drift(
        initial_poses,
        settled_poses,
        MANIPULABLES,
    )
    max_manipulable_drift = max(position_drift.values())
    horizontal_drift = horizontal_pose_drift(
        initial_poses,
        settled_poses,
        MANIPULABLES,
    )
    max_manipulable_horizontal_drift = max(
        horizontal_drift.values()
    )
    vertical_displacement = vertical_pose_displacement(
        initial_poses,
        settled_poses,
        MANIPULABLES,
    )
    max_manipulable_orientation_drift = max(
        orientation_drift.values()
    )
    receiver_drift = None
    receiver_horizontal_drift = None
    receiver_vertical_displacement = None
    receiver_orientation_drift = None
    if receiver is not None:
        receiver_position, receiver_orientation = pose_drift(
            initial_poses,
            settled_poses,
            [receiver],
        )
        receiver_drift = receiver_position[receiver]
        receiver_horizontal_drift = horizontal_pose_drift(
            initial_poses,
            settled_poses,
            [receiver],
        )[receiver]
        receiver_vertical_displacement = vertical_pose_displacement(
            initial_poses,
            settled_poses,
            [receiver],
        )[receiver]
        receiver_orientation_drift = receiver_orientation[receiver]

    max_final_window_motion = settle_result[
        "max_final_window_motion_m"
    ]
    max_final_window_orientation = settle_result[
        "max_final_window_orientation_deg"
    ]
    ever_contacts = settle_result["ever_contacts"]

    determinism_checked = bool(
        args.check_determinism and reset_index == 0
    )
    determinism_initial = None
    determinism_settled = None
    deterministic = None
    if determinism_checked:
        repeated_env = make_environment(row["bddl_path"])
        try:
            (
                repeated_initial_obs,
                repeated_attempts,
                repeated_camera_record,
            ) = safe_reset(
                repeated_env,
                seed,
                args.max_reset_attempts,
            )
            if repeated_attempts != reset_attempts:
                raise ValueError(
                    "Deterministic reset used a different attempt count"
                )
            camera_config.validate_runtime_camera_record(
                repeated_camera_record
            )
            repeated_initial_state = np.asarray(
                repeated_env.get_sim_state(),
                dtype=np.float64,
            ).copy()
            repeated_settle = settle_environment(
                repeated_env,
                repeated_initial_obs,
                row,
                receiver,
                args.settle_steps,
            )
            repeated_settled_state = np.asarray(
                repeated_env.get_sim_state(),
                dtype=np.float64,
            ).copy()
            determinism_initial = max_abs_difference(
                initial_sim_state,
                repeated_initial_state,
            )
            determinism_settled = max_abs_difference(
                settled_sim_state,
                repeated_settled_state,
            )
            deterministic = bool(
                repeated_settle["goal_became_true_at_step"]
                == goal_true_step
                and determinism_initial <= 1e-8
                and determinism_settled <= 1e-8
            )
        finally:
            repeated_env.close()

    preview_artifacts = {}
    if preview_dir is not None and reset_index == 0:
        preview_artifacts = save_previews(
            preview_dir,
            row,
            initial_obs,
            settled_obs,
        )

    receiver_horizontal_drift_ok = (
        receiver_horizontal_drift is None
        or receiver_horizontal_drift
        <= args.max_receiver_horizontal_drift_m
    )
    receiver_orientation_ok = (
        receiver_orientation_drift is None
        or receiver_orientation_drift
        <= args.max_orientation_drift_deg
    )
    determinism_ok = (
        not determinism_checked or bool(deterministic)
    )
    gate_checks = {
        "frozen_camera_configuration": True,
        "initial_goal_false": initial_goal_false,
        "settled_goal_false": settled_goal_false,
        "goal_never_true_during_settling": goal_true_step is None,
        # Raw reset poses are sampled above the support surface. Their
        # official On predicates may therefore be false until passive
        # settling finishes; raw placement is gated only in XY.
        "initial_source_xy_bounds": initial_source_xy_ok,
        "initial_receiver_xy_bounds": initial_receiver_xy_ok,
        # The stabilized state is the formal initial placement. Source
        # objects must satisfy LIBERO's official On predicates and XY
        # bounds. Receiver placement is gated independently in XY; its
        # official On predicate remains diagnostic because tall
        # receivers such as the basket can violate the site's fixed
        # height window while remaining correctly placed and stable.
        "settled_source_official_predicates": (
            settled_source_official_ok
        ),
        "settled_source_xy_bounds": settled_source_xy_ok,
        "settled_receiver_xy_bounds": settled_receiver_xy_ok,
        "final_window_source_official_predicates": settle_result[
            "final_source_official_ok"
        ],
        "final_window_source_xy_bounds": settle_result[
            "final_source_xy_ok"
        ],
        "final_window_receiver_xy_bounds": settle_result[
            "final_receiver_xy_ok"
        ],
        "initial_images": images_valid(initial_obs),
        "settled_images": images_valid(settled_obs),
        "initial_finite_poses": poses_are_finite(initial_poses),
        "settled_finite_poses": poses_are_finite(settled_poses),
        # The simulator has not advanced at the raw-reset frame. Coarse
        # placement sampling can report a transient mesh contact there
        # even when it is absent from the first zero-action step onward.
        # Record raw contacts in the CSV, but gate every physics settle
        # step and the stabilized endpoint instead.
        "no_settled_unexpected_contacts": not settled_contacts,
        "no_unexpected_contacts_during_settling": not ever_contacts,
        # Full 3D raw-to-settled displacement is diagnostic because it
        # includes intended vertical spawn settling. Gate XY motion only.
        "manipulable_horizontal_drift": (
            max_manipulable_horizontal_drift
            <= args.max_manipulable_horizontal_drift_m
        ),
        "manipulable_orientation_drift": (
            max_manipulable_orientation_drift
            <= args.max_orientation_drift_deg
        ),
        "receiver_horizontal_drift": receiver_horizontal_drift_ok,
        "receiver_orientation_drift": receiver_orientation_ok,
        "final_window_motion": (
            max_final_window_motion
            <= args.max_final_window_motion_m
        ),
        "final_window_orientation": (
            max_final_window_orientation
            <= args.max_final_window_orientation_deg
        ),
        "determinism": determinism_ok,
    }
    failed_gate_checks = [
        name for name, value in gate_checks.items() if not bool(value)
    ]
    passed = not failed_gate_checks

    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_name": BENCHMARK_NAME,
        "mode": args.mode,
        "task_id": row["task_id"],
        "layout_id": row["layout_id"],
        "reset_index": reset_index,
        "seed": seed,
        "reset_attempts": reset_attempts,
        "tuple": row["tuple"],
        "object": row["object"],
        "skill": row["skill"],
        "spatial_region": row["spatial_region"],
        "bddl_path": str(row["bddl_path"]),
        "bddl_sha256": row["bddl_sha256"],
        "object_to_slot_json": json.dumps(
            row["mapping"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "frozen_camera_configuration_ok": True,
        "agentview_camera_record_json": json.dumps(
            camera_record,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "preview_artifacts_json": json.dumps(
            preview_artifacts,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "parsed_semantics_ok": True,
        "initial_goal_false": initial_goal_false,
        "goal_became_true_at_step": (
            "" if goal_true_step is None else goal_true_step
        ),
        "settled_goal_false": settled_goal_false,
        "initial_source_regions_ok": initial_source_xy_ok,
        "initial_source_official_predicates_ok": (
            initial_source_official_ok
        ),
        "initial_source_xy_bounds_ok": initial_source_xy_ok,
        "initial_source_region_details_json": details_to_json(
            initial_source_details
        ),
        "settled_source_regions_ok": settled_source_combined_ok,
        "settled_source_official_predicates_ok": (
            settled_source_official_ok
        ),
        "settled_source_xy_bounds_ok": settled_source_xy_ok,
        "settled_source_region_details_json": details_to_json(
            settled_source_details
        ),
        "all_settle_source_regions_ok": bool(
            settle_result["all_source_official_ok"]
            and settle_result["all_source_xy_ok"]
        ),
        "all_settle_source_official_predicates_ok": settle_result[
            "all_source_official_ok"
        ],
        "all_settle_source_xy_bounds_ok": settle_result[
            "all_source_xy_ok"
        ],
        "final_window_source_regions_ok": bool(
            settle_result["final_source_official_ok"]
            and settle_result["final_source_xy_ok"]
        ),
        "final_window_source_official_predicates_ok": settle_result[
            "final_source_official_ok"
        ],
        "final_window_source_xy_bounds_ok": settle_result[
            "final_source_xy_ok"
        ],
        "initial_receiver_region_ok": initial_receiver_xy_ok,
        "initial_receiver_official_predicate_ok": (
            initial_receiver_official_ok
        ),
        "initial_receiver_xy_bounds_ok": initial_receiver_xy_ok,
        "initial_receiver_region_details_json": details_to_json(
            initial_receiver_details
        ),
        "settled_receiver_region_ok": settled_receiver_xy_ok,
        "settled_receiver_official_predicate_ok": (
            settled_receiver_official_ok
        ),
        "settled_receiver_xy_bounds_ok": settled_receiver_xy_ok,
        "settled_receiver_region_details_json": details_to_json(
            settled_receiver_details
        ),
        "all_settle_receiver_region_ok": settle_result[
            "all_receiver_xy_ok"
        ],
        "all_settle_receiver_official_predicate_ok": settle_result[
            "all_receiver_official_ok"
        ],
        "all_settle_receiver_xy_bounds_ok": settle_result[
            "all_receiver_xy_ok"
        ],
        "final_window_receiver_region_ok": settle_result[
            "final_receiver_xy_ok"
        ],
        "final_window_receiver_official_predicate_ok": settle_result[
            "final_receiver_official_ok"
        ],
        "final_window_receiver_xy_bounds_ok": settle_result[
            "final_receiver_xy_ok"
        ],
        "initial_images_ok": images_valid(initial_obs),
        "settled_images_ok": images_valid(settled_obs),
        "initial_finite_poses": poses_are_finite(initial_poses),
        "settled_finite_poses": poses_are_finite(settled_poses),
        "initial_unexpected_contact_count": len(initial_contacts),
        "initial_unexpected_contact_pairs_json": json.dumps(
            initial_contacts,
            separators=(",", ":"),
        ),
        "settled_unexpected_contact_count": len(settled_contacts),
        "settled_unexpected_contact_pairs_json": json.dumps(
            settled_contacts,
            separators=(",", ":"),
        ),
        "ever_unexpected_contact_count": len(ever_contacts),
        "ever_unexpected_contact_pairs_json": json.dumps(
            ever_contacts,
            separators=(",", ":"),
        ),
        "initial_poses_json": poses_to_json(initial_poses),
        "settled_poses_json": poses_to_json(settled_poses),
        "per_object_drift_m_json": json.dumps(
            position_drift,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "max_manipulable_drift_m": max_manipulable_drift,
        "per_object_horizontal_drift_m_json": json.dumps(
            horizontal_drift,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "max_manipulable_horizontal_drift_m": (
            max_manipulable_horizontal_drift
        ),
        "per_object_vertical_displacement_m_json": json.dumps(
            vertical_displacement,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "per_object_orientation_drift_deg_json": json.dumps(
            orientation_drift,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "max_manipulable_orientation_drift_deg": (
            max_manipulable_orientation_drift
        ),
        "receiver_drift_m": (
            "" if receiver_drift is None else receiver_drift
        ),
        "receiver_horizontal_drift_m": (
            ""
            if receiver_horizontal_drift is None
            else receiver_horizontal_drift
        ),
        "receiver_vertical_displacement_m": (
            ""
            if receiver_vertical_displacement is None
            else receiver_vertical_displacement
        ),
        "receiver_orientation_drift_deg": (
            ""
            if receiver_orientation_drift is None
            else receiver_orientation_drift
        ),
        "max_final_window_motion_m": max_final_window_motion,
        "max_final_window_orientation_deg": (
            max_final_window_orientation
        ),
        "determinism_checked": determinism_checked,
        "determinism_initial_state_max_abs": (
            "" if determinism_initial is None else determinism_initial
        ),
        "determinism_settled_state_max_abs": (
            "" if determinism_settled is None else determinism_settled
        ),
        "deterministic": (
            "" if deterministic is None else deterministic
        ),
        "failed_gate_checks_json": json.dumps(
            failed_gate_checks,
            separators=(",", ":"),
        ),
        "passed": passed,
        "error": "",
    }


def error_row(
    row: dict,
    reset_index: int,
    seed: int,
    args,
    error: Exception,
) -> dict:
    result = {field: "" for field in OUTPUT_FIELDS}
    result.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "benchmark_name": BENCHMARK_NAME,
            "mode": args.mode,
            "task_id": row["task_id"],
            "layout_id": row["layout_id"],
            "reset_index": reset_index,
            "seed": seed,
            "tuple": row["tuple"],
            "object": row["object"],
            "skill": row["skill"],
            "spatial_region": row["spatial_region"],
            "bddl_path": str(row["bddl_path"]),
            "bddl_sha256": row["bddl_sha256"],
            "object_to_slot_json": json.dumps(
                row.get("mapping", {}),
                sort_keys=True,
                separators=(",", ":"),
            ),
            "failed_gate_checks_json": '["exception"]',
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
        }
    )
    return result


def write_csv_atomic(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=OUTPUT_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def write_json_atomic(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def make_manifest(
    args,
    layout_spec: Path,
    output: Path,
    expected_rows: int,
    results: list[dict],
    run_complete: bool,
) -> dict:
    passed = sum(
        parse_bool(row["passed"], "passed")
        for row in results
    )
    full_shape = (
        args.mode == "full"
        and expected_rows == 720
        and len(results) == 720
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_name": BENCHMARK_NAME,
        "mode": args.mode,
        "git_commit": git_commit(),
        "validator_path": str(Path(__file__).resolve()),
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "camera_helper_path": str(
            Path(camera_config.__file__).resolve()
        ),
        "camera_helper_sha256": sha256_file(
            Path(camera_config.__file__).resolve()
        ),
        "frozen_observation_camera": (
            camera_config.frozen_camera_spec()
        ),
        "layout_spec": str(layout_spec),
        "layout_spec_sha256": sha256_file(layout_spec),
        "output": str(output),
        "preview_dir": (
            None
            if args.preview_dir is None
            else str(resolve_repo_path(args.preview_dir))
        ),
        "software_versions": {
            name: package_version(name)
            for name in ["libero", "robosuite", "mujoco", "numpy"]
        },
        "parameters": {
            "resets_per_layout": args.resets_per_layout,
            "settle_steps": args.settle_steps,
            "settle_action": "all_zero_osc_pose_action",
            "seed_base": args.seed_base,
            "max_reset_attempts": args.max_reset_attempts,
            "camera_application_point": (
                "after_every_successful_inner_reset_before_first_"
                "observation"
            ),
            "reset_returned_observation_used": False,
            "camera_application_adds_physics_step": False,
            "preview_policy": (
                "initial_and_settled_images_for_reset0_of_each_selected_"
                "layout_when_preview_dir_is_set"
            ),
            "formal_initial_state": (
                "passively_settled_after_zero_action_steps"
            ),
            "raw_reset_placement_gate": "xy_bounds_only",
            "raw_reset_unexpected_contacts": "diagnostic_only",
            "post_zero_action_step_unexpected_contacts": (
                "hard_gate_at_every_settle_step_and_stabilized_endpoint"
            ),
            "settled_placement_gate": (
                "source_official_on_and_xy_plus_receiver_xy_at_"
                "endpoint_and_every_final_window_step"
            ),
            "receiver_official_on_predicate": "diagnostic_only",
            "receiver_terminal_region_condition": (
                "independent_center_xy_check_required_during_oracle_"
                "and_formal_rollout_evaluation"
            ),
            "raw_to_settled_3d_drift": "diagnostic_only",
            "raw_to_settled_vertical_displacement": (
                "signed_diagnostic_only"
            ),
            "max_manipulable_horizontal_drift_m": (
                args.max_manipulable_horizontal_drift_m
            ),
            "max_receiver_horizontal_drift_m": (
                args.max_receiver_horizontal_drift_m
            ),
            "max_orientation_drift_deg": (
                args.max_orientation_drift_deg
            ),
            "final_window_steps": FULL_FINAL_WINDOW_STEPS,
            "max_final_window_motion_m": (
                args.max_final_window_motion_m
            ),
            "max_final_window_orientation_deg": (
                args.max_final_window_orientation_deg
            ),
            "determinism_check": args.check_determinism,
            "determinism_scope": (
                "independent_environment_for_reset0_of_each_layout"
                if args.check_determinism
                else "not_checked"
            ),
        },
        "expected_rows": expected_rows,
        "completed_rows": len(results),
        "passed_rows": passed,
        "failed_rows": len(results) - passed,
        "run_complete": run_complete,
        "full_protocol_shape": full_shape,
        "reset_gate_passed": bool(
            run_complete and full_shape and passed == 720
        ),
        "all_pretraining_gates_passed": False,
        "agentview_fov_selection": {
            "candidate_vertical_fov_degrees": [60, 65, 70, 75, 80],
            "selected_vertical_fov_degrees": (
                camera_config.AGENTVIEW_FOVY_DEG
            ),
            "selected_from_task0_all_four_layout_contact_sheets": True,
            "selection_reason": (
                "smallest_setting_with_about_six_pixels_of_projected_"
                "worst_case_margin_after_sampling_drift_and_tilt"
            ),
        },
        "agentview_fov_selection_gate_passed": True,
        "manual_agentview_left_right_confirmation_required": False,
        "representative_v4_camera_preview_gate_not_covered": True,
        "oracle_goal_gate_not_covered": True,
        "successful_trajectory_gate_not_covered": True,
        "push_threshold_calibration_not_covered": True,
    }


def filter_rows(rows: list[dict], args) -> list[dict]:
    selected = rows
    if args.task_ids is not None:
        requested = set(args.task_ids)
        selected = [
            row for row in selected
            if row["task_id"] in requested
        ]
        missing = requested - {row["task_id"] for row in selected}
        if missing:
            raise ValueError(f"Unknown task IDs: {sorted(missing)}")
    if args.layout_ids is not None:
        requested = set(args.layout_ids)
        selected = [
            row for row in selected
            if row["layout_id"] in requested
        ]
        missing = requested - {row["layout_id"] for row in selected}
        if missing:
            raise ValueError(f"Unknown layout IDs: {sorted(missing)}")
    if not selected:
        raise ValueError("No task-layout rows selected")
    return selected


def validate_registered_geometry():
    import libero.libero.envs.objects  # noqa: F401
    from libero.libero.envs.base_object import OBJECTS_DICT

    missing = sorted(set(SOURCE_SAMPLER_RADII_M) - set(OBJECTS_DICT))
    if missing:
        raise RuntimeError(
            f"Required LIBERO objects are not registered: {missing}"
        )
    for object_name, expected_radius in SOURCE_SAMPLER_RADII_M.items():
        model = OBJECTS_DICT[object_name](
            name=f"geometry_audit_{object_name}"
        )
        actual_radius = float(model.horizontal_radius)
        if not np.isclose(
            actual_radius,
            expected_radius,
            rtol=0.0,
            atol=1e-9,
        ):
            raise RuntimeError(
                f"Registered object radius changed for {object_name}: "
                f"{actual_radius} != {expected_radius}"
            )


def seed_for(row: dict, reset_index: int, seed_base: int) -> int:
    return (
        seed_base
        + row["task_id"] * 10000
        + row["layout_id"] * 1000
        + reset_index
    )


def main():
    args = parse_args()
    normalize_run_config(args)
    validate_registered_geometry()

    layout_spec = resolve_repo_path(args.layout_spec)
    all_rows = read_layout_spec(layout_spec)
    rows = filter_rows(all_rows, args)
    output = resolve_repo_path(args.output)
    manifest = (
        resolve_repo_path(args.manifest)
        if args.manifest is not None
        else output.with_suffix(".manifest.json")
    )
    preview_dir = (
        resolve_repo_path(args.preview_dir)
        if args.preview_dir is not None
        else None
    )

    results = []
    expected_rows = len(rows) * args.resets_per_layout
    run_index = 0
    write_json_atomic(
        manifest,
        make_manifest(
            args,
            layout_spec,
            output,
            expected_rows,
            results,
            run_complete=False,
        ),
    )

    print("=" * 80)
    print("LIBERO 36 proxy reset validation")
    print("mode:", args.mode)
    print("task-layout rows:", len(rows))
    print("resets per layout:", args.resets_per_layout)
    print("expected resets:", expected_rows)
    print("settle steps:", args.settle_steps)
    print("determinism:", args.check_determinism)
    print(
        "agentview vertical FOV:",
        camera_config.AGENTVIEW_FOVY_DEG,
    )
    print("output:", output)
    print("manifest:", manifest)

    for row in rows:
        env = None
        try:
            try:
                env = make_environment(row["bddl_path"])
            except Exception as error:
                for reset_index in range(args.resets_per_layout):
                    run_index += 1
                    result = error_row(
                        row,
                        reset_index,
                        seed_for(row, reset_index, args.seed_base),
                        args,
                        error,
                    )
                    results.append(result)
                    write_csv_atomic(output, results)
                    write_json_atomic(
                        manifest,
                        make_manifest(
                            args,
                            layout_spec,
                            output,
                            expected_rows,
                            results,
                            run_complete=False,
                        ),
                    )
                if args.fail_fast:
                    raise RuntimeError(
                        "Environment construction failed"
                    ) from error
                continue

            for reset_index in range(args.resets_per_layout):
                run_index += 1
                seed = seed_for(row, reset_index, args.seed_base)
                try:
                    result = reset_once(
                        env=env,
                        row=row,
                        reset_index=reset_index,
                        seed=seed,
                        args=args,
                        preview_dir=preview_dir,
                    )
                except Exception as error:
                    result = error_row(
                        row,
                        reset_index,
                        seed,
                        args,
                        error,
                    )
                results.append(result)
                write_csv_atomic(output, results)
                write_json_atomic(
                    manifest,
                    make_manifest(
                        args,
                        layout_spec,
                        output,
                        expected_rows,
                        results,
                        run_complete=False,
                    ),
                )
                print(
                    f"[{run_index:03d}/{expected_rows:03d}] "
                    f"task={row['task_id']:02d} "
                    f"layout={row['layout_id']} "
                    f"reset={reset_index} "
                    f"passed={result['passed']} "
                    f"failed={result['failed_gate_checks_json']} "
                    f"error={result['error']}"
                )
                if (
                    args.fail_fast
                    and not parse_bool(result["passed"], "passed")
                ):
                    raise RuntimeError(
                        "LIBERO 36 validation failed with --fail-fast"
                    )
        finally:
            if env is not None:
                env.close()

    passed = sum(
        parse_bool(row["passed"], "passed")
        for row in results
    )
    final_manifest = make_manifest(
        args,
        layout_spec,
        output,
        expected_rows,
        results,
        run_complete=True,
    )
    write_json_atomic(manifest, final_manifest)

    print("=" * 80)
    print("LIBERO 36 reset validation complete")
    print("rows:", len(results))
    print("passed:", passed)
    print("failed:", len(results) - passed)
    print("reset gate passed:", final_manifest["reset_gate_passed"])
    print(
        "all pretraining gates passed:",
        final_manifest["all_pretraining_gates_passed"],
    )
    print("output:", output)
    print("manifest:", manifest)
    if preview_dir is not None:
        print("previews:", preview_dir)

    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
