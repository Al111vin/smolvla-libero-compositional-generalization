from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import pdb
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glx")

import numpy as np

try:
    import libero_36_camera as camera_config
    import validate_libero_36_envs as reset_validator
except ModuleNotFoundError:
    from scripts import libero_36_camera as camera_config
    from scripts import validate_libero_36_envs as reset_validator


PROTOCOL_VERSION = "libero_36_proxy_tabletop_draft_v4"
BENCHMARK_NAME = "libero_registered_object_36_proxy_tabletop"
GATE_PROTOCOL_VERSION = "libero_36_oracle_goal_v1"
GATE_ID = "gate4_terminal_goal_semantics"

DEFAULT_LAYOUT_SPEC = Path("data/libero_36/layout_spec.csv")
DEFAULT_RESET_MANIFEST = Path(
    "results/libero_36_reset_audit.manifest.json"
)
DEFAULT_OUTPUT = Path("results/libero_36_goal_audit.csv")

CANONICAL_LAYOUT_ID = 0
FULL_TASK_IDS = tuple(range(36))
FULL_EXPECTED_CASES = 96
HOLD_STEPS = 20
ACQUIRE_MAX_STEPS = 60
ACQUIRE_STABLE_STEPS = 5
FINAL_WINDOW_STEPS = 5
SEED_BASE = 460000
MAX_RESET_ATTEMPTS = 100
MAX_FINAL_WINDOW_MOTION_M = 0.001
MAX_FINAL_WINDOW_ORIENTATION_DEG = 1.0

CASE_POSITIVE = "positive"
CASE_RELATION_NEGATIVE = "relation_negative"
CASE_RECEIVER_REGION_NEGATIVE = "receiver_region_negative"

CASE_INDEX = {
    CASE_POSITIVE: 0,
    CASE_RELATION_NEGATIVE: 1,
    CASE_RECEIVER_REGION_NEGATIVE: 2,
}

ALTERNATE_REGION = {
    "left": "middle",
    "middle": "right",
    "right": "left",
}

OUTPUT_FIELDS = [
    "protocol_version",
    "benchmark_name",
    "gate_protocol_version",
    "gate_id",
    "mode",
    "task_id",
    "layout_id",
    "tuple",
    "object",
    "skill",
    "spatial_region",
    "case_type",
    "case_index",
    "seed",
    "target_instance",
    "receiver_instance",
    "alternate_region",
    "original_bddl_path",
    "original_bddl_sha256",
    "derived_bddl_sha256",
    "object_to_slot_json",
    "derived_init_predicates_json",
    "construction_algorithm",
    "reset_attempts",
    "frozen_camera_record_json",
    "parsed_semantics_ok",
    "derived_initial_state_ok",
    "expected_relation",
    "expected_receiver_region",
    "expected_composite_success",
    "trajectory_constraints_evaluated",
    "push_threshold_status",
    "acquired",
    "acquired_at_step",
    "acquire_relation_trace_json",
    "acquire_receiver_region_trace_json",
    "acquire_composite_trace_json",
    "relation_trace_json",
    "receiver_region_trace_json",
    "composite_success_trace_json",
    "placement_trace_json",
    "receiver_table_contact_trace_json",
    "unexpected_contact_trace_json",
    "relation_hold_ok",
    "receiver_region_hold_ok",
    "composite_hold_ok",
    "placement_hold_ok",
    "receiver_table_contact_hold_ok",
    "no_unexpected_contacts",
    "all_finite_poses",
    "all_images_valid",
    "max_final_window_motion_m",
    "max_final_window_orientation_deg",
    "stable_final_window",
    "constructed_poses_json",
    "terminal_poses_json",
    "preview_artifacts_json",
    "determinism_checked",
    "repeated_reset_attempts",
    "determinism_initial_state_max_abs",
    "determinism_terminal_state_max_abs",
    "determinism_trajectory_state_max_abs",
    "determinism_trace_match",
    "deterministic",
    "failed_gate_checks_json",
    "passed",
    "error",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate the frozen LIBERO-36 Gate-4 terminal goal "
            "semantics with positive and isolated negative cases."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        required=True,
    )
    parser.add_argument(
        "--layout-spec",
        type=Path,
        default=DEFAULT_LAYOUT_SPEC,
    )
    parser.add_argument(
        "--reset-manifest",
        type=Path,
        default=DEFAULT_RESET_MANIFEST,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--task-ids", type=int, nargs="*")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.mode == "full" and args.task_ids is not None:
        raise ValueError("Full mode does not permit task filters")
    if args.mode == "smoke" and not args.task_ids:
        raise ValueError("Smoke mode requires --task-ids")
    if args.mode == "smoke" and args.output is None:
        raise ValueError("Smoke mode requires an explicit --output")
    if args.output is None:
        args.output = DEFAULT_OUTPUT
    return args


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=np.float64)
    return sha256_bytes(array.tobytes(order="C"))


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
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


def disable_upstream_interactive_debugger():
    def fail_noninteractively(*_args, **_kwargs):
        raise RuntimeError(
            "Upstream LIBERO placement sampler attempted to enter pdb"
        )

    pdb.set_trace = fail_noninteractively


def validate_runtime_versions():
    versions = {
        name: package_version(name)
        for name in ["hf_libero", "robosuite", "mujoco", "numpy"]
    }
    unknown = [name for name, value in versions.items() if value == "unknown"]
    if unknown:
        raise RuntimeError(
            f"Required software versions are unavailable: {unknown}"
        )
    return versions


def json_compact(payload) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )


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


def cases_for_skill(skill: str) -> list[str]:
    cases = [CASE_POSITIVE, CASE_RELATION_NEGATIVE]
    if skill in {"put_on_top", "put_inside"}:
        cases.append(CASE_RECEIVER_REGION_NEGATIVE)
    elif skill != "push_to":
        raise ValueError(f"Unknown skill: {skill}")
    return cases


def expected_case_count(rows: list[dict]) -> int:
    return sum(len(cases_for_skill(row["skill"])) for row in rows)


def validate_reset_gate(
    path: Path,
    layout_spec: Path,
) -> dict:
    with path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    required = {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_name": BENCHMARK_NAME,
        "mode": "full",
        "expected_rows": 720,
        "completed_rows": 720,
        "passed_rows": 720,
        "failed_rows": 0,
        "run_complete": True,
        "full_protocol_shape": True,
        "reset_gate_passed": True,
    }
    changed = {
        key: (manifest.get(key), value)
        for key, value in required.items()
        if manifest.get(key) != value
    }
    if changed:
        raise ValueError(f"Gate-3 reset manifest is invalid: {changed}")
    if manifest.get("layout_spec_sha256") != sha256_file(layout_spec):
        raise ValueError("Gate-3 layout-spec hash does not match")
    if manifest.get("camera_helper_sha256") != sha256_file(
        Path(camera_config.__file__).resolve()
    ):
        raise ValueError("Gate-3 camera-helper hash does not match")
    if manifest.get("validator_sha256") != sha256_file(
        Path(reset_validator.__file__).resolve()
    ):
        raise ValueError("Gate-3 reset-validator hash does not match")
    if manifest.get("frozen_observation_camera") != (
        camera_config.frozen_camera_spec()
    ):
        raise ValueError("Gate-3 frozen camera record does not match")

    recorded_versions = manifest.get("software_versions")
    if not isinstance(recorded_versions, dict):
        raise ValueError("Gate-3 software versions are missing")
    changed_versions = {
        name: (value, package_version(name))
        for name, value in recorded_versions.items()
        if value != package_version(name)
    }
    if changed_versions:
        raise ValueError(
            f"Gate-3 software versions changed: {changed_versions}"
        )

    reset_output_value = manifest.get("output")
    if not reset_output_value:
        raise ValueError("Gate-3 output path is missing")
    reset_output = Path(reset_output_value)
    if not reset_output.is_absolute():
        reset_output = resolve_repo_path(reset_output)
    if not reset_output.is_file():
        raise FileNotFoundError(
            f"Gate-3 output CSV is missing: {reset_output}"
        )
    with reset_output.open(newline="", encoding="utf-8") as file:
        reset_rows = list(csv.DictReader(file))
    if len(reset_rows) != 720:
        raise ValueError(
            f"Gate-3 output has {len(reset_rows)} rows, expected 720"
        )
    if any(
        row.get("protocol_version") != PROTOCOL_VERSION
        or row.get("benchmark_name") != BENCHMARK_NAME
        or row.get("mode") != "full"
        or str(row.get("passed", "")).strip().lower() != "true"
        or row.get("error") not in {"", None}
        for row in reset_rows
    ):
        raise ValueError("Gate-3 output CSV contains an invalid row")
    manifest["validated_output_path"] = str(reset_output)
    manifest["validated_output_sha256"] = sha256_file(reset_output)
    return manifest


def selected_layout_rows(
    layout_spec: Path,
    args,
) -> list[dict]:
    all_rows = reset_validator.read_layout_spec(layout_spec)
    rows = [
        row
        for row in all_rows
        if row["layout_id"] == CANONICAL_LAYOUT_ID
    ]
    task_ids = (
        set(FULL_TASK_IDS)
        if args.mode == "full"
        else set(args.task_ids)
    )
    unknown = task_ids - {row["task_id"] for row in rows}
    if unknown:
        raise ValueError(f"Unknown task IDs: {sorted(unknown)}")
    rows = sorted(
        [row for row in rows if row["task_id"] in task_ids],
        key=lambda row: row["task_id"],
    )
    if args.mode == "full":
        if [row["task_id"] for row in rows] != list(FULL_TASK_IDS):
            raise ValueError("Full mode does not cover exactly 36 tasks")
        if expected_case_count(rows) != FULL_EXPECTED_CASES:
            raise ValueError("Full mode does not cover exactly 96 cases")
    return rows


def target_instance(row: dict) -> str:
    return f"{row['object']}_1"


def positive_target_predicate(row: dict) -> list[str]:
    target = target_instance(row)
    if row["skill"] == "put_on_top":
        return ["on", target, "plate_1"]
    if row["skill"] == "put_inside":
        return ["in", target, "basket_1_contain_region"]
    if row["skill"] == "push_to":
        return ["on", target, row["receiver_region"]]
    raise ValueError(f"Unknown skill: {row['skill']}")


def source_predicate(row: dict, object_type: str) -> list[str]:
    return [
        "on",
        f"{object_type}_1",
        (
            f"{reset_validator.WORKSPACE_FIXTURE}_source_slot_"
            f"{row['mapping'][object_type]}_region"
        ),
    ]


def receiver_predicate(row: dict, region: str) -> list[str] | None:
    receiver = reset_validator.receiver_for_skill(row["skill"])
    if receiver is None:
        return None
    return [
        "on",
        receiver,
        f"{reset_validator.WORKSPACE_FIXTURE}_target_{region}_region",
    ]


def bddl_predicate_line(predicate: list[str]) -> str:
    name, *arguments = predicate
    return f"    ({name.title()} {' '.join(arguments)})"


def derived_initial_predicates(
    row: dict,
    case_type: str,
) -> list[list[str]]:
    alternate = ALTERNATE_REGION[row["spatial_region"]]
    predicates = []
    for object_type in reset_validator.OBJECT_TYPES:
        if object_type == row["object"]:
            if case_type in {
                CASE_POSITIVE,
                CASE_RECEIVER_REGION_NEGATIVE,
            }:
                predicate = positive_target_predicate(row)
            else:
                predicate = [
                    "on",
                    target_instance(row),
                    (
                        f"{reset_validator.WORKSPACE_FIXTURE}_target_"
                        f"{alternate}_region"
                    ),
                ]
        else:
            predicate = source_predicate(row, object_type)
        predicates.append(predicate)
    receiver_region = (
        alternate
        if case_type == CASE_RECEIVER_REGION_NEGATIVE
        else row["spatial_region"]
    )
    receiver = receiver_predicate(row, receiver_region)
    if receiver is not None:
        predicates.append(receiver)
    return predicates


def derive_bddl(
    row: dict,
    case_type: str,
) -> tuple[str, list[list[str]], str]:
    path = Path(row["bddl_path"])
    original = path.read_text(encoding="utf-8")
    if sha256_bytes(original.encode()) != row["bddl_sha256"]:
        raise ValueError(f"Original BDDL hash changed: {path}")
    predicates = derived_initial_predicates(row, case_type)
    original_target = source_predicate(row, row["object"])
    replacement_target = next(
        predicate
        for predicate in predicates
        if predicate[1] == target_instance(row)
    )
    old_line = bddl_predicate_line(original_target)
    new_line = bddl_predicate_line(replacement_target)
    if original.count(old_line) != 1:
        raise ValueError(
            f"Expected one target init predicate, found "
            f"{original.count(old_line)}"
        )
    derived = original.replace(old_line, new_line, 1)

    if case_type == CASE_RECEIVER_REGION_NEGATIVE:
        original_receiver = receiver_predicate(
            row,
            row["spatial_region"],
        )
        replacement_receiver = receiver_predicate(
            row,
            ALTERNATE_REGION[row["spatial_region"]],
        )
        if original_receiver is None or replacement_receiver is None:
            raise ValueError("Receiver negative requires a receiver")
        old_line = bddl_predicate_line(original_receiver)
        new_line = bddl_predicate_line(replacement_receiver)
        if derived.count(old_line) != 1:
            raise ValueError(
                "Expected exactly one receiver init predicate"
            )
        derived = derived.replace(old_line, new_line, 1)

    goal_marker = "  (:goal\n"
    original_goal = original[original.index(goal_marker) :]
    derived_goal = derived[derived.index(goal_marker) :]
    if derived_goal != original_goal:
        raise ValueError("Derived BDDL changed the goal section")
    algorithm = (
        "deterministic_bddl_init_substitution_using_libero_"
        "conditional_and_table_region_samplers"
    )
    return derived, predicates, algorithm


def expected_atoms(case_type: str) -> tuple[bool, bool, bool]:
    if case_type == CASE_POSITIVE:
        return True, True, True
    if case_type == CASE_RELATION_NEGATIVE:
        return False, True, False
    if case_type == CASE_RECEIVER_REGION_NEGATIVE:
        return True, False, False
    raise ValueError(f"Unknown case type: {case_type}")


def receiver_requested_region_ok(
    inner,
    poses,
    row: dict,
    receiver: str | None,
) -> bool:
    if receiver is None:
        return True
    details = reset_validator.receiver_region_details(
        inner,
        poses,
        receiver,
        row["receiver_region"],
    )
    return bool(details[receiver]["xy_in_bounds"])


def placement_atom(
    inner,
    poses,
    row: dict,
    case_type: str,
    relation: bool,
    receiver: str | None,
) -> bool:
    alternate = ALTERNATE_REGION[row["spatial_region"]]
    if case_type == CASE_POSITIVE:
        return relation
    if case_type == CASE_RELATION_NEGATIVE:
        region = (
            f"{reset_validator.WORKSPACE_FIXTURE}_target_"
            f"{alternate}_region"
        )
        official = reset_validator.state_on_region(
            inner,
            target_instance(row),
            region,
        )
        xy_ok = reset_validator.xy_in_bounds(
            poses[target_instance(row)][0],
            reset_validator.TARGET_REGION_RANGES[alternate],
        )
        return bool(official and xy_ok)
    if case_type == CASE_RECEIVER_REGION_NEGATIVE:
        if receiver is None:
            return False
        details = reset_validator.receiver_region_details(
            inner,
            poses,
            receiver,
            (
                f"{reset_validator.WORKSPACE_FIXTURE}_target_"
                f"{alternate}_region"
            ),
        )
        return bool(
            relation and details[receiver]["xy_in_bounds"]
        )
    raise ValueError(f"Unknown case type: {case_type}")


def unexpected_contacts(
    inner,
    row: dict,
    case_type: str,
    receiver: str | None,
) -> list[list[str]]:
    pairs = reset_validator.contact_pairs(inner, receiver)
    allowed = set()
    if (
        receiver is not None
        and case_type in {
            CASE_POSITIVE,
            CASE_RECEIVER_REGION_NEGATIVE,
        }
    ):
        allowed.add(frozenset([target_instance(row), receiver]))
    return [
        pair
        for pair in pairs
        if frozenset(pair) not in allowed
    ]


def receiver_table_contact(
    inner,
    receiver: str | None,
) -> bool:
    if receiver is None:
        return True
    return bool(
        inner.check_contact(
            inner.get_object(receiver),
            inner.get_object(reset_validator.WORKSPACE_FIXTURE),
        )
    )


def frame_record(
    env,
    obs,
    row: dict,
    case_type: str,
    receiver: str | None,
) -> dict:
    poses = reset_validator.capture_poses(obs, receiver)
    relation = bool(env.check_success())
    receiver_ok = receiver_requested_region_ok(
        env.env,
        poses,
        row,
        receiver,
    )
    composite = bool(relation and receiver_ok)
    placement = placement_atom(
        env.env,
        poses,
        row,
        case_type,
        relation,
        receiver,
    )
    contacts = unexpected_contacts(
        env.env,
        row,
        case_type,
        receiver,
    )
    table_contact = receiver_table_contact(env.env, receiver)
    sim_state = np.asarray(
        env.get_sim_state(),
        dtype=np.float64,
    ).copy()
    return {
        "obs": obs,
        "poses": poses,
        "relation": relation,
        "receiver": receiver_ok,
        "composite": composite,
        "placement": placement,
        "receiver_table_contact": table_contact,
        "unexpected_contacts": contacts,
        "finite": reset_validator.poses_are_finite(poses),
        "images_valid": reset_validator.images_valid(obs),
        "sim_state": sim_state,
        "sim_state_sha256": sha256_array(sim_state),
    }


def matches_expected(frame: dict, case_type: str) -> bool:
    expected_relation, expected_receiver, expected_composite = (
        expected_atoms(case_type)
    )
    return bool(
        frame["relation"] == expected_relation
        and frame["receiver"] == expected_receiver
        and frame["composite"] == expected_composite
        and frame["placement"]
        and frame["receiver_table_contact"]
        and not frame["unexpected_contacts"]
        and frame["finite"]
        and frame["images_valid"]
    )


def max_recent_motion(
    motion: list[float],
    orientation: list[float],
) -> tuple[float, float]:
    window = min(FINAL_WINDOW_STEPS, len(motion))
    if not window:
        return 0.0, 0.0
    return max(motion[-window:]), max(orientation[-window:])


def copy_preview_observation(obs) -> dict:
    return {
        key: np.asarray(obs[key]).copy()
        for key in ["agentview_image", "robot0_eye_in_hand_image"]
    }


def evaluate_once(
    bddl_path: Path,
    row: dict,
    case_type: str,
    expected_initial_predicates: list[list[str]],
    seed: int,
) -> dict:
    env = reset_validator.make_environment(bddl_path)
    try:
        obs, attempts, camera_record = reset_validator.safe_reset(
            env,
            seed,
            MAX_RESET_ATTEMPTS,
        )
        camera_config.validate_runtime_camera_record(camera_record)
        reset_validator.validate_parsed_semantics(env.env, row)
        actual_initial = reset_validator.normalize_predicates(
            env.env.parsed_problem["initial_state"]
        )
        expected_initial = reset_validator.normalize_predicates(
            expected_initial_predicates
        )
        initial_state_ok = actual_initial == expected_initial
        if not initial_state_ok:
            raise ValueError(
                f"Derived initial state mismatch: {actual_initial} != "
                f"{expected_initial}"
            )

        receiver = reset_validator.receiver_for_skill(row["skill"])
        initial_sim_state = np.asarray(
            env.get_sim_state(),
            dtype=np.float64,
        ).copy()
        zero = np.zeros(int(env.env.action_dim), dtype=np.float32)

        acquire_frames = []
        acquire_motion = []
        acquire_orientation = []
        previous_poses = reset_validator.capture_poses(obs, receiver)
        stable_streak = 0
        acquired_at = None
        acquired_frame = None
        for step in range(1, ACQUIRE_MAX_STEPS + 1):
            obs, _, _, _ = env.step(zero)
            frame = frame_record(
                env,
                obs,
                row,
                case_type,
                receiver,
            )
            translation, orientation = reset_validator.max_pose_motion(
                previous_poses,
                frame["poses"],
            )
            acquire_motion.append(translation)
            acquire_orientation.append(orientation)
            previous_poses = frame["poses"]
            acquire_frames.append(frame)
            if matches_expected(frame, case_type):
                stable_streak += 1
            else:
                stable_streak = 0
            recent_motion, recent_orientation = max_recent_motion(
                acquire_motion,
                acquire_orientation,
            )
            if (
                stable_streak >= ACQUIRE_STABLE_STEPS
                and recent_motion <= MAX_FINAL_WINDOW_MOTION_M
                and recent_orientation
                <= MAX_FINAL_WINDOW_ORIENTATION_DEG
            ):
                acquired_at = step
                acquired_frame = frame
                break

        acquired = acquired_at is not None
        if not acquired:
            acquired_frame = acquire_frames[-1]

        constructed_poses = acquired_frame["poses"]
        constructed_preview = copy_preview_observation(
            acquired_frame["obs"]
        )

        hold_frames = []
        hold_motion = []
        hold_orientation = []
        previous_poses = constructed_poses
        if acquired:
            for _ in range(HOLD_STEPS):
                obs, _, _, _ = env.step(zero)
                frame = frame_record(
                    env,
                    obs,
                    row,
                    case_type,
                    receiver,
                )
                translation, orientation = (
                    reset_validator.max_pose_motion(
                        previous_poses,
                        frame["poses"],
                    )
                )
                hold_motion.append(translation)
                hold_orientation.append(orientation)
                previous_poses = frame["poses"]
                hold_frames.append(frame)

        terminal_frame = (
            hold_frames[-1] if hold_frames else acquired_frame
        )
        terminal_sim_state = np.asarray(
            env.get_sim_state(),
            dtype=np.float64,
        ).copy()
        max_motion, max_orientation = max_recent_motion(
            hold_motion,
            hold_orientation,
        )

        expected_relation, expected_receiver, expected_composite = (
            expected_atoms(case_type)
        )
        relation_trace = [frame["relation"] for frame in hold_frames]
        receiver_trace = [frame["receiver"] for frame in hold_frames]
        composite_trace = [frame["composite"] for frame in hold_frames]
        placement_trace = [frame["placement"] for frame in hold_frames]
        receiver_table_trace = [
            frame["receiver_table_contact"] for frame in hold_frames
        ]
        contact_trace = [
            frame["unexpected_contacts"] for frame in hold_frames
        ]
        relation_hold_ok = bool(
            len(relation_trace) == HOLD_STEPS
            and all(value == expected_relation for value in relation_trace)
        )
        receiver_hold_ok = bool(
            len(receiver_trace) == HOLD_STEPS
            and all(value == expected_receiver for value in receiver_trace)
        )
        composite_hold_ok = bool(
            len(composite_trace) == HOLD_STEPS
            and all(
                value == expected_composite
                for value in composite_trace
            )
        )
        placement_hold_ok = bool(
            len(placement_trace) == HOLD_STEPS
            and all(placement_trace)
        )
        receiver_table_hold_ok = bool(
            len(receiver_table_trace) == HOLD_STEPS
            and all(receiver_table_trace)
        )
        no_unexpected = bool(
            len(contact_trace) == HOLD_STEPS
            and not any(contact_trace)
        )
        all_finite = bool(
            len(hold_frames) == HOLD_STEPS
            and all(frame["finite"] for frame in hold_frames)
        )
        all_images = bool(
            len(hold_frames) == HOLD_STEPS
            and all(frame["images_valid"] for frame in hold_frames)
        )
        stable = bool(
            len(hold_frames) == HOLD_STEPS
            and max_motion <= MAX_FINAL_WINDOW_MOTION_M
            and max_orientation <= MAX_FINAL_WINDOW_ORIENTATION_DEG
        )
        gates = {
            "parsed_semantics": True,
            "derived_initial_state": initial_state_ok,
            "oracle_state_acquired": acquired,
            "relation_hold": relation_hold_ok,
            "receiver_region_hold": receiver_hold_ok,
            "composite_hold": composite_hold_ok,
            "placement_hold": placement_hold_ok,
            "receiver_table_contact_hold": receiver_table_hold_ok,
            "no_unexpected_contacts": no_unexpected,
            "finite_poses": all_finite,
            "valid_images": all_images,
            "stable_final_window": stable,
        }
        failed = [name for name, value in gates.items() if not value]
        trace_signature = {
            "acquired_at_step": acquired_at,
            "camera_record": camera_record,
            "acquire": [
                {
                    "relation": frame["relation"],
                    "receiver": frame["receiver"],
                    "composite": frame["composite"],
                    "placement": frame["placement"],
                    "receiver_table_contact": frame[
                        "receiver_table_contact"
                    ],
                    "contacts": frame["unexpected_contacts"],
                    "finite": frame["finite"],
                    "images_valid": frame["images_valid"],
                    "sim_state_sha256": frame["sim_state_sha256"],
                }
                for frame in acquire_frames
            ],
            "hold": [
                {
                    "relation": frame["relation"],
                    "receiver": frame["receiver"],
                    "composite": frame["composite"],
                    "placement": frame["placement"],
                    "receiver_table_contact": frame[
                        "receiver_table_contact"
                    ],
                    "contacts": frame["unexpected_contacts"],
                    "finite": frame["finite"],
                    "images_valid": frame["images_valid"],
                    "sim_state_sha256": frame["sim_state_sha256"],
                }
                for frame in hold_frames
            ],
        }
        return {
            "reset_attempts": attempts,
            "camera_record": camera_record,
            "initial_state_ok": initial_state_ok,
            "initial_sim_state": initial_sim_state,
            "terminal_sim_state": terminal_sim_state,
            "acquired": acquired,
            "acquired_at_step": acquired_at,
            "acquire_frames": acquire_frames,
            "hold_frames": hold_frames,
            "relation_trace": relation_trace,
            "receiver_trace": receiver_trace,
            "composite_trace": composite_trace,
            "placement_trace": placement_trace,
            "receiver_table_trace": receiver_table_trace,
            "contact_trace": contact_trace,
            "relation_hold_ok": relation_hold_ok,
            "receiver_hold_ok": receiver_hold_ok,
            "composite_hold_ok": composite_hold_ok,
            "placement_hold_ok": placement_hold_ok,
            "receiver_table_hold_ok": receiver_table_hold_ok,
            "no_unexpected": no_unexpected,
            "all_finite": all_finite,
            "all_images": all_images,
            "max_motion": max_motion,
            "max_orientation": max_orientation,
            "stable": stable,
            "constructed_poses": constructed_poses,
            "terminal_poses": terminal_frame["poses"],
            "constructed_preview": constructed_preview,
            "terminal_preview": copy_preview_observation(
                terminal_frame["obs"]
            ),
            "trace_signature": trace_signature,
            "failed": failed,
        }
    finally:
        env.close()


def max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if not left.size:
        return 0.0
    return float(np.max(np.abs(left - right)))


def trajectory_max_difference(
    first_frames: list[dict],
    repeated_frames: list[dict],
) -> float:
    if len(first_frames) != len(repeated_frames):
        return float("inf")
    if not first_frames:
        return 0.0
    return max(
        max_abs_difference(
            first["sim_state"],
            repeated["sim_state"],
        )
        for first, repeated in zip(first_frames, repeated_frames)
    )


def save_previews(
    preview_dir: Path,
    row: dict,
    case_type: str,
    result: dict,
) -> dict:
    from PIL import Image

    preview_dir.mkdir(parents=True, exist_ok=True)
    stem = f"task_{row['task_id']:03d}_{case_type}"
    artifacts = {}
    for stage in ["constructed", "terminal"]:
        observation = result[f"{stage}_preview"]
        for camera, key in [
            ("agentview", "agentview_image"),
            ("wrist", "robot0_eye_in_hand_image"),
        ]:
            path = preview_dir / f"{stem}_{stage}_{camera}.png"
            Image.fromarray(observation[key]).save(path)
            artifacts[f"{stage}_{camera}"] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }
    return artifacts


def base_output_row(
    args,
    row: dict,
    case_type: str,
    seed: int,
    derived_sha256: str,
    predicates: list[list[str]],
    algorithm: str,
) -> dict:
    expected_relation, expected_receiver, expected_composite = (
        expected_atoms(case_type)
    )
    receiver = reset_validator.receiver_for_skill(row["skill"])
    output = {field: "" for field in OUTPUT_FIELDS}
    output.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "benchmark_name": BENCHMARK_NAME,
            "gate_protocol_version": GATE_PROTOCOL_VERSION,
            "gate_id": GATE_ID,
            "mode": args.mode,
            "task_id": row["task_id"],
            "layout_id": row["layout_id"],
            "tuple": row["tuple"],
            "object": row["object"],
            "skill": row["skill"],
            "spatial_region": row["spatial_region"],
            "case_type": case_type,
            "case_index": CASE_INDEX[case_type],
            "seed": seed,
            "target_instance": target_instance(row),
            "receiver_instance": receiver or "",
            "alternate_region": ALTERNATE_REGION[
                row["spatial_region"]
            ],
            "original_bddl_path": str(row["bddl_path"]),
            "original_bddl_sha256": row["bddl_sha256"],
            "derived_bddl_sha256": derived_sha256,
            "object_to_slot_json": json_compact(row["mapping"]),
            "derived_init_predicates_json": json_compact(predicates),
            "construction_algorithm": algorithm,
            "expected_relation": expected_relation,
            "expected_receiver_region": expected_receiver,
            "expected_composite_success": expected_composite,
            "trajectory_constraints_evaluated": False,
            "push_threshold_status": (
                "not_evaluated_gate5_required"
                if row["skill"] == "push_to"
                else "not_applicable"
            ),
            "determinism_checked": True,
            "passed": False,
        }
    )
    return output


def run_case(
    args,
    row: dict,
    case_type: str,
    temporary_dir: Path,
    preview_dir: Path | None,
) -> dict:
    case_index = CASE_INDEX[case_type]
    seed = SEED_BASE + row["task_id"] * 10 + case_index
    derived, predicates, algorithm = derive_bddl(row, case_type)
    derived_sha256 = sha256_bytes(derived.encode())
    bddl_path = temporary_dir / (
        f"task_{row['task_id']:03d}_{case_type}.bddl"
    )
    bddl_path.write_text(derived, encoding="utf-8")
    output = base_output_row(
        args,
        row,
        case_type,
        seed,
        derived_sha256,
        predicates,
        algorithm,
    )
    try:
        first = evaluate_once(
            bddl_path,
            row,
            case_type,
            predicates,
            seed,
        )
        repeated = evaluate_once(
            bddl_path,
            row,
            case_type,
            predicates,
            seed,
        )
        initial_difference = max_abs_difference(
            first["initial_sim_state"],
            repeated["initial_sim_state"],
        )
        terminal_difference = max_abs_difference(
            first["terminal_sim_state"],
            repeated["terminal_sim_state"],
        )
        acquire_trajectory_difference = trajectory_max_difference(
            first["acquire_frames"],
            repeated["acquire_frames"],
        )
        hold_trajectory_difference = trajectory_max_difference(
            first["hold_frames"],
            repeated["hold_frames"],
        )
        trajectory_difference = max(
            acquire_trajectory_difference,
            hold_trajectory_difference,
        )
        trace_match = (
            first["trace_signature"] == repeated["trace_signature"]
        )
        deterministic = bool(
            first["reset_attempts"] == repeated["reset_attempts"]
            and initial_difference <= 1e-8
            and terminal_difference <= 1e-8
            and trajectory_difference <= 1e-8
            and trace_match
        )
        failed = list(first["failed"])
        if repeated["failed"]:
            failed.append("repeated_case_failed")
        if not deterministic:
            failed.append("deterministic_replay")
        failed = sorted(set(failed))
        preview_artifacts = {}
        if preview_dir is not None:
            preview_artifacts = save_previews(
                preview_dir,
                row,
                case_type,
                first,
            )
        acquire_frames = first["acquire_frames"]
        output.update(
            {
                "reset_attempts": first["reset_attempts"],
                "frozen_camera_record_json": json_compact(
                    first["camera_record"]
                ),
                "parsed_semantics_ok": True,
                "derived_initial_state_ok": first["initial_state_ok"],
                "acquired": first["acquired"],
                "acquired_at_step": first["acquired_at_step"],
                "acquire_relation_trace_json": json_compact(
                    [frame["relation"] for frame in acquire_frames]
                ),
                "acquire_receiver_region_trace_json": json_compact(
                    [frame["receiver"] for frame in acquire_frames]
                ),
                "acquire_composite_trace_json": json_compact(
                    [frame["composite"] for frame in acquire_frames]
                ),
                "relation_trace_json": json_compact(
                    first["relation_trace"]
                ),
                "receiver_region_trace_json": json_compact(
                    first["receiver_trace"]
                ),
                "composite_success_trace_json": json_compact(
                    first["composite_trace"]
                ),
                "placement_trace_json": json_compact(
                    first["placement_trace"]
                ),
                "receiver_table_contact_trace_json": json_compact(
                    first["receiver_table_trace"]
                ),
                "unexpected_contact_trace_json": json_compact(
                    first["contact_trace"]
                ),
                "relation_hold_ok": first["relation_hold_ok"],
                "receiver_region_hold_ok": first["receiver_hold_ok"],
                "composite_hold_ok": first["composite_hold_ok"],
                "placement_hold_ok": first["placement_hold_ok"],
                "receiver_table_contact_hold_ok": first[
                    "receiver_table_hold_ok"
                ],
                "no_unexpected_contacts": first["no_unexpected"],
                "all_finite_poses": first["all_finite"],
                "all_images_valid": first["all_images"],
                "max_final_window_motion_m": first["max_motion"],
                "max_final_window_orientation_deg": first[
                    "max_orientation"
                ],
                "stable_final_window": first["stable"],
                "constructed_poses_json": (
                    reset_validator.poses_to_json(
                        first["constructed_poses"]
                    )
                ),
                "terminal_poses_json": reset_validator.poses_to_json(
                    first["terminal_poses"]
                ),
                "preview_artifacts_json": json_compact(
                    preview_artifacts
                ),
                "repeated_reset_attempts": repeated["reset_attempts"],
                "determinism_initial_state_max_abs": initial_difference,
                "determinism_terminal_state_max_abs": (
                    terminal_difference
                ),
                "determinism_trajectory_state_max_abs": (
                    trajectory_difference
                ),
                "determinism_trace_match": trace_match,
                "deterministic": deterministic,
                "failed_gate_checks_json": json_compact(failed),
                "passed": not failed,
                "error": "",
            }
        )
    except Exception as error:
        output.update(
            {
                "failed_gate_checks_json": json_compact(["exception"]),
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
            }
        )
    return output


def exact_coverage(rows: list[dict], selected_rows: list[dict]) -> bool:
    expected = {
        (row["task_id"], case_type)
        for row in selected_rows
        for case_type in cases_for_skill(row["skill"])
    }
    actual = {
        (int(row["task_id"]), row["case_type"])
        for row in rows
    }
    return actual == expected and len(rows) == len(expected)


def make_manifest(
    args,
    layout_spec: Path,
    reset_manifest: Path,
    reset_gate_manifest: dict,
    output: Path,
    selected_rows: list[dict],
    results: list[dict],
    run_complete: bool,
) -> dict:
    expected_cases = expected_case_count(selected_rows)
    passed = sum(bool(row["passed"]) for row in results)
    counts = Counter(
        (row["skill"], row["case_type"], bool(row["passed"]))
        for row in results
    )
    count_rows = [
        {
            "skill": skill,
            "case_type": case_type,
            "passed": outcome,
            "count": count,
        }
        for (skill, case_type, outcome), count in sorted(counts.items())
    ]
    coverage_ok = exact_coverage(results, selected_rows)
    full_shape = bool(
        args.mode == "full"
        and expected_cases == FULL_EXPECTED_CASES
        and len(results) == FULL_EXPECTED_CASES
        and coverage_ok
    )
    gate4_passed = bool(
        run_complete
        and full_shape
        and passed == FULL_EXPECTED_CASES
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_name": BENCHMARK_NAME,
        "gate_protocol_version": GATE_PROTOCOL_VERSION,
        "gate_id": GATE_ID,
        "mode": args.mode,
        "git_commit": git_commit(),
        "validator_path": str(Path(__file__).resolve()),
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "camera_helper_path": str(Path(camera_config.__file__).resolve()),
        "camera_helper_sha256": sha256_file(
            Path(camera_config.__file__).resolve()
        ),
        "reset_validator_path": str(
            Path(reset_validator.__file__).resolve()
        ),
        "reset_validator_sha256": sha256_file(
            Path(reset_validator.__file__).resolve()
        ),
        "frozen_observation_camera": camera_config.frozen_camera_spec(),
        "layout_spec": str(layout_spec),
        "layout_spec_sha256": sha256_file(layout_spec),
        "reset_manifest": str(reset_manifest),
        "reset_manifest_sha256": sha256_file(reset_manifest),
        "reset_output": reset_gate_manifest["validated_output_path"],
        "reset_output_sha256": reset_gate_manifest[
            "validated_output_sha256"
        ],
        "gate3_reset_gate_required": True,
        "output": str(output),
        "output_sha256": (
            sha256_file(output) if output.is_file() else None
        ),
        "preview_dir": (
            None
            if args.preview_dir is None
            else str(resolve_repo_path(args.preview_dir))
        ),
        "software_versions": {
            name: package_version(name)
            for name in [
                "hf_libero",
                "libero",
                "robosuite",
                "mujoco",
                "numpy",
            ]
        },
        "parameters": {
            "canonical_layout_id": CANONICAL_LAYOUT_ID,
            "task_ids": [row["task_id"] for row in selected_rows],
            "hold_steps": HOLD_STEPS,
            "acquire_max_steps": ACQUIRE_MAX_STEPS,
            "acquire_stable_steps": ACQUIRE_STABLE_STEPS,
            "final_window_steps": FINAL_WINDOW_STEPS,
            "seed_base": SEED_BASE,
            "max_reset_attempts": MAX_RESET_ATTEMPTS,
            "max_final_window_motion_m": (
                MAX_FINAL_WINDOW_MOTION_M
            ),
            "max_final_window_orientation_deg": (
                MAX_FINAL_WINDOW_ORIENTATION_DEG
            ),
            "alternate_region_mapping": ALTERNATE_REGION,
            "case_matrix": {
                "put_on_top": cases_for_skill("put_on_top"),
                "put_inside": cases_for_skill("put_inside"),
                "push_to": cases_for_skill("push_to"),
            },
            "success_definition": (
                "exact_bddl_goal_and_receiver_requested_region_xy_"
                "for_put_tasks"
            ),
            "derived_bddl_contract": (
                "only_init_placement_predicates_change"
            ),
            "deterministic_replay": True,
            "upstream_interactive_debugger_disabled": True,
        },
        "expected_cases": expected_cases,
        "completed_cases": len(results),
        "passed_cases": passed,
        "failed_cases": len(results) - passed,
        "counts": count_rows,
        "run_complete": run_complete,
        "exact_case_coverage": coverage_ok,
        "full_protocol_shape": full_shape,
        "gate4_passed": gate4_passed,
        "push_trajectory_constraints_evaluated": False,
        "push_lift_threshold_calibrated": False,
        "physical_feasibility_gate_not_covered": True,
        "all_pretraining_gates_passed": False,
    }


def main():
    args = parse_args()
    layout_spec = resolve_repo_path(args.layout_spec)
    reset_manifest = resolve_repo_path(args.reset_manifest)
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
    if output.exists() or manifest.exists():
        if not args.overwrite:
            raise FileExistsError(
                "Output or manifest already exists; pass --overwrite "
                "only after confirming replacement is intended"
            )
        output.unlink(missing_ok=True)
        manifest.unlink(missing_ok=True)

    disable_upstream_interactive_debugger()
    validate_runtime_versions()
    reset_gate_manifest = validate_reset_gate(
        reset_manifest,
        layout_spec,
    )
    selected_rows = selected_layout_rows(layout_spec, args)
    expected_cases = expected_case_count(selected_rows)

    results = []
    write_json_atomic(
        manifest,
        make_manifest(
            args,
            layout_spec,
            reset_manifest,
            reset_gate_manifest,
            output,
            selected_rows,
            results,
            run_complete=False,
        ),
    )
    print("=" * 80)
    print("LIBERO 36 Gate-4 terminal goal validation")
    print("mode:", args.mode)
    print("logical tasks:", len(selected_rows))
    print("canonical layout:", CANONICAL_LAYOUT_ID)
    print("expected cases:", expected_cases)
    print("hold steps:", HOLD_STEPS)
    print("deterministic replay:", True)
    print("output:", output)
    print("manifest:", manifest)
    print(
        "NOTE: push trajectory constraints are not evaluated by Gate 4."
    )

    run_index = 0
    with tempfile.TemporaryDirectory(
        prefix="libero36_gate4_"
    ) as temporary:
        temporary_dir = Path(temporary)
        for row in selected_rows:
            for case_type in cases_for_skill(row["skill"]):
                run_index += 1
                result = run_case(
                    args,
                    row,
                    case_type,
                    temporary_dir,
                    preview_dir,
                )
                results.append(result)
                write_csv_atomic(output, results)
                write_json_atomic(
                    manifest,
                    make_manifest(
                        args,
                        layout_spec,
                        reset_manifest,
                        reset_gate_manifest,
                        output,
                        selected_rows,
                        results,
                        run_complete=False,
                    ),
                )
                print(
                    f"[{run_index:03d}/{expected_cases:03d}] "
                    f"task={row['task_id']:02d} "
                    f"skill={row['skill']} "
                    f"case={case_type} "
                    f"passed={result['passed']} "
                    f"acquired={result['acquired']} "
                    f"error={result['error']}"
                )
                if args.fail_fast and not bool(result["passed"]):
                    raise RuntimeError(
                        "LIBERO 36 Gate-4 validation failed with "
                        "--fail-fast"
                    )

    final_manifest = make_manifest(
        args,
        layout_spec,
        reset_manifest,
        reset_gate_manifest,
        output,
        selected_rows,
        results,
        run_complete=True,
    )
    write_json_atomic(manifest, final_manifest)
    passed = sum(bool(row["passed"]) for row in results)
    print("=" * 80)
    print("LIBERO 36 Gate-4 validation complete")
    print("cases:", len(results))
    print("passed:", passed)
    print("failed:", len(results) - passed)
    print("gate4 passed:", final_manifest["gate4_passed"])
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
