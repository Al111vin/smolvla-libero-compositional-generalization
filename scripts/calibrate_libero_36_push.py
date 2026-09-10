from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pdb
import subprocess
from collections import Counter
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glx")

import numpy as np
from robosuite.utils import transform_utils as transform

try:
    import libero_36_camera as camera_config
    import validate_libero_36_envs as reset_validator
    import validate_libero_36_goals as goal_validator
except ModuleNotFoundError:
    from scripts import libero_36_camera as camera_config
    from scripts import validate_libero_36_envs as reset_validator
    from scripts import validate_libero_36_goals as goal_validator


PROTOCOL_VERSION = "libero_36_proxy_tabletop_draft_v5"
BENCHMARK_NAME = "libero_registered_object_36_proxy_tabletop"
CALIBRATION_PROTOCOL = "libero_36_push_calibration_pilot_v2"
CALIBRATION_ID = "gate5a_push_trajectory_calibration_pilot_v2"

DEFAULT_LAYOUT_SPEC = Path("data/libero_36/layout_spec.csv")
DEFAULT_RESET_MANIFEST = Path(
    "results/libero_36_reset_audit.manifest.json"
)
DEFAULT_GOAL_MANIFEST = Path(
    "results/libero_36_goal_audit.manifest.json"
)
DEFAULT_OUTPUT = Path("results/libero_36_push_calibration.csv")

CALIBRATION_LAYOUT_ID = 1
FULL_PUSH_TASK_IDS = tuple(
    object_index * 9 + 6 + region_index
    for object_index in range(4)
    for region_index in range(3)
)
SETTLE_STEPS = 20
TERMINAL_HOLD_STEPS = 20
FINAL_WINDOW_STEPS = 5
MAX_RESET_ATTEMPTS = 100
GATE3_CALIBRATION_RESET_INDEX = 0
STATE_REPLAY_ATOL = 1e-8
POSITION_TOLERANCE_M = 0.006
ORIENTATION_TOLERANCE_RAD = 0.04
STABLE_POSE_STEPS = 4
MAX_TRANSIT_STEPS = 100
MAX_CONTACT_SEARCH_M = 0.085
CONTACT_SEARCH_INCREMENT_M = 0.0025
PUSH_INCREMENT_M = 0.006
PUSH_OVERSHOOT_M = 0.045
MAX_PUSH_STEPS = 120
MAX_ACTIVE_ACTIONS = 930
MAX_LIFT_PROVISIONAL_M = 0.03
MAX_FINAL_MOTION_M = 0.001
MAX_FINAL_ORIENTATION_DEG = 1.0
ROTATION_AWARE_PAD_TARGET = False

# Diagnostic controller geometry. The old pilot placed every high
# transit waypoint 75 mm behind the object at a fixed world z of 1.10
# m. At the outer source slots this combined two reach-extending
# offsets and put the Panda beyond its Cartesian workspace before the
# contact search could begin. The revised pilot first moves above the
# live target center, then approaches a compact pre-contact point. The
# standoff is grounded in the frozen source sampler radius plus a
# conservative fingerpad / non-contact clearance. These are controller
# search parameters only; none of the Gate-5 acceptance thresholds are
# changed.
PRECONTACT_PAD_CLEARANCE_M = 0.015
TRANSIT_PAD_CLEARANCE_M = 0.10

# Candidate height is the selected fingerpad center relative to the
# target object's live body origin. These values are diagnostic search
# parameters, not benchmark constants or promoted thresholds.
BASE_PAD_HEIGHT_OFFSET_M = {
    "akita_black_bowl": 0.025,
    "white_yellow_mug": 0.025,
    "alphabet_soup": 0.000,
    "cream_cheese": 0.000,
}
HEIGHT_DELTAS_M = (-0.01, 0.0, 0.01)
PUSH_FINGERS = ("left", "right")


OUTPUT_FIELDS = [
    "protocol_version",
    "benchmark_name",
    "calibration_protocol",
    "calibration_id",
    "diagnostic_only",
    "mode",
    "task_id",
    "layout_id",
    "tuple",
    "object",
    "skill",
    "spatial_region",
    "target_instance",
    "seed",
    "gate3_reset_index",
    "candidate_index",
    "pusher_finger",
    "pad_height_offset_m",
    "candidate_order_key",
    "bddl_path",
    "bddl_sha256",
    "object_to_slot_json",
    "reset_attempts",
    "replay_reset_attempts",
    "camera_record_json",
    "initial_sim_state_sha256",
    "passive_settle_state_sha256",
    "passive_z_min_m",
    "passive_z_max_m",
    "passive_z_range_m",
    "max_passive_positive_z_noise_m",
    "stabilized_target_z_m",
    "settle_final_window_table_support_ok",
    "settle_final_window_source_official_ok",
    "settle_final_window_source_xy_ok",
    "max_settle_horizontal_drift_m",
    "max_settle_orientation_drift_deg",
    "selected_pad_initial_offset_json",
    "opposite_pad_initial_offset_json",
    "push_direction_json",
    "destination_xy_json",
    "approach_standoff_m",
    "transit_pad_clearance_m",
    "transit_pad_xyz_json",
    "behind_pad_xyz_json",
    "transit_min_pad_error_m",
    "transit_final_pad_error_m",
    "transit_min_rotation_error_rad",
    "transit_final_rotation_error_rad",
    "transit_longest_valid_run",
    "behind_min_pad_error_m",
    "behind_final_pad_error_m",
    "behind_min_rotation_error_rad",
    "behind_final_rotation_error_rad",
    "behind_longest_valid_run",
    "contact_acquired",
    "contact_acquired_step",
    "relation_ever_true",
    "terminal_hold_complete",
    "terminal_relation_hold_ok",
    "terminal_xy_hold_ok",
    "terminal_table_support_hold_ok",
    "terminal_stable",
    "max_final_window_motion_m",
    "max_final_window_orientation_deg",
    "selected_pad_contact_steps",
    "opposite_pad_contact_steps",
    "bilateral_contact_steps",
    "grasp_proxy_steps",
    "opposite_pad_never_contacts",
    "unexpected_object_contact_steps",
    "robot_distractor_contact_steps",
    "nonselected_robot_target_contact_steps",
    "no_bilateral_contact_or_grasp",
    "target_table_support_all_active_steps",
    "max_target_lift_m",
    "provisional_lift_check_passed",
    "provisional_lift_threshold_m",
    "trajectory_threshold_promoted",
    "original_steps",
    "replay_steps",
    "replay_raw_initial_state_max_abs",
    "replay_settle_trajectory_state_max_abs",
    "replay_settled_state_max_abs",
    "replay_initial_state_max_abs",
    "replay_trajectory_state_max_abs",
    "replay_terminal_state_max_abs",
    "replay_action_max_abs",
    "replay_actions_byte_exact",
    "replay_reset_attempts_match",
    "replay_camera_record_match",
    "replay_trace_match",
    "deterministic_replay",
    "npz_path",
    "npz_sha256",
    "attempt_succeeded",
    "failed_checks_json",
    "error",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostic-only Gate-5A one-finger scripted "
            "push calibration pilot on frozen LIBERO-36 layout 1."
        )
    )
    parser.add_argument(
        "--mode", choices=["smoke", "full"], required=True
    )
    parser.add_argument(
        "--layout-spec", type=Path, default=DEFAULT_LAYOUT_SPEC
    )
    parser.add_argument(
        "--reset-manifest", type=Path, default=DEFAULT_RESET_MANIFEST
    )
    parser.add_argument(
        "--goal-manifest", type=Path, default=DEFAULT_GOAL_MANIFEST
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--attempts-dir", type=Path)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    values = np.asarray(array, dtype=np.float64)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def json_compact(value) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":")
    )


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


def write_json_atomic(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def write_npz_atomic(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def empty_attempt_arrays() -> dict[str, np.ndarray]:
    return {
        "actions": np.empty((0, 7), dtype=np.float32),
        "states": np.empty((0, 0), dtype=np.float64),
        "raw_initial_state": np.empty((0,), dtype=np.float64),
        "settle_actions": np.empty((0, 7), dtype=np.float32),
        "settle_states": np.empty((0, 0), dtype=np.float64),
        "settle_target_table_support": np.empty(
            (0,), dtype=np.bool_
        ),
        "settle_source_official": np.empty((0,), dtype=np.bool_),
        "settle_source_xy": np.empty((0,), dtype=np.bool_),
        "replay_raw_initial_state": np.empty((0,), dtype=np.float64),
        "replay_settle_actions": np.empty((0, 7), dtype=np.float32),
        "replay_settle_states": np.empty((0, 0), dtype=np.float64),
        "replay_settle_target_table_support": np.empty(
            (0,), dtype=np.bool_
        ),
        "replay_settle_source_official": np.empty(
            (0,), dtype=np.bool_
        ),
        "replay_settle_source_xy": np.empty((0,), dtype=np.bool_),
        "replay_actions": np.empty((0, 7), dtype=np.float32),
        "replay_states": np.empty((0, 0), dtype=np.float64),
        "phase": np.empty((0,), dtype="U32"),
        "target_xyz": np.empty((0, 3), dtype=np.float64),
        "eef_xyz": np.empty((0, 3), dtype=np.float64),
        "left_contact": np.empty((0,), dtype=np.bool_),
        "right_contact": np.empty((0,), dtype=np.bool_),
        "grasp_proxy": np.empty((0,), dtype=np.bool_),
        "table_support": np.empty((0,), dtype=np.bool_),
        "unexpected_object_contact": np.empty((0,), dtype=np.bool_),
        "robot_distractor_contact": np.empty((0,), dtype=np.bool_),
        "nonselected_robot_target_contact": np.empty(
            (0,), dtype=np.bool_
        ),
        "robot_target_contact_pairs": np.empty((0,), dtype="U1024"),
        "relation": np.empty((0,), dtype=np.bool_),
        "xy_in_target": np.empty((0,), dtype=np.bool_),
        "replay_left_contact": np.empty((0,), dtype=np.bool_),
        "replay_right_contact": np.empty((0,), dtype=np.bool_),
        "replay_grasp_proxy": np.empty((0,), dtype=np.bool_),
        "replay_table_support": np.empty((0,), dtype=np.bool_),
        "replay_unexpected_object_contact": np.empty(
            (0,), dtype=np.bool_
        ),
        "replay_robot_distractor_contact": np.empty(
            (0,), dtype=np.bool_
        ),
        "replay_nonselected_robot_target_contact": np.empty(
            (0,), dtype=np.bool_
        ),
        "replay_robot_target_contact_pairs": np.empty(
            (0,), dtype="U1024"
        ),
        "replay_relation": np.empty((0,), dtype=np.bool_),
        "replay_xy_in_target": np.empty((0,), dtype=np.bool_),
    }


def prepare_output_paths(
    output: Path,
    manifest: Path,
    attempts_dir: Path,
    overwrite: bool,
):
    existing = output.exists() or manifest.exists()
    attempt_files = (
        sorted(attempts_dir.glob("*.npz"))
        if attempts_dir.exists()
        else []
    )
    unknown_attempt_files = (
        [path for path in attempts_dir.iterdir() if path.suffix != ".npz"]
        if attempts_dir.exists()
        else []
    )
    if unknown_attempt_files:
        raise ValueError(
            "Attempts directory contains non-NPZ files; refusing to "
            f"modify it: {unknown_attempt_files[:3]}"
        )
    if (existing or attempt_files) and not overwrite:
        raise FileExistsError(
            "Output artifacts already exist; use --overwrite only "
            "after confirming replacement is intended"
        )
    if overwrite:
        output.unlink(missing_ok=True)
        manifest.unlink(missing_ok=True)
        for path in attempt_files:
            path.unlink()
    attempts_dir.mkdir(parents=True, exist_ok=True)


def validate_gate4(
    path: Path,
    layout_spec: Path,
    reset_manifest: Path,
    reset_prerequisite: dict,
) -> dict:
    with path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    required = {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_name": BENCHMARK_NAME,
        "gate_protocol_version": goal_validator.GATE_PROTOCOL_VERSION,
        "gate_id": goal_validator.GATE_ID,
        "mode": "full",
        "expected_cases": 96,
        "completed_cases": 96,
        "passed_cases": 96,
        "failed_cases": 0,
        "run_complete": True,
        "exact_case_coverage": True,
        "full_protocol_shape": True,
        "gate4_passed": True,
        "push_trajectory_constraints_evaluated": False,
        "push_lift_threshold_calibrated": False,
        "physical_feasibility_gate_not_covered": True,
        "all_pretraining_gates_passed": False,
    }
    changed = {
        key: (manifest.get(key), expected)
        for key, expected in required.items()
        if manifest.get(key) != expected
    }
    if changed:
        raise ValueError(f"Gate-4 manifest is invalid: {changed}")
    hash_checks = {
        "layout_spec_sha256": sha256_file(layout_spec),
        "camera_helper_sha256": sha256_file(
            Path(camera_config.__file__).resolve()
        ),
        "reset_validator_sha256": sha256_file(
            Path(reset_validator.__file__).resolve()
        ),
        "validator_sha256": sha256_file(
            Path(goal_validator.__file__).resolve()
        ),
    }
    changed_hashes = {
        key: (manifest.get(key), value)
        for key, value in hash_checks.items()
        if manifest.get(key) != value
    }
    if changed_hashes:
        raise ValueError(
            f"Gate-4 prerequisite hashes changed: {changed_hashes}"
        )
    if manifest.get("frozen_observation_camera") != (
        camera_config.frozen_camera_spec()
    ):
        raise ValueError("Gate-4 frozen camera record changed")
    recorded_versions = manifest.get("software_versions")
    if not isinstance(recorded_versions, dict):
        raise ValueError("Gate-4 software versions are missing")
    changed_versions = {
        name: (value, goal_validator.package_version(name))
        for name, value in recorded_versions.items()
        if value != goal_validator.package_version(name)
    }
    if changed_versions:
        raise ValueError(
            f"Gate-4 software versions changed: {changed_versions}"
        )
    prerequisite_links = {
        "reset_manifest_sha256": sha256_file(reset_manifest),
        "reset_output_sha256": reset_prerequisite[
            "validated_output_sha256"
        ],
    }
    changed_links = {
        key: (manifest.get(key), value)
        for key, value in prerequisite_links.items()
        if manifest.get(key) != value
    }
    if changed_links:
        raise ValueError(
            "Gate-4 does not bind the current validated Gate-3 "
            f"artifacts: {changed_links}"
        )
    output_value = manifest.get("output")
    if not output_value:
        raise ValueError("Gate-4 output path is missing")
    output = Path(output_value)
    if not output.is_absolute():
        output = resolve_repo_path(output)
    if not output.is_file():
        raise FileNotFoundError(f"Gate-4 CSV is missing: {output}")
    if manifest.get("output_sha256") != sha256_file(output):
        raise ValueError("Gate-4 output hash does not match")
    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    required_columns = {
        "protocol_version",
        "benchmark_name",
        "gate_protocol_version",
        "gate_id",
        "mode",
        "task_id",
        "layout_id",
        "skill",
        "case_type",
        "case_index",
        "trajectory_constraints_evaluated",
        "push_threshold_status",
        "determinism_checked",
        "deterministic",
        "failed_gate_checks_json",
        "passed",
        "error",
    }
    missing_columns = required_columns - set(rows[0] if rows else {})
    if missing_columns:
        raise ValueError(
            f"Gate-4 CSV columns are missing: {sorted(missing_columns)}"
        )
    if len(rows) != 96 or any(
        row.get("protocol_version") != PROTOCOL_VERSION
        or row.get("benchmark_name") != BENCHMARK_NAME
        or row.get("gate_protocol_version")
        != goal_validator.GATE_PROTOCOL_VERSION
        or row.get("gate_id") != goal_validator.GATE_ID
        or row.get("mode") != "full"
        or str(row.get("passed", "")).strip().lower() != "true"
        or row.get("error") not in {"", None}
        for row in rows
    ):
        raise ValueError("Gate-4 CSV is incomplete or contains failures")
    skill_order = ["put_on_top", "put_inside", "push_to"]
    for row in rows:
        task_id = int(row["task_id"])
        expected_skill = skill_order[(task_id // 3) % 3]
        expected_threshold_status = (
            "not_evaluated_gate5_required"
            if expected_skill == "push_to"
            else "not_applicable"
        )
        semantic_checks = {
            "canonical_layout": int(row["layout_id"]) == 0,
            "skill": row["skill"] == expected_skill,
            "case_index": int(row["case_index"])
            == goal_validator.CASE_INDEX[row["case_type"]],
            "trajectory_not_evaluated": str(
                row["trajectory_constraints_evaluated"]
            ).strip().lower()
            == "false",
            "push_threshold_status": row["push_threshold_status"]
            == expected_threshold_status,
            "determinism_checked": str(
                row["determinism_checked"]
            ).strip().lower()
            == "true",
            "deterministic": str(row["deterministic"])
            .strip()
            .lower()
            == "true",
            "no_failed_checks": json.loads(
                row["failed_gate_checks_json"]
            )
            == [],
        }
        failed_semantics = [
            name for name, passed in semantic_checks.items() if not passed
        ]
        if failed_semantics:
            raise ValueError(
                "Gate-4 CSV semantic evidence is invalid for "
                f"task {task_id}: {failed_semantics}"
            )
    case_counts = Counter(row["case_type"] for row in rows)
    expected_case_counts = Counter(
        {
            goal_validator.CASE_POSITIVE: 36,
            goal_validator.CASE_RELATION_NEGATIVE: 36,
            goal_validator.CASE_RECEIVER_REGION_NEGATIVE: 24,
        }
    )
    if case_counts != expected_case_counts:
        raise ValueError(
            f"Gate-4 case matrix changed: {dict(case_counts)}"
        )
    task_case_pairs = {
        (int(row["task_id"]), row["case_type"]) for row in rows
    }
    expected_pairs = {
        (task_id, case_name)
        for task_id in range(36)
        for case_name in goal_validator.cases_for_skill(
            ["put_on_top", "put_inside", "push_to"][task_id // 3 % 3]
        )
    }
    if task_case_pairs != expected_pairs or len(rows) != len(task_case_pairs):
        raise ValueError("Gate-4 task/case matrix is not exact")
    manifest["validated_output_path"] = str(output)
    manifest["validated_output_sha256"] = sha256_file(output)
    return manifest


def selected_rows(layout_spec: Path, args) -> list[dict]:
    rows = [
        row
        for row in reset_validator.read_layout_spec(layout_spec)
        if row["layout_id"] == CALIBRATION_LAYOUT_ID
        and row["skill"] == "push_to"
    ]
    available = {row["task_id"] for row in rows}
    requested = (
        set(FULL_PUSH_TASK_IDS)
        if args.mode == "full"
        else set(args.task_ids)
    )
    invalid = requested - available
    if invalid:
        raise ValueError(
            "Gate-5A accepts only push task IDs on layout 1; invalid: "
            f"{sorted(invalid)}"
        )
    rows = sorted(
        [row for row in rows if row["task_id"] in requested],
        key=lambda row: row["task_id"],
    )
    if args.mode == "full" and tuple(
        row["task_id"] for row in rows
    ) != FULL_PUSH_TASK_IDS:
        raise ValueError("Full mode must contain exactly 12 push tasks")
    for row in rows:
        bddl_path = Path(row["bddl_path"])
        if not bddl_path.is_file():
            raise FileNotFoundError(f"Missing BDDL file: {bddl_path}")
        if sha256_file(bddl_path) != row["bddl_sha256"]:
            raise ValueError(
                f"BDDL hash changed for task {row['task_id']} layout 1"
            )
    return rows


def bind_validated_gate3_seeds(
    rows: list[dict], reset_prerequisite: dict
):
    reset_output = Path(reset_prerequisite["validated_output_path"])
    with reset_output.open(newline="", encoding="utf-8") as file:
        audit_rows = list(csv.DictReader(file))
    by_key = {}
    for audit in audit_rows:
        key = (
            int(audit["task_id"]),
            int(audit["layout_id"]),
            int(audit["reset_index"]),
        )
        if key in by_key:
            raise ValueError(f"Duplicate Gate-3 reset row: {key}")
        by_key[key] = audit
    for row in rows:
        key = (
            int(row["task_id"]),
            CALIBRATION_LAYOUT_ID,
            GATE3_CALIBRATION_RESET_INDEX,
        )
        if key not in by_key:
            raise ValueError(
                f"Gate-3 calibration reset evidence is missing: {key}"
            )
        audit = by_key[key]
        expected_seed = reset_validator.seed_for(
            row,
            GATE3_CALIBRATION_RESET_INDEX,
            reset_validator.FULL_SEED_BASE,
        )
        checks = {
            "passed": str(audit["passed"]).strip().lower() == "true",
            "error_empty": audit.get("error", "") == "",
            "determinism_checked": str(
                audit["determinism_checked"]
            ).strip().lower()
            == "true",
            "deterministic": str(audit["deterministic"])
            .strip()
            .lower()
            == "true",
            "settled_source_official": str(
                audit["settled_source_official_predicates_ok"]
            ).strip().lower()
            == "true",
            "settled_source_xy": str(
                audit["settled_source_xy_bounds_ok"]
            ).strip().lower()
            == "true",
            "final_source_official": str(
                audit["final_window_source_official_predicates_ok"]
            ).strip().lower()
            == "true",
            "final_source_xy": str(
                audit["final_window_source_xy_bounds_ok"]
            ).strip().lower()
            == "true",
            "seed": int(audit["seed"]) == expected_seed,
            "bddl_sha256": audit["bddl_sha256"] == row["bddl_sha256"],
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(
                "Gate-3 calibration reset evidence is invalid for "
                f"{key}: {failed}"
            )
        row["calibration_seed"] = expected_seed
        row["gate3_reset_index"] = GATE3_CALIBRATION_RESET_INDEX


def candidates_for(row: dict) -> list[dict]:
    # Task 34 has a narrow, collision-free left-finger entry.  Keep this
    # task-specific controller candidate isolated from the generic height
    # sweep so already-passed tasks retain their frozen candidate protocol.
    if int(row["task_id"]) == 34:
        return [
            {
                "candidate_index": 0,
                "pusher_finger": "left",
                "pad_height_offset_m": 0.010,
                "desired_yaw_degrees": 32.0,
                "retreat_distance_m": 0.005,
                "push_iterations": 110,
                "push_inner_steps": 4,
                "candidate_order_key": "task34_left_yaw32_retreat005",
            }
        ]
    base = BASE_PAD_HEIGHT_OFFSET_M[row["object"]]
    candidates = []
    index = 0
    for delta in HEIGHT_DELTAS_M:
        for finger in PUSH_FINGERS:
            candidates.append(
                {
                    "candidate_index": index,
                    "pusher_finger": finger,
                    "pad_height_offset_m": base + delta,
                    "candidate_order_key": (
                        f"height_delta={delta:+.3f};finger={finger}"
                    ),
                }
            )
            index += 1
    return candidates


def target_instance(row: dict) -> str:
    return f"{row['object']}_1"


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


def collision_geom_names(env, object_name: str) -> list[str]:
    inner = env.env
    model = inner.sim.model
    root = int(inner.obj_body_id[object_name])
    bodies = descendant_body_ids(model, root)
    names = []
    for geom_id, body_id in enumerate(
        np.asarray(model.geom_bodyid, dtype=np.int64)
    ):
        if int(body_id) not in bodies:
            continue
        if (
            int(model.geom_contype[geom_id]) == 0
            and int(model.geom_conaffinity[geom_id]) == 0
        ):
            continue
        name = model.geom_id2name(geom_id)
        if name:
            names.append(name)
    if not names:
        raise RuntimeError(
            f"No collision geometry found for {object_name}"
        )
    return sorted(set(names))


def contact_geometry(env, target: str) -> dict:
    inner = env.env
    important = inner.robots[0].gripper.important_geoms
    left = list(important.get("left_fingerpad", []))
    right = list(important.get("right_fingerpad", []))
    if not left or not right:
        raise RuntimeError("Panda fingerpad geometry is unavailable")
    target_geoms = collision_geom_names(env, target)
    robot = inner.robots[0]
    robot_geoms = set(
        getattr(robot.robot_model, "contact_geoms", [])
    )
    robot_geoms.update(
        getattr(robot.gripper, "contact_geoms", [])
    )
    robot_geoms.update(
        name
        for names in important.values()
        for name in names
    )
    distractors = {
        name: collision_geom_names(env, name)
        for name in reset_validator.MANIPULABLES
        if name != target
    }
    for name in sorted(robot_geoms) + target_geoms + [
        geom
        for names in distractors.values()
        for geom in names
    ]:
        if int(inner.sim.model.geom_name2id(name)) < 0:
            raise RuntimeError(f"Unresolved collision geometry: {name}")
    return {
        "left": left,
        "right": right,
        "target": target_geoms,
        "distractors": distractors,
        "robot": sorted(robot_geoms),
        "nonselected_robot": {
            side: sorted(robot_geoms - set(pads))
            for side, pads in {"left": left, "right": right}.items()
        },
    }


def mean_geom_position(inner, names: list[str]) -> np.ndarray:
    ids = [int(inner.sim.model.geom_name2id(name)) for name in names]
    return np.asarray(inner.sim.data.geom_xpos[ids], dtype=np.float64).mean(
        axis=0
    )


def finger_positions(inner, geometry: dict) -> dict[str, np.ndarray]:
    return {
        side: mean_geom_position(inner, geometry[side])
        for side in ("left", "right")
    }


def active_unexpected_contacts(
    env, target: str, geometry: dict, selected_side: str
) -> tuple[bool, bool, bool]:
    inner = env.env
    object_contact = bool(reset_validator.contact_pairs(inner, None))
    distractors = [
        name for name in reset_validator.MANIPULABLES if name != target
    ]
    robot_distractor = any(
        inner.check_contact(
            geometry["robot"], geometry["distractors"][name]
        )
        for name in distractors
    )
    nonselected_target = bool(
        inner.check_contact(
            geometry["nonselected_robot"][selected_side],
            geometry["target"],
        )
    )
    return object_contact, bool(robot_distractor), nonselected_target


def robot_target_contact_pairs(
    inner, geometry: dict
) -> tuple[tuple[str, str], ...]:
    """Return the exact robot / target geometry pairs in the current state."""
    model = inner.sim.model
    target_ids = {
        int(model.geom_name2id(name)) for name in geometry["target"]
    }
    robot_ids = {
        int(model.geom_name2id(name)) for name in geometry["robot"]
    }
    pairs = set()
    for index in range(inner.sim.data.ncon):
        contact = inner.sim.data.contact[index]
        first = int(contact.geom1)
        second = int(contact.geom2)
        if first in target_ids and second in robot_ids:
            pairs.add((model.geom_id2name(second), model.geom_id2name(first)))
        elif second in target_ids and first in robot_ids:
            pairs.add((model.geom_id2name(first), model.geom_id2name(second)))
    return tuple(sorted(pairs))


def target_xyz(obs, target: str) -> np.ndarray:
    return np.asarray(obs[f"{target}_pos"], dtype=np.float64).reshape(3)


def eef_xyz(obs) -> np.ndarray:
    return np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(3)


def eef_rotation(obs) -> np.ndarray:
    quat = np.asarray(
        obs["robot0_eef_quat"], dtype=np.float64
    ).reshape(4)
    return transform.quat2mat(quat)


def rotation_action(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    error_matrix = desired @ current.T
    axis_angle = transform.quat2axisangle(
        transform.mat2quat(error_matrix)
    )
    return np.clip(axis_angle / 0.5, -0.4, 0.4)


def osc_action(
    obs,
    desired_eef: np.ndarray | None,
    desired_rotation: np.ndarray,
    position_cap: float,
    gripper_action: float = -1.0,
) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    if desired_eef is not None:
        error = np.asarray(desired_eef) - eef_xyz(obs)
        action[:3] = np.clip(
            error / 0.05, -position_cap, position_cap
        )
    action[3:6] = rotation_action(
        desired_rotation,
        eef_rotation(obs),
    )
    # Gate-5A never closes the gripper during active control. Passive
    # settling and terminal hold use strict all-zero environment actions.
    action[6] = gripper_action
    return action


def push_contact_record(env, geometry: dict) -> tuple[bool, bool, bool]:
    inner = env.env
    left = bool(
        inner.check_contact(geometry["left"], geometry["target"])
    )
    right = bool(
        inner.check_contact(geometry["right"], geometry["target"])
    )
    grasp = bool(
        inner._check_grasp(
            inner.robots[0].gripper,
            geometry["target"],
        )
    )
    if grasp != (left and right):
        raise RuntimeError(
            "Robosuite grasp proxy disagrees with bilateral fingerpads"
        )
    return left, right, grasp


def table_support(inner, target: str) -> bool:
    return bool(
        inner.check_contact(
            inner.get_object(target),
            inner.get_object(reset_validator.WORKSPACE_FIXTURE),
        )
    )


def target_region_status(env, obs, row: dict) -> tuple[bool, bool]:
    official = bool(env.check_success())
    xy = target_xyz(obs, target_instance(row))[:2]
    bounds = reset_validator.TARGET_REGION_RANGES[
        row["spatial_region"]
    ]
    xy_ok = reset_validator.xy_in_bounds(
        np.array([xy[0], xy[1], 0.0]), bounds
    )
    return official, xy_ok


def max_state_difference(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return math.inf
    if not left.size:
        return 0.0
    return float(np.max(np.abs(left - right)))


class AttemptRecorder:
    def __init__(
        self,
        env,
        row: dict,
        target: str,
        geometry: dict,
        selected_side: str,
        stabilized_z: float,
        active_action_limit: int | None = MAX_ACTIVE_ACTIONS,
        allow_non_grasping_gripper_motion: bool = False,
    ):
        self.env = env
        self.row = row
        self.target = target
        self.geometry = geometry
        self.selected_side = selected_side
        self.stabilized_z = stabilized_z
        self.active_action_limit = active_action_limit
        self.allow_non_grasping_gripper_motion = (
            allow_non_grasping_gripper_motion
        )
        self.actions: list[np.ndarray] = []
        self.states: list[np.ndarray] = [
            np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        ]
        self.phase: list[str] = []
        self.target_positions: list[np.ndarray] = []
        self.eef_positions: list[np.ndarray] = []
        self.left_contact: list[bool] = []
        self.right_contact: list[bool] = []
        self.grasp_proxy: list[bool] = []
        self.table_support: list[bool] = []
        self.unexpected_object_contact: list[bool] = []
        self.robot_distractor_contact: list[bool] = []
        self.nonselected_robot_target_contact: list[bool] = []
        self.robot_target_contact_pairs: list[str] = []
        self.relation: list[bool] = []
        self.xy_in_target: list[bool] = []
        self.obs = None

    def step(self, obs, action: np.ndarray, phase: str):
        action = np.asarray(action)
        if action.shape != (7,):
            raise ValueError(f"Invalid action shape: {action.shape}")
        if action.dtype != np.float32:
            raise ValueError(f"Action dtype is not float32: {action.dtype}")
        if not np.isfinite(action).all():
            raise ValueError("Action contains non-finite values")
        if np.any(action < -1.0) or np.any(action > 1.0):
            raise ValueError("Action is outside [-1, 1]")
        if phase == "terminal_hold":
            if not np.array_equal(
                action, np.zeros(7, dtype=np.float32)
            ):
                raise ValueError(
                    "Terminal hold action must be exactly all-zero"
                )
        elif (
            not self.allow_non_grasping_gripper_motion
            and float(action[6]) != -1.0
        ):
            raise ValueError("Active action must keep gripper at -1")
        if (
            self.active_action_limit is not None
            and len(self.actions) >= self.active_action_limit
        ):
            raise RuntimeError("Active-action safety horizon exceeded")
        next_obs, _, done, _ = self.env.step(action)
        left, right, grasp = push_contact_record(
            self.env, self.geometry
        )
        support = table_support(self.env.env, self.target)
        (
            object_contact,
            robot_distractor,
            nonselected_target,
        ) = active_unexpected_contacts(
            self.env,
            self.target,
            self.geometry,
            self.selected_side,
        )
        relation, xy_ok = target_region_status(
            self.env, next_obs, self.row
        )
        if bool(done) != bool(relation):
            raise RuntimeError(
                "Environment done flag disagrees with goal predicate: "
                f"done={done}, relation={relation}, phase={phase}"
            )
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.states.append(
            np.asarray(
                self.env.get_sim_state(), dtype=np.float64
            ).copy()
        )
        self.phase.append(phase)
        self.target_positions.append(target_xyz(next_obs, self.target))
        self.eef_positions.append(eef_xyz(next_obs))
        self.left_contact.append(left)
        self.right_contact.append(right)
        self.grasp_proxy.append(grasp)
        self.table_support.append(support)
        self.unexpected_object_contact.append(object_contact)
        self.robot_distractor_contact.append(robot_distractor)
        self.nonselected_robot_target_contact.append(
            nonselected_target
        )
        self.robot_target_contact_pairs.append(
            json_compact(robot_target_contact_pairs(self.env.env, self.geometry))
        )
        self.relation.append(relation)
        self.xy_in_target.append(xy_ok)
        self.obs = next_obs
        return next_obs

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "actions": np.asarray(self.actions, dtype=np.float32),
            "states": np.asarray(self.states, dtype=np.float64),
            "phase": np.asarray(self.phase, dtype="U32"),
            "target_xyz": np.asarray(
                self.target_positions, dtype=np.float64
            ).reshape(-1, 3),
            "eef_xyz": np.asarray(
                self.eef_positions, dtype=np.float64
            ).reshape(-1, 3),
            "left_contact": np.asarray(
                self.left_contact, dtype=np.bool_
            ),
            "right_contact": np.asarray(
                self.right_contact, dtype=np.bool_
            ),
            "grasp_proxy": np.asarray(
                self.grasp_proxy, dtype=np.bool_
            ),
            "table_support": np.asarray(
                self.table_support, dtype=np.bool_
            ),
            "unexpected_object_contact": np.asarray(
                self.unexpected_object_contact, dtype=np.bool_
            ),
            "robot_distractor_contact": np.asarray(
                self.robot_distractor_contact, dtype=np.bool_
            ),
            "nonselected_robot_target_contact": np.asarray(
                self.nonselected_robot_target_contact,
                dtype=np.bool_,
            ),
            "robot_target_contact_pairs": np.asarray(
                self.robot_target_contact_pairs, dtype="U1024"
            ),
            "relation": np.asarray(self.relation, dtype=np.bool_),
            "xy_in_target": np.asarray(
                self.xy_in_target, dtype=np.bool_
            ),
        }


def settle_environment(env, row: dict, seed: int) -> dict:
    obs, attempts, camera_record = reset_validator.safe_reset(
        env, seed, MAX_RESET_ATTEMPTS
    )
    camera_config.validate_runtime_camera_record(camera_record)
    reset_validator.validate_parsed_semantics(env.env, row)
    target = target_instance(row)
    initial_state = np.asarray(
        env.get_sim_state(), dtype=np.float64
    ).copy()
    initial_poses = reset_validator.capture_poses(obs, None)
    initial_source = reset_validator.source_region_details(
        env.env, initial_poses, row["mapping"]
    )
    _, initial_source_xy, _ = reset_validator.region_detail_checks(
        initial_source
    )
    if bool(env.check_success()):
        raise RuntimeError("Push goal is true at raw reset")
    if not initial_source_xy:
        raise RuntimeError("Raw-reset source XY placement failed Gate 3")
    if not reset_validator.images_valid(obs):
        raise RuntimeError("Raw-reset observations failed image checks")
    if not reset_validator.poses_are_finite(initial_poses):
        raise RuntimeError("Raw-reset poses are non-finite")
    desired_rotation = eef_rotation(obs)
    z_values = [float(target_xyz(obs, target)[2])]
    settle_states = [initial_state.copy()]
    settle_actions = []
    step_motion = []
    step_orientation = []
    target_support_trace = []
    source_official_trace = []
    source_xy_trace = []
    previous_poses = initial_poses
    zero_action = np.zeros(int(env.env.action_dim), dtype=np.float32)
    if zero_action.shape != (7,):
        raise RuntimeError(
            f"Unexpected environment action dimension: {zero_action.shape}"
        )
    for step_index in range(1, SETTLE_STEPS + 1):
        action = zero_action.copy()
        obs, _, done, _ = env.step(action)
        if done:
            raise RuntimeError("Environment ended during passive settle")
        settle_actions.append(action)
        settle_states.append(
            np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        )
        z_values.append(float(target_xyz(obs, target)[2]))
        poses = reset_validator.capture_poses(obs, None)
        if not reset_validator.poses_are_finite(poses):
            raise RuntimeError(
                f"Non-finite pose during settle step {step_index}"
            )
        source = reset_validator.source_region_details(
            env.env, poses, row["mapping"]
        )
        source_official, source_xy, _ = (
            reset_validator.region_detail_checks(source)
        )
        source_official_trace.append(source_official)
        source_xy_trace.append(source_xy)
        contacts = reset_validator.contact_pairs(env.env, None)
        if contacts:
            raise RuntimeError(
                "Unexpected object-object contact during settle: "
                f"{contacts}"
            )
        motion, orientation = reset_validator.max_pose_motion(
            previous_poses, poses
        )
        step_motion.append(motion)
        step_orientation.append(orientation)
        target_support_trace.append(table_support(env.env, target))
        previous_poses = poses
        if bool(env.check_success()):
            raise RuntimeError(
                f"Push goal became true during settle step {step_index}"
            )
    final_window = min(FINAL_WINDOW_STEPS, len(step_motion))
    if (
        max(step_motion[-final_window:])
        > reset_validator.FULL_MAX_FINAL_WINDOW_MOTION_M
        or max(step_orientation[-final_window:])
        > reset_validator.FULL_MAX_FINAL_WINDOW_ORIENTATION_DEG
    ):
        raise RuntimeError("Passive-settle final window is unstable")
    settled_poses = reset_validator.capture_poses(obs, None)
    if not reset_validator.images_valid(obs):
        raise RuntimeError("Settled observations failed image checks")
    settled_source = reset_validator.source_region_details(
        env.env, settled_poses, row["mapping"]
    )
    settled_official, settled_xy, _ = (
        reset_validator.region_detail_checks(settled_source)
    )
    if not (settled_official and settled_xy):
        raise RuntimeError("Settled source placement failed Gate 3")
    if reset_validator.contact_pairs(env.env, None):
        raise RuntimeError("Settled state has object-object contact")
    support_array = np.asarray(
        target_support_trace, dtype=np.bool_
    )
    source_official_array = np.asarray(
        source_official_trace, dtype=np.bool_
    )
    source_xy_array = np.asarray(source_xy_trace, dtype=np.bool_)
    if (
        support_array.size < FINAL_WINDOW_STEPS
        or not support_array[-FINAL_WINDOW_STEPS:].all()
    ):
        raise RuntimeError(
            "Target lacks direct table support in the passive-settle "
            "final window"
        )
    if (
        source_official_array.size < FINAL_WINDOW_STEPS
        or not source_official_array[-FINAL_WINDOW_STEPS:].all()
    ):
        raise RuntimeError(
            "Source official predicates fail in the passive-settle "
            "final window"
        )
    if (
        source_xy_array.size < FINAL_WINDOW_STEPS
        or not source_xy_array[-FINAL_WINDOW_STEPS:].all()
    ):
        raise RuntimeError(
            "Source XY bounds fail in the passive-settle final window"
        )
    horizontal_drift = reset_validator.horizontal_pose_drift(
        initial_poses,
        settled_poses,
        reset_validator.MANIPULABLES,
    )
    _, orientation_drift = reset_validator.pose_drift(
        initial_poses,
        settled_poses,
        reset_validator.MANIPULABLES,
    )
    max_horizontal_drift = max(horizontal_drift.values())
    max_orientation_drift = max(orientation_drift.values())
    if max_horizontal_drift > (
        reset_validator.FULL_MAX_MANIPULABLE_HORIZONTAL_DRIFT_M
    ):
        raise RuntimeError(
            "Passive-settle horizontal drift exceeds Gate 3"
        )
    if max_orientation_drift > (
        reset_validator.FULL_MAX_ORIENTATION_DRIFT_DEG
    ):
        raise RuntimeError(
            "Passive-settle orientation drift exceeds Gate 3"
        )
    settled_state = np.asarray(
        env.get_sim_state(), dtype=np.float64
    ).copy()
    z_array = np.asarray(z_values, dtype=np.float64)
    frozen_z = z_array[-FINAL_WINDOW_STEPS:]
    stabilized_z = float(np.mean(frozen_z))
    positive_noise = max(
        0.0,
        float(np.max(frozen_z) - stabilized_z),
    )
    return {
        "obs": obs,
        "attempts": attempts,
        "camera_record": camera_record,
        "initial_state": initial_state,
        "settled_state": settled_state,
        "settle_actions": np.asarray(settle_actions, dtype=np.float32),
        "settle_states": np.asarray(settle_states, dtype=np.float64),
        "settle_target_table_support": support_array,
        "settle_source_official": source_official_array,
        "settle_source_xy": source_xy_array,
        "desired_rotation": desired_rotation,
        "z_values": z_array,
        "max_positive_z_noise": positive_noise,
        "stabilized_z": stabilized_z,
        "max_horizontal_drift": max_horizontal_drift,
        "max_orientation_drift": max_orientation_drift,
    }


def desired_eef_for_pad(
    env,
    obs,
    geometry: dict,
    side: str,
    desired_pad: np.ndarray,
    desired_rotation: np.ndarray | None = None,
) -> np.ndarray:
    current_pad = finger_positions(env.env, geometry)[side]
    if desired_rotation is not None:
        current_eef = eef_xyz(obs)
        current_rotation = eef_rotation(obs)
        local_pad_offset = current_rotation.T @ (current_pad - current_eef)
        return np.asarray(desired_pad) - (
            np.asarray(desired_rotation) @ local_pad_offset
        )
    return eef_xyz(obs) + (np.asarray(desired_pad) - current_pad)


ACTIVE_GRIPPER_ACTION = -1.0

def move_pad_to(
    recorder: AttemptRecorder,
    obs,
    geometry: dict,
    side: str,
    desired_pad: np.ndarray,
    desired_rotation: np.ndarray,
    phase: str,
    max_steps: int,
    position_cap: float,
    gripper_action: float | None = None,
) -> tuple[object, bool, dict[str, float | int | bool]]:
    if gripper_action is None:
        gripper_action = ACTIVE_GRIPPER_ACTION
    stable = 0
    longest_stable = 0
    position_errors = []
    rotation_errors = []
    for _ in range(max_steps):
        current_rotation = desired_rotation
        desired_eef = desired_eef_for_pad(
            recorder.env,
            obs,
            geometry,
            side,
            desired_pad,
            current_rotation if ROTATION_AWARE_PAD_TARGET else None,
        )
        action = osc_action(
            obs, desired_eef, desired_rotation, position_cap, gripper_action
        )
        obs = recorder.step(obs, action, phase)
        pad = finger_positions(recorder.env.env, geometry)[side]
        position_error = float(np.linalg.norm(pad - desired_pad))
        rotation_error = float(
            np.linalg.norm(
                    transform.quat2axisangle(
                    transform.mat2quat(
                        desired_rotation @ eef_rotation(obs).T
                    )
                )
            )
        )
        position_errors.append(position_error)
        rotation_errors.append(rotation_error)
        if (
            position_error <= POSITION_TOLERANCE_M
            and rotation_error <= ORIENTATION_TOLERANCE_RAD
        ):
            stable += 1
        else:
            stable = 0
        longest_stable = max(longest_stable, stable)
        if stable >= STABLE_POSE_STEPS:
            return obs, True, {
                "min_position_error_m": min(position_errors),
                "final_position_error_m": position_errors[-1],
                "min_rotation_error_rad": min(rotation_errors),
                "final_rotation_error_rad": rotation_errors[-1],
                "longest_valid_run": longest_stable,
                "reached": True,
            }
    return obs, False, {
        "min_position_error_m": min(position_errors),
        "final_position_error_m": position_errors[-1],
        "min_rotation_error_rad": min(rotation_errors),
        "final_rotation_error_rad": rotation_errors[-1],
        "longest_valid_run": longest_stable,
        "reached": False,
    }


def run_original_attempt(
    row: dict,
    candidate: dict,
    seed: int,
) -> dict:
    global ROTATION_AWARE_PAD_TARGET
    ROTATION_AWARE_PAD_TARGET = bool(
        candidate.get("rotation_aware_pad_target", False)
    )
    global ACTIVE_GRIPPER_ACTION
    ACTIVE_GRIPPER_ACTION = float(candidate.get("active_gripper_action", -1.0))
    target = target_instance(row)
    env = reset_validator.make_environment(Path(row["bddl_path"]))
    recorder = None
    settled = None
    try:
        settled = settle_environment(env, row, seed)
        reset_rotation = settled["desired_rotation"].copy()
        desired_yaw = float(candidate.get("desired_yaw_degrees", 0.0))
        if desired_yaw:
            yaw = math.radians(desired_yaw)
            yaw_rotation = np.array(
                [
                    [math.cos(yaw), -math.sin(yaw), 0.0],
                    [math.sin(yaw), math.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            settled["desired_rotation"] = (
                yaw_rotation @ settled["desired_rotation"]
            )
        # Diagnostic candidates may add a world-frame pitch after yaw.
        desired_pitch = float(candidate.get("desired_pitch_degrees", 0.0))
        if desired_pitch:
            pitch = math.radians(desired_pitch)
            pitch_rotation = np.array(
                [[math.cos(pitch), 0.0, math.sin(pitch)],
                 [0.0, 1.0, 0.0],
                 [-math.sin(pitch), 0.0, math.cos(pitch)]],
                dtype=np.float64,
            )
            settled["desired_rotation"] = (
                pitch_rotation @ settled["desired_rotation"]
            )
        obs = settled["obs"]
        geometry = contact_geometry(env, target)
        pads = finger_positions(env.env, geometry)
        selected = candidate["pusher_finger"]
        opposite = "right" if selected == "left" else "left"
        selected_offset = pads[selected] - eef_xyz(obs)
        opposite_offset = pads[opposite] - eef_xyz(obs)
        start_target = target_xyz(obs, target)
        bounds = reset_validator.TARGET_REGION_RANGES[
            row["spatial_region"]
        ]
        destination = np.array(
            [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2],
            dtype=np.float64,
        )
        controller_route = candidate.get("controller_route_xy")
        route_points = (
            np.asarray(controller_route, dtype=np.float64).reshape(-1, 2)
            if controller_route is not None
            else None
        )
        controller_destination = np.asarray(
            route_points[0]
            if route_points is not None
            else candidate.get("controller_destination_xy", destination),
            dtype=np.float64,
        ).reshape(2)
        # Keep the reachable approach geometry independent from the final
        # push endpoint.  This lets a candidate tune the force direction
        # without making the pre-contact transit target unreachable.
        push_destination = np.asarray(
            candidate.get("push_destination_xy", controller_destination),
            dtype=np.float64,
        ).reshape(2)
        approach_destination = np.asarray(
            candidate.get("approach_destination_xy", controller_destination),
            dtype=np.float64,
        ).reshape(2)
        controller_destination_tolerance = float(
            candidate.get("controller_destination_tolerance_m", 0.015)
        )
        displacement = push_destination - start_target[:2]
        distance = float(np.linalg.norm(displacement))
        if distance <= 0.1:
            raise RuntimeError("Push path is unexpectedly short")
        direction = displacement / distance
        push_direction_yaw = float(
            candidate.get("push_direction_yaw_degrees", 0.0)
        )
        if push_direction_yaw:
            yaw = math.radians(push_direction_yaw)
            push_direction_rotation = np.array(
                [
                    [math.cos(yaw), -math.sin(yaw)],
                    [math.sin(yaw), math.cos(yaw)],
                ],
                dtype=np.float64,
            )
            direction = push_direction_rotation @ direction
        approach_displacement = approach_destination - start_target[:2]
        approach_distance = float(np.linalg.norm(approach_displacement))
        if approach_distance <= 0.1:
            raise RuntimeError("Approach path is unexpectedly short")
        approach_direction = approach_displacement / approach_distance
        active_action_limit = candidate.get(
            "active_action_limit", MAX_ACTIVE_ACTIONS
        )
        if active_action_limit is not None:
            active_action_limit = int(active_action_limit)
            if active_action_limit <= 0:
                raise ValueError("active_action_limit must be positive or null")
        allow_auxiliary_target_contact = bool(
            candidate.get("allow_auxiliary_target_contact", False)
        )
        recorder = AttemptRecorder(
            env,
            row,
            target,
            geometry,
            selected,
            settled["stabilized_z"],
            active_action_limit,
        )
        pad_z = (
            start_target[2] + candidate["pad_height_offset_m"]
        )
        approach_rotation = settled["desired_rotation"]
        approach_yaw = float(
            candidate.get("approach_yaw_degrees", 0.0)
        )
        if approach_yaw:
            yaw = math.radians(approach_yaw)
            approach_rotation = np.array(
                [
                    [math.cos(yaw), -math.sin(yaw), 0.0],
                    [math.sin(yaw), math.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ) @ settled["desired_rotation"]
        approach_standoff = float(
            candidate.get(
                "approach_standoff_m",
                reset_validator.SOURCE_SAMPLER_RADII_M[row["object"]]
                + PRECONTACT_PAD_CLEARANCE_M,
            )
        )
        behind_pad = np.array(
            [
                start_target[0]
                - approach_direction[0] * approach_standoff,
                start_target[1]
                - approach_direction[1] * approach_standoff,
                pad_z,
            ]
        )
        if candidate.get("behind_pad_offset_xy") is not None:
            behind_pad[:2] = start_target[:2] + np.asarray(
                candidate["behind_pad_offset_xy"], dtype=np.float64
            ).reshape(2)
        approach_lateral_offset = candidate.get(
            "approach_lateral_offset_xy"
        )
        if approach_lateral_offset is not None:
            behind_pad[:2] += np.asarray(
                approach_lateral_offset, dtype=np.float64
            ).reshape(2)
        # Approach in two stages. First move to a reachable point above
        # the live target center; only then move out and down to the
        # compact pre-contact point. This avoids combining maximum
        # horizontal extension with the old fixed 1.10 m high pose.
        transit_pad = np.array(
            [
                start_target[0],
                start_target[1],
                pad_z + TRANSIT_PAD_CLEARANCE_M,
            ]
        )
        obs, high_ok, transit_diagnostics = move_pad_to(
            recorder,
            obs,
            geometry,
            selected,
            transit_pad,
            (
                reset_rotation
                if candidate.get("rotate_high_before_descend", False)
                else settled["desired_rotation"]
            ),
            "transit_high",
            MAX_TRANSIT_STEPS,
            0.6,
        )
        # Some corridor entries are safe only when the wrist rotates above
        # the manipulables.  Rotating while descending sweeps the hand or
        # opposite finger through a neighbour, even though both endpoint
        # poses are collision-free.  This opt-in primitive preserves every
        # Gate-5 contact and action check; it merely separates the two
        # Cartesian motions.
        if candidate.get("rotate_high_before_descend", False):
            obs, rotate_high_ok, _ = move_pad_to(
                recorder,
                obs,
                geometry,
                selected,
                transit_pad,
                settled["desired_rotation"],
                "rotate_high",
                60,
                0.30,
            )
            high_ok = high_ok and rotate_high_ok
        precontact_waypoints = candidate.get("precontact_waypoints_xy", [])
        if precontact_waypoints:
            for waypoint_xy in precontact_waypoints:
                precontact_pad = np.array(
                    [
                        float(waypoint_xy[0]),
                        float(waypoint_xy[1]),
                        pad_z + TRANSIT_PAD_CLEARANCE_M,
                    ],
                    dtype=np.float64,
                )
                obs, precontact_ok, _ = move_pad_to(
                    recorder,
                    obs,
                    geometry,
                    selected,
                    precontact_pad,
                    settled["desired_rotation"],
                    "approach_precontact_high",
                    MAX_TRANSIT_STEPS,
                    0.6,
                )
                high_ok = high_ok and precontact_ok
        if candidate.get("side_step_approach", False):
            side_high_pad = behind_pad.copy()
            side_high_pad[2] = pad_z + TRANSIT_PAD_CLEARANCE_M
            obs, side_high_ok, side_high_diagnostics = move_pad_to(
                recorder,
                obs,
                geometry,
                selected,
                side_high_pad,
                settled["desired_rotation"],
                "approach_side_high",
                MAX_TRANSIT_STEPS,
                0.6,
            )
            high_ok = high_ok and side_high_ok
        obs, behind_ok, behind_diagnostics = move_pad_to(
            recorder,
            obs,
            geometry,
            selected,
            behind_pad,
            approach_rotation,
            "descend_behind",
            MAX_TRANSIT_STEPS,
            0.25,
        )
        contact_step = None
        contact_pad = behind_pad.copy()
        contact_search_rotation = approach_rotation
        search_steps = int(
            math.ceil(
                MAX_CONTACT_SEARCH_M / CONTACT_SEARCH_INCREMENT_M
            )
        )
        if high_ok and (
            behind_ok or candidate.get("allow_approximate_behind", False)
        ):
            contact_search_inner_steps = int(
                candidate.get("contact_search_inner_steps", 5)
            )
            if contact_search_inner_steps <= 0 or contact_search_inner_steps > 5:
                raise ValueError(
                    f"Invalid contact search inner steps: {contact_search_inner_steps}"
                )
            for search_index in range(search_steps):
                contact_pad[:2] += approach_direction * float(
                    candidate.get("contact_search_increment_m",
                                  CONTACT_SEARCH_INCREMENT_M)
                )
                search_yaw = float(
                    candidate.get("contact_search_yaw_degrees", 0.0)
                )
                if search_yaw:
                    fraction = min(
                        1.0, float(search_index + 1) / float(search_steps)
                    )
                    yaw = math.radians(search_yaw * fraction)
                    contact_search_rotation = np.array(
                        [
                            [math.cos(yaw), -math.sin(yaw), 0.0],
                            [math.sin(yaw), math.cos(yaw), 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        dtype=np.float64,
                    ) @ settled["desired_rotation"]
                else:
                    contact_search_rotation = approach_rotation
                obs, _, _ = move_pad_to(
                    recorder,
                    obs,
                    geometry,
                    selected,
                    contact_pad,
                    contact_search_rotation,
                    "contact_search",
                    contact_search_inner_steps,
                    0.12,
                )
                selected_contact = (
                    recorder.left_contact[-1]
                    if selected == "left"
                    else recorder.right_contact[-1]
                )
                opposite_contact = (
                    recorder.right_contact[-1]
                    if selected == "left"
                    else recorder.left_contact[-1]
                )
                contact_usable = selected_contact or (
                    allow_auxiliary_target_contact
                    and recorder.nonselected_robot_target_contact[-1]
                )
                if contact_usable and not opposite_contact:
                    recovery_offsets = candidate.get(
                        "contact_recovery_offsets_xy", []
                    )
                    recovered = (
                        allow_auxiliary_target_contact
                        or not recorder.nonselected_robot_target_contact[-1]
                    )
                    if recovery_offsets and not recovered:
                        for recovery_offset in recovery_offsets:
                            recovery_pad = contact_pad.copy()
                            recovery_pad[:2] += np.asarray(
                                recovery_offset, dtype=np.float64
                            ).reshape(2)
                            obs, _, _ = move_pad_to(
                                recorder,
                                obs,
                                geometry,
                                selected,
                                recovery_pad,
                                contact_search_rotation,
                                "contact_recovery",
                                10,
                                0.10,
                            )
                            selected_contact = (
                                recorder.left_contact[-1]
                                if selected == "left"
                                else recorder.right_contact[-1]
                            )
                            opposite_contact = (
                                recorder.right_contact[-1]
                                if selected == "left"
                                else recorder.left_contact[-1]
                            )
                            recovered = (
                                (
                                    selected_contact
                                    or (
                                        allow_auxiliary_target_contact
                                        and recorder.nonselected_robot_target_contact[-1]
                                    )
                                )
                                and not opposite_contact
                                and (
                                    allow_auxiliary_target_contact
                                    or not recorder.nonselected_robot_target_contact[-1]
                                )
                            )
                            if recovered:
                                contact_pad = recovery_pad
                                break
                    if recovery_offsets and not recovered:
                        continue
                    contact_step = len(recorder.actions)
                    break
                if opposite_contact or recorder.grasp_proxy[-1]:
                    break

        if contact_step is not None:
            if route_points is not None:
                route_nodes = np.vstack([start_target[:2], route_points])
                push_distance = float(
                    np.linalg.norm(np.diff(route_nodes, axis=0), axis=1).sum()
                ) + PUSH_OVERSHOOT_M
            else:
                push_distance = distance + PUSH_OVERSHOOT_M
            contact_follow = bool(candidate.get("contact_follow", False))
            contact_follow_penetration = float(
                candidate.get("contact_follow_penetration_m", 0.002)
            )
            contact_offset_xy = (
                finger_positions(env.env, geometry)[selected][:2]
                - target_xyz(obs, target)[:2]
            )
            nominal_increments = min(
                int(candidate.get("max_push_steps", MAX_PUSH_STEPS)),
                int(math.ceil(push_distance / PUSH_INCREMENT_M)),
            )
            increments = int(
                candidate.get("push_iterations", nominal_increments)
            )
            if increments <= 0:
                raise ValueError(
                    f"Invalid push iteration count: {increments}"
                )
            push_inner_steps = int(candidate.get("push_inner_steps", 4))
            if push_inner_steps <= 0 or push_inner_steps > 4:
                raise ValueError(
                    f"Invalid push inner step count: {push_inner_steps}"
                )
            route_index = 0
            segment_start = start_target[:2].copy()
            segment_distance = distance
            # Routed pushes must begin along the first waypoint segment.
            if route_points is not None and len(route_points) > 0:
                first_segment = np.asarray(route_points[0], dtype=np.float64).reshape(2) - segment_start
                first_distance = float(np.linalg.norm(first_segment))
                if first_distance > 1e-9:
                    segment_distance = first_distance
                    direction = first_segment / first_distance
            push_base_rotation = (
                approach_rotation
                if approach_yaw
                else settled["desired_rotation"]
            )
            push_rotation = (
                contact_search_rotation
                if candidate.get("use_contact_search_rotation", False)
                else push_base_rotation
            )
            if "push_yaw_degrees" in candidate:
                yaw = math.radians(float(candidate["push_yaw_degrees"]))
                push_rotation = np.array(
                    [
                        [math.cos(yaw), -math.sin(yaw), 0.0],
                        [math.sin(yaw), math.cos(yaw), 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                ) @ push_rotation
            for _ in range(increments):
                if contact_follow:
                    push_direction = direction
                    dynamic_feedback_switched = False
                    feedback_start_index = int(
                        candidate.get("feedback_route_start_index", 0)
                    )
                    if candidate.get(
                        "contact_follow_feedback", False
                    ) and route_index >= feedback_start_index:
                        live_xy = target_xyz(obs, target)[:2]
                        feedback_destination_spec = candidate.get(
                            "feedback_destination_xy",
                            controller_destination,
                        )
                        if isinstance(
                            feedback_destination_spec, (list, tuple)
                        ) and feedback_destination_spec and isinstance(
                            feedback_destination_spec[0], (list, tuple)
                        ):
                            switch_x = candidate.get(
                                "feedback_switch_x"
                            )
                            if route_points is None and switch_x is not None:
                                feedback_index = int(
                                    target_xyz(obs, target)[0]
                                    <= float(switch_x)
                                )
                                dynamic_feedback_switched = bool(
                                    feedback_index
                                )
                                feedback_index = min(
                                    feedback_index,
                                    len(feedback_destination_spec) - 1,
                                )
                            else:
                                feedback_index = min(
                                    route_index,
                                    len(feedback_destination_spec) - 1,
                                )
                            feedback_destination_spec = (
                                feedback_destination_spec[feedback_index]
                            )
                        feedback_destination = np.asarray(
                            feedback_destination_spec,
                            dtype=np.float64,
                        ).reshape(2)
                        feedback_delta = feedback_destination - live_xy
                        feedback_distance = float(
                            np.linalg.norm(feedback_delta)
                        )
                        if feedback_distance > 1e-6:
                            push_direction = (
                                feedback_delta / feedback_distance
                            )
                            feedback_yaw = float(
                                candidate.get(
                                    "feedback_direction_yaw_degrees", 0.0
                                )
                            )
                        if feedback_yaw:
                                yaw = math.radians(feedback_yaw)
                                feedback_rotation = np.array(
                                    [
                                        [math.cos(yaw), -math.sin(yaw)],
                                        [math.sin(yaw), math.cos(yaw)],
                                    ],
                                    dtype=np.float64,
                                )
                                push_direction = (
                                feedback_rotation @ push_direction
                            )
                    active_penetration = contact_follow_penetration
                    if dynamic_feedback_switched:
                        active_penetration = float(
                            candidate.get(
                                "feedback_switch_penetration_m",
                                contact_follow_penetration,
                            )
                        )
                    # Keep the selected pad at a bounded compression relative
                    # to the live object pose.  The open-loop controller can
                    # overtake a stalled object, causing the palm or opposite
                    # finger to become the pusher.  This mode preserves the
                    # acquired one-finger contact while still applying forward
                    # pressure.
                    contact_pad[:2] = (
                        target_xyz(obs, target)[:2]
                        + contact_offset_xy
                        + push_direction * active_penetration
                    )
                else:
                    contact_pad[:2] += direction * PUSH_INCREMENT_M
                obs, _, _ = move_pad_to(
                    recorder,
                    obs,
                    geometry,
                    selected,
                    contact_pad,
                    push_rotation,
                    "push",
                    push_inner_steps,
                    float(candidate.get("push_position_cap", 0.14)),
                )
                relation, xy_ok = target_region_status(env, obs, row)
                if relation and xy_ok:
                    break
                if route_points is not None:
                    live_xy = target_xyz(obs, target)[:2]
                    progress = float(
                        np.dot(live_xy - segment_start, direction)
                    )
                    trigger_points = candidate.get(
                        "route_transition_trigger_xy"
                    )
                    trigger_reached = False
                    if trigger_points is not None:
                        trigger_index = min(
                            route_index, len(trigger_points) - 1
                        )
                        trigger_xy = np.asarray(
                            trigger_points[trigger_index],
                            dtype=np.float64,
                        ).reshape(2)
                        trigger_tolerance = float(
                            candidate.get(
                                "route_transition_trigger_tolerance_m",
                                0.025,
                            )
                        )
                        trigger_reached = bool(
                            np.linalg.norm(live_xy - trigger_xy)
                            <= trigger_tolerance
                        )
                    if (
                        progress
                        >= segment_distance
                        - controller_destination_tolerance
                        or trigger_reached
                    ):
                        if candidate.get("debug_route_transition", False):
                            print(
                                "ROUTE_TRANSITION",
                                route_index,
                                live_xy.tolist(),
                                progress,
                                segment_distance,
                                trigger_reached,
                                flush=True,
                            )
                        if route_index + 1 >= len(route_points):
                            break
                        route_index += 1
                        segment_start = live_xy.copy()
                        controller_destination = route_points[route_index]
                        segment_displacement = (
                            controller_destination - segment_start
                        )
                        segment_distance = float(
                            np.linalg.norm(segment_displacement)
                        )
                        direction = (
                            segment_displacement / segment_distance
                        )
                        if candidate.get(
                            "reposition_on_route_transition", False
                        ):
                            live_target = target_xyz(obs, target)
                            current_pad = finger_positions(
                                env.env, geometry
                            )[selected]
                            detach_offset = candidate.get(
                                "route_detach_offset_xy"
                            )
                            detach_offsets = candidate.get(
                                "route_detach_offsets_xy"
                            )
                            if detach_offsets is not None:
                                offset_index = min(
                                    route_index - 1,
                                    len(detach_offsets) - 1,
                                )
                                detach_offset = detach_offsets[offset_index]
                            if detach_offset is not None:
                                detach_pad = current_pad.copy()
                                detach_pad[:2] += np.asarray(
                                    detach_offset, dtype=np.float64
                                ).reshape(2)
                                obs, detach_ok, _ = move_pad_to(
                                    recorder,
                                    obs,
                                    geometry,
                                    selected,
                                    detach_pad,
                                    push_rotation,
                                    "route_detach_lateral",
                                    int(
                                        candidate.get(
                                            "route_detach_steps", 80
                                        )
                                    ),
                                    0.20,
                                )
                                # A blocked lateral escape can still leave the
                                # selected pad in a collision-free, useful
                                # offset after the safety horizon. Continue
                                # with the live pose and lift from there.
                            current_pad = finger_positions(
                                env.env, geometry
                            )[selected]
                            route_lift = float(candidate.get('route_transition_lift_m', 0.0))
                            if route_lift > 0.0:
                                lift_pad = current_pad.copy()
                                lift_pad[2] += route_lift
                                obs, _, _ = move_pad_to(
                                    recorder, obs, geometry, selected, lift_pad,
                                    push_rotation, "route_lift", int(candidate.get('route_lift_steps', 20)), 0.16
                                )
                                lift_travel = float(candidate.get('route_transition_lift_travel_m', 0.0))
                                if lift_travel > 0.0:
                                    travel_pad = finger_positions(env.env, geometry)[selected].copy()
                                    travel_pad[:2] += direction * lift_travel
                                    obs, _, _ = move_pad_to(
                                        recorder, obs, geometry, selected, travel_pad,
                                        push_rotation, "route_lift_travel", int(candidate.get('route_lift_travel_steps', 30)), 0.16
                                    )
                                current_pad = finger_positions(env.env, geometry)[selected]
                                current_pad[2] -= route_lift
                            if candidate.get("route_skip_reacquire", False):
                                yaw_spec = candidate.get(
                                    "route_transition_yaw_delta_degrees", 0.0
                                )
                                if isinstance(yaw_spec, (list, tuple)):
                                    yaw_index = min(
                                        route_index - 1,
                                        len(yaw_spec) - 1,
                                    )
                                    yaw_value = float(yaw_spec[yaw_index])
                                else:
                                    yaw_value = float(yaw_spec)
                                transition_yaw_delta = math.radians(yaw_value)
                                cos_yaw = math.cos(transition_yaw_delta)
                                sin_yaw = math.sin(transition_yaw_delta)
                                transition_rotation = np.array(
                                    [
                                        [cos_yaw, -sin_yaw, 0.0],
                                        [sin_yaw, cos_yaw, 0.0],
                                        [0.0, 0.0, 1.0],
                                    ],
                                    dtype=np.float64,
                                ) @ push_rotation
                                push_rotation = transition_rotation
                                contact_pad = current_pad.copy()
                                live_xy = target_xyz(obs, target)[:2]
                                contact_offset_xy = (
                                    current_pad[:2] - live_xy
                                )
                                contact_follow_penetration = float(
                                    candidate.get(
                                        "route_contact_follow_penetration_m",
                                        contact_follow_penetration,
                                    )
                                )
                                if candidate.get(chr(34)+'route_recontact_reset_to_target'+chr(34), False):
                                    reset_offset = np.asarray(candidate.get(chr(34)+'route_recontact_offset_xy'+chr(34), contact_offset_xy), dtype=np.float64).reshape(2)
                                    contact_pad[:2] = live_xy + reset_offset
                                # Optional low-speed recontact phase for a
                                # route transition.  A direct direction
                                # change can leave the pad a few millimetres
                                # away from the object; continuing the push
                                # then silently loses the selected contact.
                                # Search only along the new push direction
                                # and stop on the first strict-safe contact.
                                if candidate.get(
                                    "route_transition_recontact_search",
                                    False,
                                ):
                                    recontact_steps = int(
                                        candidate.get(
                                            "route_transition_recontact_steps",
                                            search_steps,
                                        )
                                    )
                                    recontact_ok = False
                                    for _ in range(max(0, recontact_steps)):
                                        recontact_sign = float(
                                            candidate.get(
                                                "route_transition_recontact_sign",
                                                1.0,
                                            )
                                        )
                                        recontact_increment = float(
                                            candidate.get(
                                                chr(34)+'route_transition_recontact_increment_m'+chr(34),
                                                CONTACT_SEARCH_INCREMENT_M,
                                            )
                                        )
                                        contact_pad[:2] += (
                                            direction
                                            * recontact_sign
                                            * recontact_increment
                                        )
                                        obs, _, _ = move_pad_to(
                                            recorder,
                                            obs,
                                            geometry,
                                            selected,
                                            contact_pad,
                                            push_rotation,
                                            "route_recontact_search",
                                            5,
                                            0.12,
                                        )
                                        selected_now = (
                                            recorder.left_contact[-1]
                                            if selected == "left"
                                            else recorder.right_contact[-1]
                                        )
                                        opposite_now = (
                                            recorder.right_contact[-1]
                                            if selected == "left"
                                            else recorder.left_contact[-1]
                                        )
                                        if (
                                            selected_now
                                            and not opposite_now
                                            and not recorder.grasp_proxy[-1]
                                            and not recorder.nonselected_robot_target_contact[-1]
                                            and not recorder.robot_distractor_contact[-1]
                                        ):
                                            recontact_ok = True
                                            break
                                        if (
                                            opposite_now
                                            or recorder.grasp_proxy[-1]
                                            or recorder.nonselected_robot_target_contact[-1]
                                            or recorder.robot_distractor_contact[-1]
                                        ):
                                            break
                                    if not recontact_ok:
                                        # Let the normal terminal checks record
                                        # this as an incomplete route rather
                                        # than pushing without a verified pad.
                                        break
                                continue
                            route_pad_height_offset = float(
                                candidate.get(
                                    "route_pad_height_offset_m",
                                    candidate["pad_height_offset_m"],
                                )
                            )
                            route_high_z = (
                                live_target[2]
                                + route_pad_height_offset
                                + TRANSIT_PAD_CLEARANCE_M
                            )
                            lift_pad = current_pad.copy()
                            lift_pad[2] = route_high_z
                            obs, lift_ok, _ = move_pad_to(
                                recorder,
                                obs,
                                geometry,
                                selected,
                                lift_pad,
                                push_rotation,
                                "route_lift",
                                60,
                                0.25,
                            )
                            route_behind = np.array(
                                [
                                    live_target[0]
                                    - direction[0] * approach_standoff,
                                    live_target[1]
                                    - direction[1] * approach_standoff,
                                    live_target[2]
                                    + route_pad_height_offset,
                                ],
                                dtype=np.float64,
                            )
                            lateral_spec = candidate.get(
                                "route_pad_lateral_offsets_m",
                                candidate.get(
                                    "route_pad_lateral_offset_m", 0.0
                                ),
                            )
                            if isinstance(lateral_spec, (list, tuple)):
                                lateral_index = min(
                                    route_index - 1, len(lateral_spec) - 1
                                )
                                route_lateral_offset = float(
                                    lateral_spec[lateral_index]
                                )
                            else:
                                route_lateral_offset = float(lateral_spec)
                            route_behind[:2] += (
                                np.array(
                                    [-direction[1], direction[0]],
                                    dtype=np.float64,
                                )
                                * route_lateral_offset
                            )
                            route_high = route_behind.copy()
                            route_high[2] = route_high_z
                            yaw_spec = candidate.get(
                                "route_transition_yaw_delta_degrees", 0.0
                            )
                            if isinstance(yaw_spec, (list, tuple)):
                                yaw_index = min(
                                    route_index - 1, len(yaw_spec) - 1
                                )
                                yaw_value = float(yaw_spec[yaw_index])
                            else:
                                yaw_value = float(yaw_spec)
                            transition_yaw_delta = math.radians(yaw_value)
                            cos_yaw = math.cos(transition_yaw_delta)
                            sin_yaw = math.sin(transition_yaw_delta)
                            transition_rotation = np.array(
                                [
                                    [cos_yaw, -sin_yaw, 0.0],
                                    [sin_yaw, cos_yaw, 0.0],
                                    [0.0, 0.0, 1.0],
                                ],
                                dtype=np.float64,
                            ) @ push_rotation
                            obs, route_high_ok, _ = move_pad_to(
                                recorder,
                                obs,
                                geometry,
                                selected,
                                route_high,
                                transition_rotation,
                                "route_high",
                                80,
                                0.35,
                            )
                            obs, route_behind_ok, _ = move_pad_to(
                                recorder,
                                obs,
                                geometry,
                                selected,
                                route_behind,
                                transition_rotation,
                                "route_behind",
                                80,
                                0.25,
                            )
                            if not (
                                lift_ok
                                and route_high_ok
                                and route_behind_ok
                            ):
                                break
                            route_contact = False
                            contact_pad = route_behind.copy()
                            for _ in range(search_steps):
                                contact_pad[:2] += (
                                    direction
                                    * CONTACT_SEARCH_INCREMENT_M
                                )
                                obs, _, _ = move_pad_to(
                                    recorder,
                                    obs,
                                    geometry,
                                    selected,
                                    contact_pad,
                                    transition_rotation,
                                    "route_contact_search",
                                    5,
                                    0.12,
                                )
                                selected_contact = (
                                    recorder.left_contact[-1]
                                    if selected == "left"
                                    else recorder.right_contact[-1]
                                )
                                opposite_contact = (
                                    recorder.right_contact[-1]
                                    if selected == "left"
                                    else recorder.left_contact[-1]
                                )
                                route_contact_usable = selected_contact or (
                                    allow_auxiliary_target_contact
                                    and recorder.nonselected_robot_target_contact[-1]
                                )
                                if (
                                    route_contact_usable
                                    and not opposite_contact
                                    and not recorder.grasp_proxy[-1]
                                    and (
                                        allow_auxiliary_target_contact
                                        or not recorder.nonselected_robot_target_contact[-1]
                                    )
                                ):
                                    route_contact = True
                                    break
                                if (
                                    opposite_contact
                                    or recorder.grasp_proxy[-1]
                                    or (
                                        not allow_auxiliary_target_contact
                                        and recorder.nonselected_robot_target_contact[-1]
                                    )
                                ):
                                    break
                            if not route_contact:
                                break
                            push_rotation = transition_rotation
                            contact_follow_penetration = float(
                                candidate.get(
                                    "route_contact_follow_penetration_m",
                                    contact_follow_penetration,
                                )
                            )
                            live_xy = target_xyz(obs, target)[:2]
                        else:
                            live_xy = target_xyz(obs, target)[:2]
                            yaw_spec = candidate.get(
                                "route_transition_yaw_delta_degrees", 0.0
                            )
                            if isinstance(yaw_spec, (list, tuple)):
                                yaw_index = min(
                                    route_index - 1, len(yaw_spec) - 1
                                )
                                yaw_value = float(yaw_spec[yaw_index])
                            else:
                                yaw_value = float(yaw_spec)
                            transition_yaw_delta = math.radians(yaw_value)
                            cos_yaw = math.cos(transition_yaw_delta)
                            sin_yaw = math.sin(transition_yaw_delta)
                            transition_rotation = np.array(
                                [
                                    [cos_yaw, -sin_yaw, 0.0],
                                    [sin_yaw, cos_yaw, 0.0],
                                    [0.0, 0.0, 1.0],
                                ],
                                dtype=np.float64,
                            ) @ push_rotation
                            push_rotation = transition_rotation
                            contact_offset_xy = (
                                finger_positions(env.env, geometry)[selected][:2]
                                - live_xy
                            )
                        contact_offset_xy = (
                            finger_positions(env.env, geometry)[selected][:2]
                            - live_xy
                        )
                if (
                    route_points is None
                    and (
                        "controller_destination_xy" in candidate
                        or "push_destination_xy" in candidate
                    )
                    and np.linalg.norm(
                        target_xyz(obs, target)[:2] - push_destination
                    )
                    <= controller_destination_tolerance
                ):
                    break
                if recorder.grasp_proxy[-1]:
                    break
                if (
                    not allow_auxiliary_target_contact
                    and recorder.nonselected_robot_target_contact[-1]
                ):
                    break

        relation_now, xy_now = target_region_status(env, obs, row)
        if relation_now and xy_now:
            retreat_pad = finger_positions(env.env, geometry)[selected]
            retreat_distance = float(
                candidate.get("retreat_distance_m", 0.06)
            )
            if retreat_distance < 0.0 or retreat_distance > 0.12:
                raise ValueError(
                    f"Invalid retreat distance: {retreat_distance}"
                )
            retreat_pad[:2] -= direction * retreat_distance
            obs, _, _ = move_pad_to(
                recorder,
                obs,
                geometry,
                selected,
                retreat_pad,
                settled["desired_rotation"],
                "retreat",
                40,
                0.25,
            )

        hold_start = len(recorder.actions)
        hold_reference_poses = reset_validator.capture_poses(obs, None)
        hold_motion = []
        hold_orientation = []
        for _ in range(TERMINAL_HOLD_STEPS):
            action = np.zeros(7, dtype=np.float32)
            obs = recorder.step(obs, action, "terminal_hold")
            poses = reset_validator.capture_poses(obs, None)
            motion, orientation = reset_validator.max_pose_motion(
                hold_reference_poses, poses
            )
            hold_motion.append(motion)
            hold_orientation.append(orientation)
            hold_reference_poses = poses

        arrays = recorder.arrays()
        hold_slice = slice(hold_start, len(recorder.actions))
        final_window = min(FINAL_WINDOW_STEPS, len(hold_motion))
        max_final_motion = (
            max(hold_motion[-final_window:]) if final_window else math.inf
        )
        max_final_orientation = (
            max(hold_orientation[-final_window:])
            if final_window
            else math.inf
        )
        selected_contacts = (
            arrays["left_contact"]
            if selected == "left"
            else arrays["right_contact"]
        )
        opposite_contacts = (
            arrays["right_contact"]
            if selected == "left"
            else arrays["left_contact"]
        )
        target_z = arrays["target_xyz"][:, 2]
        max_lift = float(
            max(0.0, np.max(target_z) - settled["stabilized_z"])
        ) if target_z.size else math.inf
        terminal_relation = bool(
            arrays["relation"][hold_slice].size
            == TERMINAL_HOLD_STEPS
            and arrays["relation"][hold_slice].all()
        )
        terminal_xy = bool(
            arrays["xy_in_target"][hold_slice].size
            == TERMINAL_HOLD_STEPS
            and arrays["xy_in_target"][hold_slice].all()
        )
        terminal_support = bool(
            arrays["table_support"][hold_slice].size
            == TERMINAL_HOLD_STEPS
            and arrays["table_support"][hold_slice].all()
        )
        no_bilateral = bool(
            not np.any(
                arrays["left_contact"] & arrays["right_contact"]
            )
            and not arrays["grasp_proxy"].any()
        )
        opposite_never_contacts = bool(not opposite_contacts.any())
        no_unexpected_object_contact = bool(
            not arrays["unexpected_object_contact"].any()
        )
        no_robot_distractor_contact = bool(
            not arrays["robot_distractor_contact"].any()
        )
        no_nonselected_robot_target_contact = bool(
            not arrays["nonselected_robot_target_contact"].any()
        )
        support_all = bool(arrays["table_support"].all())
        terminal_stable = bool(
            max_final_motion <= MAX_FINAL_MOTION_M
            and max_final_orientation <= MAX_FINAL_ORIENTATION_DEG
        )
        failed = []
        checks = {
            "high_transit_reached": high_ok,
            "behind_pose_reached": behind_ok,
            "one_finger_contact_acquired": contact_step is not None,
            "terminal_relation_hold": terminal_relation,
            "terminal_xy_hold": terminal_xy,
            "terminal_table_support_hold": terminal_support,
            "terminal_stable": terminal_stable,
            "no_bilateral_contact_or_grasp": no_bilateral,
            "opposite_pad_never_contacts": opposite_never_contacts,
            "no_unexpected_object_contacts": (
                no_unexpected_object_contact
            ),
            "no_robot_distractor_contacts": (
                no_robot_distractor_contact
            ),
            "selected_fingerpad_only_target_contact": (
                no_nonselected_robot_target_contact
            ),
            "target_table_support_all_active_steps": support_all,
            "provisional_lift_check": max_lift <= MAX_LIFT_PROVISIONAL_M,
        }
        failed.extend(name for name, ok in checks.items() if not ok)
        return {
            "settled": settled,
            "arrays": arrays,
            "selected_offset": selected_offset,
            "opposite_offset": opposite_offset,
            "direction": direction,
            "destination": destination,
            "approach_standoff": approach_standoff,
            "transit_pad": transit_pad,
            "behind_pad": behind_pad,
            "transit_diagnostics": transit_diagnostics,
            "behind_diagnostics": behind_diagnostics,
            "contact_step": contact_step,
            "relation_ever_true": bool(arrays["relation"].any()),
            "terminal_relation": terminal_relation,
            "terminal_xy": terminal_xy,
            "terminal_support": terminal_support,
            "terminal_stable": terminal_stable,
            "max_final_motion": max_final_motion,
            "max_final_orientation": max_final_orientation,
            "selected_contact_steps": int(selected_contacts.sum()),
            "opposite_contact_steps": int(opposite_contacts.sum()),
            "bilateral_steps": int(
                np.sum(
                    arrays["left_contact"]
                    & arrays["right_contact"]
                )
            ),
            "grasp_steps": int(arrays["grasp_proxy"].sum()),
            "no_bilateral": no_bilateral,
            "opposite_never_contacts": opposite_never_contacts,
            "unexpected_object_contact_steps": int(
                arrays["unexpected_object_contact"].sum()
            ),
            "robot_distractor_contact_steps": int(
                arrays["robot_distractor_contact"].sum()
            ),
            "nonselected_robot_target_contact_steps": int(
                arrays["nonselected_robot_target_contact"].sum()
            ),
            "support_all": support_all,
            "max_lift": max_lift,
            "failed": failed,
        }
    except Exception as error:
        if recorder is not None:
            error.gate5_attempt_arrays = recorder.arrays()
        if settled is not None:
            error.gate5_settled_evidence = settled
        raise
    finally:
        env.close()


def replay_attempt(
    row: dict,
    seed: int,
    actions: np.ndarray,
    selected_side: str,
    allow_non_grasping_gripper_motion: bool = False,
) -> dict:
    target = target_instance(row)
    env = reset_validator.make_environment(Path(row["bddl_path"]))
    try:
        settled = settle_environment(env, row, seed)
        geometry = contact_geometry(env, target)
        obs = settled["obs"]
        states = [
            np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        ]
        left_trace = []
        right_trace = []
        grasp_trace = []
        support_trace = []
        object_contact_trace = []
        robot_distractor_trace = []
        nonselected_target_trace = []
        robot_target_pairs_trace = []
        relation_trace = []
        xy_trace = []
        hold_start = len(actions) - TERMINAL_HOLD_STEPS
        if hold_start < 0:
            raise ValueError("Recorded trajectory is shorter than hold")
        for index, action in enumerate(actions):
            action = np.asarray(action)
            if (
                action.shape != (7,)
                or action.dtype != np.float32
                or not np.isfinite(action).all()
                or np.any(action < -1.0)
                or np.any(action > 1.0)
            ):
                raise ValueError(f"Invalid recorded action {index}")
            if (
                index < hold_start
                and not allow_non_grasping_gripper_motion
                and float(action[6]) != -1.0
            ):
                raise ValueError(
                    f"Active replay action {index} does not keep "
                    "the gripper open"
                )
            if index >= hold_start and not np.array_equal(
                action, np.zeros(7, dtype=np.float32)
            ):
                raise ValueError(
                    f"Terminal replay action {index} is not all-zero"
                )
            obs, _, done, _ = env.step(
                action
            )
            states.append(
                np.asarray(
                    env.get_sim_state(), dtype=np.float64
                ).copy()
            )
            left, right, grasp = push_contact_record(env, geometry)
            (
                object_contact,
                robot_distractor,
                nonselected_target,
            ) = active_unexpected_contacts(
                env,
                target,
                geometry,
                selected_side,
            )
            relation, xy_ok = target_region_status(env, obs, row)
            if bool(done) != bool(relation):
                raise RuntimeError(
                    "Replay done flag disagrees with goal predicate: "
                    f"done={done}, relation={relation}, "
                    f"action={index + 1}"
                )
            left_trace.append(left)
            right_trace.append(right)
            grasp_trace.append(grasp)
            support_trace.append(table_support(env.env, target))
            object_contact_trace.append(object_contact)
            robot_distractor_trace.append(robot_distractor)
            nonselected_target_trace.append(nonselected_target)
            robot_target_pairs_trace.append(
                json_compact(robot_target_contact_pairs(env.env, geometry))
            )
            relation_trace.append(relation)
            xy_trace.append(xy_ok)
        return {
            "settled": settled,
            "actions": np.asarray(actions, dtype=np.float32).copy(),
            "states": np.asarray(states, dtype=np.float64),
            "left_contact": np.asarray(left_trace, dtype=np.bool_),
            "right_contact": np.asarray(right_trace, dtype=np.bool_),
            "grasp_proxy": np.asarray(grasp_trace, dtype=np.bool_),
            "table_support": np.asarray(support_trace, dtype=np.bool_),
            "unexpected_object_contact": np.asarray(
                object_contact_trace, dtype=np.bool_
            ),
            "robot_distractor_contact": np.asarray(
                robot_distractor_trace, dtype=np.bool_
            ),
            "nonselected_robot_target_contact": np.asarray(
                nonselected_target_trace, dtype=np.bool_
            ),
            "robot_target_contact_pairs": np.asarray(
                robot_target_pairs_trace, dtype="U1024"
            ),
            "relation": np.asarray(relation_trace, dtype=np.bool_),
            "xy_in_target": np.asarray(xy_trace, dtype=np.bool_),
        }
    finally:
        env.close()


def trace_matches(original: dict, replay: dict) -> bool:
    keys = [
        "left_contact",
        "right_contact",
        "grasp_proxy",
        "table_support",
        "unexpected_object_contact",
        "robot_distractor_contact",
        "nonselected_robot_target_contact",
        "robot_target_contact_pairs",
        "relation",
        "xy_in_target",
    ]
    return all(
        np.array_equal(original[key], replay[key]) for key in keys
    )


def base_output(row: dict, candidate: dict, seed: int) -> dict:
    output = {field: "" for field in OUTPUT_FIELDS}
    output.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "benchmark_name": BENCHMARK_NAME,
            "calibration_protocol": CALIBRATION_PROTOCOL,
            "calibration_id": CALIBRATION_ID,
            "diagnostic_only": True,
            "task_id": row["task_id"],
            "layout_id": row["layout_id"],
            "tuple": row["tuple"],
            "object": row["object"],
            "skill": row["skill"],
            "spatial_region": row["spatial_region"],
            "target_instance": target_instance(row),
            "seed": seed,
            "gate3_reset_index": row["gate3_reset_index"],
            "candidate_index": candidate["candidate_index"],
            "pusher_finger": candidate["pusher_finger"],
            "pad_height_offset_m": candidate["pad_height_offset_m"],
            "candidate_order_key": candidate["candidate_order_key"],
            "bddl_path": row["bddl_path"],
            "bddl_sha256": row["bddl_sha256"],
            "object_to_slot_json": json_compact(row["mapping"]),
            "provisional_lift_threshold_m": MAX_LIFT_PROVISIONAL_M,
            "trajectory_threshold_promoted": False,
            "attempt_succeeded": False,
            "failed_checks_json": json_compact(["not_run"]),
        }
    )
    return output


def run_attempt(
    args,
    row: dict,
    candidate: dict,
    attempts_dir: Path,
) -> dict:
    seed = int(row["calibration_seed"])
    output = base_output(row, candidate, seed)
    output["mode"] = args.mode
    npz_path = attempts_dir / (
        f"task_{row['task_id']:02d}_layout_1_"
        f"candidate_{candidate['candidate_index']:02d}.npz"
    )
    arrays = empty_attempt_arrays()
    replay_arrays = empty_attempt_arrays()
    try:
        original = run_original_attempt(row, candidate, seed)
        arrays = original["arrays"]
        replay = replay_attempt(
            row,
            seed,
            arrays["actions"],
            candidate["pusher_finger"],
        )
        settled = original["settled"]
        replay_settled = replay["settled"]
        arrays["raw_initial_state"] = settled["initial_state"]
        arrays["stabilized_target_z_m"] = np.asarray(
            [settled["stabilized_z"]], dtype=np.float64
        )
        arrays["settle_actions"] = settled["settle_actions"]
        arrays["settle_states"] = settled["settle_states"]
        arrays["settle_target_table_support"] = settled[
            "settle_target_table_support"
        ]
        arrays["settle_source_official"] = settled[
            "settle_source_official"
        ]
        arrays["settle_source_xy"] = settled["settle_source_xy"]
        replay_arrays["replay_raw_initial_state"] = replay_settled[
            "initial_state"
        ]
        replay_arrays["replay_settle_actions"] = replay_settled[
            "settle_actions"
        ]
        replay_arrays["replay_settle_states"] = replay_settled[
            "settle_states"
        ]
        replay_arrays["replay_settle_target_table_support"] = (
            replay_settled["settle_target_table_support"]
        )
        replay_arrays["replay_settle_source_official"] = (
            replay_settled["settle_source_official"]
        )
        replay_arrays["replay_settle_source_xy"] = replay_settled[
            "settle_source_xy"
        ]
        replay_arrays["replay_actions"] = replay["actions"]
        replay_arrays["replay_states"] = replay["states"]
        for key in [
            "left_contact",
            "right_contact",
            "grasp_proxy",
            "table_support",
            "unexpected_object_contact",
            "robot_distractor_contact",
            "nonselected_robot_target_contact",
            "robot_target_contact_pairs",
            "relation",
            "xy_in_target",
        ]:
            replay_arrays[f"replay_{key}"] = replay[key]
        raw_initial_difference = max_state_difference(
            settled["initial_state"], replay_settled["initial_state"]
        )
        settle_trajectory_difference = max_state_difference(
            settled["settle_states"], replay_settled["settle_states"]
        )
        settled_difference = max_state_difference(
            settled["settled_state"], replay_settled["settled_state"]
        )
        trajectory_difference = max_state_difference(
            arrays["states"], replay["states"]
        )
        terminal_difference = max_state_difference(
            arrays["states"][-(TERMINAL_HOLD_STEPS + 1):],
            replay["states"][-(TERMINAL_HOLD_STEPS + 1):],
        )
        action_difference = max_state_difference(
            arrays["actions"], replay["actions"]
        )
        actions_byte_exact = bool(
            arrays["actions"].dtype == replay["actions"].dtype
            and arrays["actions"].shape == replay["actions"].shape
            and arrays["actions"].tobytes(order="C")
            == replay["actions"].tobytes(order="C")
        )
        reset_attempts_match = bool(
            settled["attempts"] == replay_settled["attempts"]
        )
        camera_record_match = bool(
            settled["camera_record"] == replay_settled["camera_record"]
        )
        settle_actions_byte_exact = bool(
            settled["settle_actions"].dtype
            == replay_settled["settle_actions"].dtype
            and settled["settle_actions"].shape
            == replay_settled["settle_actions"].shape
            and settled["settle_actions"].tobytes(order="C")
            == replay_settled["settle_actions"].tobytes(order="C")
        )
        settle_support_trace_match = bool(
            np.array_equal(
                settled["settle_target_table_support"],
                replay_settled["settle_target_table_support"],
            )
        )
        settle_source_trace_match = bool(
            np.array_equal(
                settled["settle_source_official"],
                replay_settled["settle_source_official"],
            )
            and np.array_equal(
                settled["settle_source_xy"],
                replay_settled["settle_source_xy"],
            )
        )
        trace_match = trace_matches(arrays, replay)
        deterministic = bool(
            raw_initial_difference <= STATE_REPLAY_ATOL
            and settle_trajectory_difference <= STATE_REPLAY_ATOL
            and settled_difference <= STATE_REPLAY_ATOL
            and trajectory_difference <= STATE_REPLAY_ATOL
            and terminal_difference <= STATE_REPLAY_ATOL
            and action_difference == 0.0
            and actions_byte_exact
            and reset_attempts_match
            and camera_record_match
            and settle_actions_byte_exact
            and settle_support_trace_match
            and settle_source_trace_match
            and trace_match
        )
        failed = list(original["failed"])
        if not deterministic:
            failed.append("exact_same_seed_action_replay")
        failed = sorted(set(failed))
        z_values = settled["z_values"]
        output.update(
            {
                "reset_attempts": settled["attempts"],
                "replay_reset_attempts": replay["settled"]["attempts"],
                "camera_record_json": json_compact(
                    settled["camera_record"]
                ),
                "initial_sim_state_sha256": sha256_array(
                    settled["initial_state"]
                ),
                "passive_settle_state_sha256": sha256_array(
                    settled["settled_state"]
                ),
                "passive_z_min_m": float(np.min(z_values)),
                "passive_z_max_m": float(np.max(z_values)),
                "passive_z_range_m": float(
                    np.max(z_values) - np.min(z_values)
                ),
                "max_passive_positive_z_noise_m": settled[
                    "max_positive_z_noise"
                ],
                "stabilized_target_z_m": settled["stabilized_z"],
                "settle_final_window_table_support_ok": bool(
                    settled["settle_target_table_support"][
                        -FINAL_WINDOW_STEPS:
                    ].all()
                ),
                "settle_final_window_source_official_ok": bool(
                    settled["settle_source_official"][
                        -FINAL_WINDOW_STEPS:
                    ].all()
                ),
                "settle_final_window_source_xy_ok": bool(
                    settled["settle_source_xy"][
                        -FINAL_WINDOW_STEPS:
                    ].all()
                ),
                "max_settle_horizontal_drift_m": settled[
                    "max_horizontal_drift"
                ],
                "max_settle_orientation_drift_deg": settled[
                    "max_orientation_drift"
                ],
                "selected_pad_initial_offset_json": json_compact(
                    original["selected_offset"].tolist()
                ),
                "opposite_pad_initial_offset_json": json_compact(
                    original["opposite_offset"].tolist()
                ),
                "push_direction_json": json_compact(
                    original["direction"].tolist()
                ),
                "destination_xy_json": json_compact(
                    original["destination"].tolist()
                ),
                "approach_standoff_m": original[
                    "approach_standoff"
                ],
                "transit_pad_clearance_m": (
                    TRANSIT_PAD_CLEARANCE_M
                ),
                "transit_pad_xyz_json": json_compact(
                    original["transit_pad"].tolist()
                ),
                "behind_pad_xyz_json": json_compact(
                    original["behind_pad"].tolist()
                ),
                "transit_min_pad_error_m": original[
                    "transit_diagnostics"
                ]["min_position_error_m"],
                "transit_final_pad_error_m": original[
                    "transit_diagnostics"
                ]["final_position_error_m"],
                "transit_min_rotation_error_rad": original[
                    "transit_diagnostics"
                ]["min_rotation_error_rad"],
                "transit_final_rotation_error_rad": original[
                    "transit_diagnostics"
                ]["final_rotation_error_rad"],
                "transit_longest_valid_run": original[
                    "transit_diagnostics"
                ]["longest_valid_run"],
                "behind_min_pad_error_m": original[
                    "behind_diagnostics"
                ]["min_position_error_m"],
                "behind_final_pad_error_m": original[
                    "behind_diagnostics"
                ]["final_position_error_m"],
                "behind_min_rotation_error_rad": original[
                    "behind_diagnostics"
                ]["min_rotation_error_rad"],
                "behind_final_rotation_error_rad": original[
                    "behind_diagnostics"
                ]["final_rotation_error_rad"],
                "behind_longest_valid_run": original[
                    "behind_diagnostics"
                ]["longest_valid_run"],
                "contact_acquired": original["contact_step"] is not None,
                "contact_acquired_step": (
                    ""
                    if original["contact_step"] is None
                    else original["contact_step"]
                ),
                "relation_ever_true": original["relation_ever_true"],
                "terminal_hold_complete": bool(
                    arrays["phase"].size >= TERMINAL_HOLD_STEPS
                    and np.all(
                        arrays["phase"][-TERMINAL_HOLD_STEPS:]
                        == "terminal_hold"
                    )
                ),
                "terminal_relation_hold_ok": original[
                    "terminal_relation"
                ],
                "terminal_xy_hold_ok": original["terminal_xy"],
                "terminal_table_support_hold_ok": original[
                    "terminal_support"
                ],
                "terminal_stable": original["terminal_stable"],
                "max_final_window_motion_m": original[
                    "max_final_motion"
                ],
                "max_final_window_orientation_deg": original[
                    "max_final_orientation"
                ],
                "selected_pad_contact_steps": original[
                    "selected_contact_steps"
                ],
                "opposite_pad_contact_steps": original[
                    "opposite_contact_steps"
                ],
                "bilateral_contact_steps": original["bilateral_steps"],
                "grasp_proxy_steps": original["grasp_steps"],
                "opposite_pad_never_contacts": original[
                    "opposite_never_contacts"
                ],
                "unexpected_object_contact_steps": original[
                    "unexpected_object_contact_steps"
                ],
                "robot_distractor_contact_steps": original[
                    "robot_distractor_contact_steps"
                ],
                "nonselected_robot_target_contact_steps": original[
                    "nonselected_robot_target_contact_steps"
                ],
                "no_bilateral_contact_or_grasp": original[
                    "no_bilateral"
                ],
                "target_table_support_all_active_steps": original[
                    "support_all"
                ],
                "max_target_lift_m": original["max_lift"],
                "provisional_lift_check_passed": (
                    original["max_lift"] <= MAX_LIFT_PROVISIONAL_M
                ),
                "original_steps": len(arrays["actions"]),
                "replay_steps": len(replay["states"]) - 1,
                "replay_raw_initial_state_max_abs": (
                    raw_initial_difference
                ),
                "replay_settle_trajectory_state_max_abs": (
                    settle_trajectory_difference
                ),
                "replay_settled_state_max_abs": settled_difference,
                "replay_initial_state_max_abs": settled_difference,
                "replay_trajectory_state_max_abs": trajectory_difference,
                "replay_terminal_state_max_abs": terminal_difference,
                "replay_action_max_abs": action_difference,
                "replay_actions_byte_exact": actions_byte_exact,
                "replay_reset_attempts_match": reset_attempts_match,
                "replay_camera_record_match": camera_record_match,
                "replay_trace_match": trace_match,
                "deterministic_replay": deterministic,
                "attempt_succeeded": not failed,
                "failed_checks_json": json_compact(failed),
                "error": "",
            }
        )
    except Exception as error:
        partial_arrays = getattr(error, "gate5_attempt_arrays", None)
        if partial_arrays is not None:
            arrays = partial_arrays
        partial_settled = getattr(error, "gate5_settled_evidence", None)
        if partial_settled is not None:
            arrays["raw_initial_state"] = partial_settled[
                "initial_state"
            ]
            arrays["settle_actions"] = partial_settled[
                "settle_actions"
            ]
            arrays["settle_states"] = partial_settled[
                "settle_states"
            ]
            arrays["settle_target_table_support"] = partial_settled[
                "settle_target_table_support"
            ]
            arrays["settle_source_official"] = partial_settled[
                "settle_source_official"
            ]
            arrays["settle_source_xy"] = partial_settled[
                "settle_source_xy"
            ]
        output.update(
            {
                "failed_checks_json": json_compact(["exception"]),
                "attempt_succeeded": False,
                "error": f"{type(error).__name__}: {error}",
            }
        )
    combined = dict(arrays)
    for key in [
        "replay_raw_initial_state",
        "replay_settle_actions",
        "replay_settle_states",
        "replay_settle_target_table_support",
        "replay_settle_source_official",
        "replay_settle_source_xy",
        "replay_actions",
        "replay_states",
        "replay_left_contact",
        "replay_right_contact",
        "replay_grasp_proxy",
        "replay_table_support",
        "replay_unexpected_object_contact",
        "replay_robot_distractor_contact",
        "replay_nonselected_robot_target_contact",
        "replay_robot_target_contact_pairs",
        "replay_relation",
        "replay_xy_in_target",
    ]:
        combined[key] = replay_arrays[key]
    combined["metadata_json"] = np.asarray(
        [
            json_compact(
                {
                    "protocol": CALIBRATION_PROTOCOL,
                    "task_id": row["task_id"],
                    "layout_id": CALIBRATION_LAYOUT_ID,
                    "candidate": {
                        "candidate_index": candidate.get("candidate_index", 0),
                        "pusher_finger": candidate.get("pusher_finger"),
                        "active_action_limit": candidate.get("active_action_limit"),
                        "controller": candidate.get("controller", "deterministic_closed_loop"),
                    },
                    "seed": seed,
                    "diagnostic_only": True,
                }
            )
        ],
        dtype="U1024",
    )
    write_npz_atomic(npz_path, **combined)
    output["npz_path"] = str(npz_path)
    output["npz_sha256"] = sha256_file(npz_path)
    return output


def exact_attempt_coverage(
    rows: list[dict], selected: list[dict]
) -> bool:
    expected = {
        (row["task_id"], candidate["candidate_index"])
        for row in selected
        for candidate in candidates_for(row)
    }
    actual = {
        (int(row["task_id"]), int(row["candidate_index"]))
        for row in rows
    }
    return actual == expected and len(rows) == len(expected)


def make_manifest(
    args,
    layout_spec: Path,
    reset_manifest: Path,
    reset_prerequisite: dict,
    goal_manifest: Path,
    goal_prerequisite: dict,
    output: Path,
    attempts_dir: Path,
    selected: list[dict],
    results: list[dict],
    run_complete: bool,
) -> dict:
    expected_attempts = sum(len(candidates_for(row)) for row in selected)
    coverage = exact_attempt_coverage(results, selected)
    full_shape = bool(
        args.mode == "full"
        and tuple(row["task_id"] for row in selected)
        == FULL_PUSH_TASK_IDS
        and len(results) == 72
        and expected_attempts == 72
        and coverage
    )
    tasks_with_success = sorted(
        {
            int(row["task_id"])
            for row in results
            if bool(row["attempt_succeeded"])
        }
    )
    counts = Counter(
        (
            row["object"],
            row["spatial_region"],
            bool(row["attempt_succeeded"]),
        )
        for row in results
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_name": BENCHMARK_NAME,
        "calibration_protocol": CALIBRATION_PROTOCOL,
        "calibration_id": CALIBRATION_ID,
        "diagnostic_only": True,
        "mode": args.mode,
        "git_commit": git_commit(),
        "validator_path": str(Path(__file__).resolve()),
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "layout_spec": str(layout_spec),
        "layout_spec_sha256": sha256_file(layout_spec),
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
        "goal_validator_path": str(
            Path(goal_validator.__file__).resolve()
        ),
        "goal_validator_sha256": sha256_file(
            Path(goal_validator.__file__).resolve()
        ),
        "reset_manifest": str(reset_manifest),
        "reset_manifest_sha256": sha256_file(reset_manifest),
        "reset_output": reset_prerequisite["validated_output_path"],
        "reset_output_sha256": reset_prerequisite[
            "validated_output_sha256"
        ],
        "goal_manifest": str(goal_manifest),
        "goal_manifest_sha256": sha256_file(goal_manifest),
        "goal_output": goal_prerequisite["validated_output_path"],
        "goal_output_sha256": goal_prerequisite[
            "validated_output_sha256"
        ],
        "frozen_observation_camera": camera_config.frozen_camera_spec(),
        "software_versions": {
            name: goal_validator.package_version(name)
            for name in [
                "hf_libero",
                "libero",
                "robosuite",
                "mujoco",
                "numpy",
            ]
        },
        "output": str(output),
        "output_sha256": sha256_file(output) if output.is_file() else None,
        "attempts_dir": str(attempts_dir),
        "npz_files": len(list(attempts_dir.glob("*.npz"))),
        "npz_artifacts": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for path in sorted(attempts_dir.glob("*.npz"))
        ],
        "parameters": {
            "calibration_layout_id": CALIBRATION_LAYOUT_ID,
            "seed_source": (
                "validated_gate3_full_reset_audit_layout1_reset0"
            ),
            "gate3_reset_index": GATE3_CALIBRATION_RESET_INDEX,
            "task_ids": [row["task_id"] for row in selected],
            "candidate_height_deltas_m": HEIGHT_DELTAS_M,
            "candidate_finger_order": PUSH_FINGERS,
            "base_pad_height_offsets_m": BASE_PAD_HEIGHT_OFFSET_M,
            "precontact_pad_clearance_m": (
                PRECONTACT_PAD_CLEARANCE_M
            ),
            "approach_standoff_formula": (
                "frozen_source_sampler_radius + "
                "precontact_pad_clearance_m"
            ),
            "transit_pad_clearance_m": TRANSIT_PAD_CLEARANCE_M,
            "transit_waypoint_xy": "live_target_center_xy",
            "push_increment_m": PUSH_INCREMENT_M,
            "active_gripper_action": -1.0,
            "terminal_hold_action": [0.0] * 7,
            "settle_steps": SETTLE_STEPS,
            "terminal_hold_steps": TERMINAL_HOLD_STEPS,
            "state_replay_atol": STATE_REPLAY_ATOL,
            "provisional_max_lift_m": MAX_LIFT_PROVISIONAL_M,
            "deterministic_candidate_order": True,
            "one_finger_live_offset_controller": True,
        },
        "expected_attempts": expected_attempts,
        "completed_attempts": len(results),
        "successful_attempts": sum(
            bool(row["attempt_succeeded"]) for row in results
        ),
        "failed_attempts": sum(
            not bool(row["attempt_succeeded"]) for row in results
        ),
        "tasks_with_success": tasks_with_success,
        "tasks_with_success_count": len(tasks_with_success),
        "counts": [
            {
                "object": obj,
                "spatial_region": region,
                "attempt_succeeded": success,
                "count": count,
            }
            for (obj, region, success), count in sorted(counts.items())
        ],
        "run_complete": run_complete,
        "exact_attempt_coverage": coverage,
        "full_protocol_shape": full_shape,
        "gate3_required_and_validated": True,
        "gate4_required_and_validated": True,
        "push_trajectory_constraints_evaluated": bool(results),
        "push_lift_threshold_calibrated": False,
        "push_lift_threshold_frozen": False,
        "trajectory_threshold_promoted": False,
        "gate5_passed": False,
        "all_pretraining_gates_passed": False,
    }


def main():
    args = parse_args()
    layout_spec = resolve_repo_path(args.layout_spec)
    reset_manifest = resolve_repo_path(args.reset_manifest)
    goal_manifest = resolve_repo_path(args.goal_manifest)
    output = resolve_repo_path(args.output)
    manifest = (
        resolve_repo_path(args.manifest)
        if args.manifest is not None
        else output.with_suffix(".manifest.json")
    )
    attempts_dir = (
        resolve_repo_path(args.attempts_dir)
        if args.attempts_dir is not None
        else output.with_name(output.stem + "_attempts")
    )
    goal_validator.disable_upstream_interactive_debugger()
    goal_validator.validate_runtime_versions()
    reset_prerequisite = goal_validator.validate_reset_gate(
        reset_manifest, layout_spec
    )
    recorded_reset_output_sha256 = reset_prerequisite.get(
        "output_sha256"
    )
    if (
        recorded_reset_output_sha256 is not None
        and recorded_reset_output_sha256
        != reset_prerequisite["validated_output_sha256"]
    ):
        raise ValueError("Gate-3 manifest output hash does not match CSV")
    goal_prerequisite = validate_gate4(
        goal_manifest,
        layout_spec,
        reset_manifest,
        reset_prerequisite,
    )
    selected = selected_rows(layout_spec, args)
    bind_validated_gate3_seeds(selected, reset_prerequisite)
    # Destructive overwrite is deliberately delayed until every frozen
    # prerequisite and every selected BDDL hash has been validated.
    prepare_output_paths(
        output, manifest, attempts_dir, args.overwrite
    )
    expected_attempts = sum(
        len(candidates_for(row)) for row in selected
    )
    results = []
    write_json_atomic(
        manifest,
        make_manifest(
            args,
            layout_spec,
            reset_manifest,
            reset_prerequisite,
            goal_manifest,
            goal_prerequisite,
            output,
            attempts_dir,
            selected,
            results,
            run_complete=False,
        ),
    )
    print("=" * 80)
    print("LIBERO 36 Gate-5A push calibration pilot")
    print("mode:", args.mode)
    print("diagnostic only:", True)
    print("calibration layout:", CALIBRATION_LAYOUT_ID)
    print("push tasks:", [row["task_id"] for row in selected])
    print("expected attempts:", expected_attempts)
    print("active gripper action:", -1.0)
    print(
        "precontact clearance beyond sampler radius:",
        PRECONTACT_PAD_CLEARANCE_M,
    )
    print("transit clearance above live pad height:", TRANSIT_PAD_CLEARANCE_M)
    print("output:", output)
    print("manifest:", manifest)
    print("attempt NPZ directory:", attempts_dir)
    print(
        "NOTE: this pilot does not freeze or promote a push-lift "
        "threshold and cannot pass Gate 5."
    )
    run_index = 0
    for row in selected:
        for candidate in candidates_for(row):
            run_index += 1
            result = run_attempt(args, row, candidate, attempts_dir)
            results.append(result)
            write_csv_atomic(output, results)
            write_json_atomic(
                manifest,
                make_manifest(
                    args,
                    layout_spec,
                    reset_manifest,
                    reset_prerequisite,
                    goal_manifest,
                    goal_prerequisite,
                    output,
                    attempts_dir,
                    selected,
                    results,
                    run_complete=False,
                ),
            )
            print(
                f"[{run_index:03d}/{expected_attempts:03d}] "
                f"task={row['task_id']:02d} "
                f"object={row['object']} "
                f"region={row['spatial_region']} "
                f"candidate={candidate['candidate_index']:02d} "
                f"finger={candidate['pusher_finger']} "
                f"height={candidate['pad_height_offset_m']:+.3f} "
                f"success={result['attempt_succeeded']} "
                f"error={result['error']}"
            )
            if not result["error"]:
                print(
                    "    waypoint min/final error: "
                    f"transit="
                    f"{100 * float(result['transit_min_pad_error_m']):.2f}/"
                    f"{100 * float(result['transit_final_pad_error_m']):.2f} cm, "
                    f"behind="
                    f"{100 * float(result['behind_min_pad_error_m']):.2f}/"
                    f"{100 * float(result['behind_final_pad_error_m']):.2f} cm"
                )
            if args.fail_fast and result["error"]:
                raise RuntimeError(
                    "Gate-5A calibration encountered an exception"
                )
    final = make_manifest(
        args,
        layout_spec,
        reset_manifest,
        reset_prerequisite,
        goal_manifest,
        goal_prerequisite,
        output,
        attempts_dir,
        selected,
        results,
        run_complete=True,
    )
    write_json_atomic(manifest, final)
    print("=" * 80)
    print("Gate-5A diagnostic calibration complete")
    print("attempts:", len(results))
    print("successful attempts:", final["successful_attempts"])
    print("tasks with >=1 success:", final["tasks_with_success_count"])
    print("exact attempt coverage:", final["exact_attempt_coverage"])
    print("threshold frozen:", False)
    print("gate5 passed:", False)
    print("all pretraining gates passed:", False)


if __name__ == "__main__":
    main()
