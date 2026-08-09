from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import calibrate_libero_36_push as push_calibrator
    import generate_libero_36_bddl as benchmark_generator
    import validate_libero_36_envs as reset_validator
except ModuleNotFoundError:
    from scripts import calibrate_libero_36_push as push_calibrator
    from scripts import generate_libero_36_bddl as benchmark_generator
    from scripts import validate_libero_36_envs as reset_validator


DIAGNOSTIC_PROTOCOL = "libero_36_source_xy_candidate_preflight_v1"
EXPECTED_BASE_PROTOCOL = "libero_36_proxy_tabletop_draft_v4"
ALIGN_X_MAX_STEPS = 200

# These candidates were declared before running this preflight.  They keep
# the reset-audited v4 y bands and move only the source row toward the robot.
# Selection is rightmost first: test c0, then c1 only if c0 fails, then c2.
CANDIDATE_X_BOUNDS = {
    "c0": (0.00, 0.07),
    "c1": (0.04, 0.11),
    "c2": (-0.02, 0.05),
}
FROZEN_Y_BOUNDS = {
    0: (-0.244, -0.182),
    1: (-0.102, -0.040),
    2: (0.040, 0.102),
    3: (0.182, 0.244),
}

# The first six rows cover all four source slots and both outer destination
# directions.  The final two retain cream-cheese sentinels because it has the
# largest frozen source sampler radius and was the first v4 reach failure.
FULL_SENTINELS = (
    (24, 2, 0, "left"),
    (26, 2, 0, "right"),
    (24, 3, 1, "left"),
    (26, 0, 2, "right"),
    (24, 1, 3, "left"),
    (26, 1, 3, "right"),
    (35, 1, 0, "right"),
    (33, 0, 3, "left"),
)
SMOKE_SENTINELS = FULL_SENTINELS[-2:]

REACH_FIELDS = [
    "diagnostic_protocol",
    "candidate",
    "candidate_protocol_version",
    "task_id",
    "layout_id",
    "expected_source_slot",
    "actual_source_slot",
    "object",
    "spatial_region",
    "seed",
    "finger",
    "height_delta_m",
    "target_xyz_json",
    "destination_xy_json",
    "behind_pad_xyz_json",
    "approach_standoff_m",
    "home_high_reached",
    "align_y_reached",
    "align_x_reached",
    "descend_reached",
    "home_high_final_error_m",
    "align_y_final_error_m",
    "align_x_final_error_m",
    "descend_final_error_m",
    "max_target_motion_m",
    "finger_target_contact_steps",
    "grasp_proxy_steps",
    "unexpected_object_contact_steps",
    "robot_distractor_contact_steps",
    "nonselected_robot_target_contact_steps",
    "target_table_support_all_steps",
    "active_actions",
    "passed",
    "error",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate and test one LIBERO-36 source-XY "
            "candidate without modifying the official draft-v4 data."
        )
    )
    parser.add_argument(
        "--candidate",
        choices=tuple(CANDIDATE_X_BOUNDS),
        required=True,
        help="Test c0 first; use c1/c2 only after the preceding candidate fails.",
    )
    parser.add_argument(
        "--phase",
        choices=("reach", "reset", "both"),
        default="reach",
        help=(
            "both runs the 144-row reset/contact preflight only when every "
            "requested reach sentinel passes."
        ),
    )
    parser.add_argument(
        "--reach-level",
        choices=("smoke", "sentinel"),
        default="smoke",
        help=(
            "smoke=8 cream-cheese trials; sentinel=32 predeclared "
            "slot/direction trials. Neither is an exhaustive reach proof."
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/libero36_source_x_preflight"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def compact_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def candidate_ranges(candidate: str) -> dict[int, tuple[float, ...]]:
    x1, x2 = CANDIDATE_X_BOUNDS[candidate]
    return {
        slot: (x1, y1, x2, y2)
        for slot, (y1, y2) in FROZEN_Y_BOUNDS.items()
    }


def candidate_protocol(candidate: str) -> str:
    return f"libero_36_source_xy_diagnostic_{candidate}_v1"


def require_temporary_work_dir(path: Path) -> Path:
    resolved = path.resolve()
    temporary_root = Path(tempfile.gettempdir()).resolve()
    if not resolved.is_relative_to(temporary_root):
        raise ValueError(
            "Diagnostic work directory must remain under the system "
            f"temporary directory {temporary_root}: {resolved}"
        )
    if resolved == temporary_root:
        raise ValueError("Refusing to use the temporary root itself")
    return resolved


def verify_base_modules():
    protocols = {
        "generator": benchmark_generator.PROTOCOL_VERSION,
        "reset_validator": reset_validator.PROTOCOL_VERSION,
        "push_calibrator": push_calibrator.PROTOCOL_VERSION,
    }
    changed = {
        name: value
        for name, value in protocols.items()
        if value != EXPECTED_BASE_PROTOCOL
    }
    if changed:
        raise RuntimeError(
            "This diagnostic was frozen against draft-v4; base protocol "
            f"changed: {changed}"
        )


def patch_candidate_modules(candidate: str):
    ranges = candidate_ranges(candidate)
    protocol = candidate_protocol(candidate)
    benchmark_generator.PROTOCOL_VERSION = protocol
    benchmark_generator.SOURCE_SLOT_RANGES = ranges
    reset_validator.PROTOCOL_VERSION = protocol
    reset_validator.SOURCE_SLOT_RANGES = ranges
    push_calibrator.PROTOCOL_VERSION = protocol
    # calibrate_libero_36_push imported this same module, but assigning it
    # explicitly makes the dependency unambiguous if import routing changes.
    push_calibrator.reset_validator = reset_validator
    return ranges, protocol


def generate_candidate(output_dir: Path, overwrite: bool):
    if output_dir.exists() and not overwrite:
        # A smoke reach, full reach, and reset/contact preflight are often
        # run as separate commands.  Reuse the exact generated candidate so
        # those phases share BDDL hashes instead of silently regenerating it.
        reset_validator.read_layout_spec(output_dir / "layout_spec.csv")
        print("Reusing validated candidate benchmark:", output_dir)
        return
    original_parse_args = benchmark_generator.parse_args
    benchmark_generator.parse_args = lambda: SimpleNamespace(
        output_dir=output_dir,
        overwrite=overwrite,
    )
    try:
        benchmark_generator.main()
    finally:
        benchmark_generator.parse_args = original_parse_args


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=REACH_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def stage_value(stages: dict, name: str, field: str, default):
    value = stages.get(name)
    if value is None:
        return default
    return value[field]


def run_reach_trial(
    row: dict,
    expected_slot: int,
    finger: str,
    height_delta: float,
    candidate: str,
    protocol: str,
) -> dict:
    target = push_calibrator.target_instance(row)
    seed = reset_validator.seed_for(
        row,
        push_calibrator.GATE3_CALIBRATION_RESET_INDEX,
        reset_validator.FULL_SEED_BASE,
    )
    actual_slot = int(row["mapping"][row["object"]])
    result = {
        "diagnostic_protocol": DIAGNOSTIC_PROTOCOL,
        "candidate": candidate,
        "candidate_protocol_version": protocol,
        "task_id": row["task_id"],
        "layout_id": row["layout_id"],
        "expected_source_slot": expected_slot,
        "actual_source_slot": actual_slot,
        "object": row["object"],
        "spatial_region": row["spatial_region"],
        "seed": seed,
        "finger": finger,
        "height_delta_m": height_delta,
        "passed": False,
        "error": "",
    }
    if actual_slot != expected_slot:
        result["error"] = (
            f"sentinel mapping changed: {actual_slot} != {expected_slot}"
        )
        return {field: result.get(field, "") for field in REACH_FIELDS}

    env = None
    recorder = None
    settled = None
    stages = {}
    try:
        env = reset_validator.make_environment(Path(row["bddl_path"]))
        settled = push_calibrator.settle_environment(env, row, seed)
        obs = settled["obs"]
        geometry = push_calibrator.contact_geometry(env, target)
        target_start = push_calibrator.target_xyz(obs, target)
        bounds = reset_validator.TARGET_REGION_RANGES[
            row["spatial_region"]
        ]
        destination = np.array(
            [
                (bounds[0] + bounds[2]) / 2,
                (bounds[1] + bounds[3]) / 2,
            ],
            dtype=np.float64,
        )
        displacement = destination - target_start[:2]
        distance = float(np.linalg.norm(displacement))
        if distance <= 0.1:
            raise RuntimeError("Push path is unexpectedly short")
        direction = displacement / distance
        base_height = push_calibrator.BASE_PAD_HEIGHT_OFFSET_M[
            row["object"]
        ]
        pad_z = target_start[2] + base_height + height_delta
        standoff = (
            reset_validator.SOURCE_SAMPLER_RADII_M[row["object"]]
            + push_calibrator.PRECONTACT_PAD_CLEARANCE_M
        )
        behind_pad = np.array(
            [
                target_start[0] - direction[0] * standoff,
                target_start[1] - direction[1] * standoff,
                pad_z,
            ],
            dtype=np.float64,
        )
        transit_z = pad_z + push_calibrator.TRANSIT_PAD_CLEARANCE_M
        initial_pad = push_calibrator.finger_positions(
            env.env, geometry
        )[finger]
        waypoints = (
            (
                "home_high",
                np.array([initial_pad[0], initial_pad[1], transit_z]),
                0.40,
            ),
            (
                "align_y",
                np.array([initial_pad[0], behind_pad[1], transit_z]),
                0.50,
            ),
            (
                "align_x",
                np.array([behind_pad[0], behind_pad[1], transit_z]),
                0.50,
            ),
            ("descend", behind_pad, 0.25),
        )
        recorder = push_calibrator.AttemptRecorder(
            env,
            row,
            target,
            geometry,
            finger,
            settled["stabilized_z"],
        )
        for phase, waypoint, position_cap in waypoints:
            max_steps = (
                ALIGN_X_MAX_STEPS
                if phase == "align_x"
                else push_calibrator.MAX_TRANSIT_STEPS
            )
            obs, reached, diagnostics = push_calibrator.move_pad_to(
                recorder,
                obs,
                geometry,
                finger,
                waypoint,
                settled["desired_rotation"],
                f"reach_{phase}",
                max_steps,
                position_cap,
            )
            stages[phase] = {**diagnostics, "reached": reached}
            if not reached:
                break

        arrays = recorder.arrays()
        if arrays["target_xyz"].size:
            max_target_motion = float(
                np.linalg.norm(
                    arrays["target_xyz"] - target_start,
                    axis=1,
                ).max()
            )
        else:
            max_target_motion = 0.0
        finger_contacts = int(
            arrays[f"{finger}_contact"].sum()
        )
        checks = {
            "all_waypoints_reached": all(
                stages.get(name, {}).get("reached", False)
                for name in ("home_high", "align_y", "align_x", "descend")
            ),
            "no_target_contact": bool(
                not arrays["left_contact"].any()
                and not arrays["right_contact"].any()
                and not arrays["grasp_proxy"].any()
            ),
            "no_unexpected_object_contact": bool(
                not arrays["unexpected_object_contact"].any()
            ),
            "no_robot_distractor_contact": bool(
                not arrays["robot_distractor_contact"].any()
            ),
            "no_nonselected_target_contact": bool(
                not arrays["nonselected_robot_target_contact"].any()
            ),
            "target_remained_stationary": (
                max_target_motion
                <= reset_validator.FULL_MAX_FINAL_WINDOW_MOTION_M
            ),
            "target_kept_table_support": bool(
                arrays["table_support"].size
                and arrays["table_support"].all()
            ),
        }
        result.update(
            {
                "target_xyz_json": compact_json(target_start.tolist()),
                "destination_xy_json": compact_json(destination.tolist()),
                "behind_pad_xyz_json": compact_json(behind_pad.tolist()),
                "approach_standoff_m": standoff,
                "home_high_reached": stage_value(
                    stages, "home_high", "reached", False
                ),
                "align_y_reached": stage_value(
                    stages, "align_y", "reached", False
                ),
                "align_x_reached": stage_value(
                    stages, "align_x", "reached", False
                ),
                "descend_reached": stage_value(
                    stages, "descend", "reached", False
                ),
                "home_high_final_error_m": stage_value(
                    stages, "home_high", "final_position_error_m", math.nan
                ),
                "align_y_final_error_m": stage_value(
                    stages, "align_y", "final_position_error_m", math.nan
                ),
                "align_x_final_error_m": stage_value(
                    stages, "align_x", "final_position_error_m", math.nan
                ),
                "descend_final_error_m": stage_value(
                    stages, "descend", "final_position_error_m", math.nan
                ),
                "max_target_motion_m": max_target_motion,
                "finger_target_contact_steps": finger_contacts,
                "grasp_proxy_steps": int(arrays["grasp_proxy"].sum()),
                "unexpected_object_contact_steps": int(
                    arrays["unexpected_object_contact"].sum()
                ),
                "robot_distractor_contact_steps": int(
                    arrays["robot_distractor_contact"].sum()
                ),
                "nonselected_robot_target_contact_steps": int(
                    arrays["nonselected_robot_target_contact"].sum()
                ),
                "target_table_support_all_steps": bool(
                    arrays["table_support"].size
                    and arrays["table_support"].all()
                ),
                "active_actions": len(arrays["actions"]),
                "passed": all(checks.values()),
                "error": "" if all(checks.values()) else compact_json(
                    [name for name, passed in checks.items() if not passed]
                ),
            }
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        if recorder is not None:
            result["active_actions"] = len(recorder.actions)
    finally:
        if env is not None:
            env.close()
    return {field: result.get(field, "") for field in REACH_FIELDS}


def run_reach_preflight(
    layout_spec: Path,
    output: Path,
    candidate: str,
    protocol: str,
    level: str,
) -> bool:
    all_rows = reset_validator.read_layout_spec(layout_spec)
    by_key = {
        (row["task_id"], row["layout_id"]): row for row in all_rows
    }
    sentinels = (
        SMOKE_SENTINELS if level == "smoke" else FULL_SENTINELS
    )
    rows = []
    expected = len(sentinels) * 2 * 2
    index = 0
    for task_id, layout_id, expected_slot, expected_region in sentinels:
        row = by_key[(task_id, layout_id)]
        if row["spatial_region"] != expected_region:
            raise RuntimeError(
                f"Sentinel region changed for {(task_id, layout_id)}"
            )
        for height_delta in (-0.01, 0.01):
            for finger in ("left", "right"):
                index += 1
                result = run_reach_trial(
                    row,
                    expected_slot,
                    finger,
                    height_delta,
                    candidate,
                    protocol,
                )
                rows.append(result)
                write_csv(output, rows)
                print(
                    f"[{index:02d}/{expected:02d}] "
                    f"candidate={candidate} task={task_id:02d} "
                    f"layout={layout_id} slot={expected_slot} "
                    f"finger={finger} height={height_delta:+.3f} "
                    f"passed={result['passed']} error={result['error']}"
                )
    passed = sum(bool(row["passed"]) for row in rows)
    print("reach trials:", len(rows))
    print("reach trials passed:", passed)
    print("reach envelope passed:", passed == len(rows))
    print("reach output:", output)
    return passed == len(rows)


def run_reset_preflight(layout_spec: Path, output: Path) -> bool:
    original_parse_args = reset_validator.parse_args
    reset_validator.parse_args = lambda: SimpleNamespace(
        mode="smoke",
        layout_spec=layout_spec,
        output=output,
        manifest=output.with_suffix(".manifest.json"),
        preview_dir=None,
        task_ids=None,
        layout_ids=None,
        resets_per_layout=1,
        settle_steps=reset_validator.FULL_SETTLE_STEPS,
        seed_base=reset_validator.FULL_SEED_BASE,
        max_reset_attempts=reset_validator.FULL_MAX_RESET_ATTEMPTS,
        max_manipulable_horizontal_drift_m=(
            reset_validator.FULL_MAX_MANIPULABLE_HORIZONTAL_DRIFT_M
        ),
        max_receiver_horizontal_drift_m=(
            reset_validator.FULL_MAX_RECEIVER_HORIZONTAL_DRIFT_M
        ),
        max_orientation_drift_deg=(
            reset_validator.FULL_MAX_ORIENTATION_DRIFT_DEG
        ),
        max_final_window_motion_m=(
            reset_validator.FULL_MAX_FINAL_WINDOW_MOTION_M
        ),
        max_final_window_orientation_deg=(
            reset_validator.FULL_MAX_FINAL_WINDOW_ORIENTATION_DEG
        ),
        check_determinism=True,
        # Complete all 144 rows so a failed candidate retains the full
        # collision/reset distribution needed to design the next candidate.
        fail_fast=False,
    )
    validator_exit = None
    try:
        try:
            reset_validator.main()
        except SystemExit as error:
            validator_exit = error
    finally:
        reset_validator.parse_args = original_parse_args
    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    passed = sum(
        str(row["passed"]).strip().lower() == "true" for row in rows
    )
    complete = len(rows) == 144
    reset_passed = complete and passed == 144
    print(
        "candidate reset/contact preflight:",
        f"{passed}/{len(rows)} passed",
    )
    if validator_exit is not None:
        print("reset validator exit code:", validator_exit.code)
    if not complete:
        raise RuntimeError(
            f"Candidate reset/contact preflight is incomplete: {len(rows)}/144"
        )
    return reset_passed


def main():
    args = parse_args()
    verify_base_modules()
    work_root = require_temporary_work_dir(args.work_dir)
    candidate_root = work_root / args.candidate
    if candidate_root.exists() and args.overwrite:
        shutil.rmtree(candidate_root)
    candidate_root.mkdir(parents=True, exist_ok=True)

    ranges, protocol = patch_candidate_modules(args.candidate)
    benchmark_dir = candidate_root / "benchmark"
    print("=" * 80)
    print("LIBERO-36 source-XY candidate preflight")
    print("diagnostic only: True")
    print("candidate:", args.candidate)
    print("candidate protocol:", protocol)
    print("source ranges:", ranges)
    print("phase:", args.phase)
    print("work root:", candidate_root)
    print(
        "NOTE: official data/libero_36 and all frozen Gate-3/Gate-4 "
        "artifacts are not modified."
    )

    generate_candidate(benchmark_dir, args.overwrite)
    layout_spec = benchmark_dir / "layout_spec.csv"
    reach_passed = None
    reach_output = None
    reset_passed = None
    reset_output = None
    if args.phase in {"reach", "both"}:
        reach_output = (
            candidate_root
            / f"{args.candidate}_reach_{args.reach_level}.csv"
        )
        reach_passed = run_reach_preflight(
            layout_spec,
            reach_output,
            args.candidate,
            protocol,
            args.reach_level,
        )
    if args.phase == "reset" or (
        args.phase == "both" and reach_passed
    ):
        reset_output = (
            candidate_root / f"{args.candidate}_reset_preflight.csv"
        )
        reset_passed = run_reset_preflight(
            layout_spec,
            reset_output,
        )
    elif args.phase == "both":
        print(
            "Skipping reset/contact preflight because the reach envelope "
            "failed. Test the next predeclared candidate."
        )
    artifact_paths = [
        benchmark_dir / "task_spec.csv",
        benchmark_dir / "layout_spec.csv",
    ]
    if reach_output is not None and reach_output.is_file():
        artifact_paths.append(reach_output)
    if reset_output is not None and reset_output.is_file():
        artifact_paths.extend(
            [reset_output, reset_output.with_suffix(".manifest.json")]
        )
    write_json(
        candidate_root / "preflight_manifest.json",
        {
            "diagnostic_protocol": DIAGNOSTIC_PROTOCOL,
            "diagnostic_only": True,
            "candidate": args.candidate,
            "candidate_protocol_version": protocol,
            "candidate_source_slot_ranges": ranges,
            "selection_rule": (
                "source-XY redesign candidate; c1 X bounds with compact "
                "collision-separated Y slots"
            ),
            "phase": args.phase,
            "reach_level": args.reach_level,
            "reach_is_exhaustive": False,
            "reach_passed": reach_passed,
            "reset_preflight_passed": reset_passed,
            "candidate_frozen": False,
            "official_protocol_modified": False,
            "reach_acceptance": {
                "path_order": [
                    "home_high",
                    "align_y",
                    "align_x",
                    "descend_precontact",
                ],
                "position_tolerance_m": (
                    push_calibrator.POSITION_TOLERANCE_M
                ),
                "orientation_tolerance_rad": (
                    push_calibrator.ORIENTATION_TOLERANCE_RAD
                ),
                "stable_pose_steps": push_calibrator.STABLE_POSE_STEPS,
                "max_target_motion_m": (
                    reset_validator.FULL_MAX_FINAL_WINDOW_MOTION_M
                ),
                "target_contact_permitted": False,
                "unexpected_contact_permitted": False,
                "direct_table_support_required": True,
            },
            "source_modules": [
                {
                    "path": str(Path(path).resolve()),
                    "sha256": sha256_file(Path(path).resolve()),
                }
                for path in (
                    __file__,
                    benchmark_generator.__file__,
                    reset_validator.__file__,
                    push_calibrator.__file__,
                )
            ],
            "artifacts": [
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for path in artifact_paths
                if path.is_file()
            ],
        },
    )
    if reach_passed is False or reset_passed is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
