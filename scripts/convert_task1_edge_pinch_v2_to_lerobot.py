#!/usr/bin/env python3
"""convert_task1_edge_pinch_v2_to_lerobot.py

Dedicated HDF5 -> LeRobotDataset converter for the Task1 edge-pinch v2
collection (task1_edge_pinch_diverse_collection_v2/, 96 files, one demo_0
per file). NOT a wrapper around the legacy scripts/convert_libero_hdf5_to_
lerobot.py -- that script assumes ONE HDF5 containing many demo_N groups
plus a top-level data.attrs['problem_info'] for the language instruction,
neither of which matches v2's one-file-per-episode structure (see
claude/task1_edge_pinch_v2_to_lerobot_conversion_design_20260917.json for
the full read-only audit this script implements). Instead this script:
  - reuses the SAME features dict (image shapes, state/action dims and
    names) as the legacy converter, so the output schema is drop-in
    compatible with the existing LeRobot reading pipeline;
  - reuses equivalent per-file field/shape validation (the v2 HDF5 field
    names -- obs/agentview_rgb, obs/eye_in_hand_rgb, obs/joint_states,
    obs/ee_pos, obs/ee_ori, obs/gripper_states, actions -- are IDENTICAL
    to what the legacy validate_demo() checks, so this transfers almost
    verbatim);
  - injects a constant language instruction (see LANGUAGE_INSTRUCTION
    below -- verified 2026-09-17 against the ACTUAL frozen 160-episode
    training set's Task1 manifest.json, which is lowercase; this does
    NOT match the BDDL file's capitalized ':language' field, and the
    lowercase, manifest-sourced string is the one used here on purpose);
  - replaces the legacy script's hardcoded 'Expected 50 episodes'/
    'Expected 5068 frames' assertion (a task0-pilot-specific magic number,
    meaningless for this dataset) with a self-consistent check against
    THIS run's own v2 manifest.json-recorded frame counts for exactly the
    files being converted -- no exception-catching workaround needed.

Safety gates (modeled on task1_edge_pinch_diverse_collection_recorder_v1.py's
formal-collection gate):
  GATE A: exactly one of --input-files or (--all-96 AND
          --confirm-full-conversion) must be given. Converting "all 96"
          requires the explicit confirmation flag; a bare file list does
          not (that's the smoke-test path).
  GATE B: every input file's on-disk SHA-256 is cross-checked against the
          v2 manifest.json's kept_files entries BEFORE conversion --
          refuses to convert any file not found there or with a mismatched
          hash (never trust an unverified/possibly-modified file).
  GATE C: --output-dir must not exist, or --overwrite must be passed
          (never silently deletes).
  --dry-run: runs gates A/B/C and all validation, creates no
             LeRobotDataset, writes no files.

Pure, h5py/lerobot-free functions (build_state_vector, validate_v2_demo_
fields, parse_v2_filename, create_features_dict, compute_expected_frame_
total) are unit-tested offline in test_convert_task1_edge_pinch_v2_to_
lerobot.py using a fake h5py module, the same pattern used for
task1_edge_pinch_v2_content_qc_v1.py.

Run (GPU, smoke test -- 2 files only):
  python3 convert_task1_edge_pinch_v2_to_lerobot.py \
      --input-dir /root/smolvla-eval-prep/results/controller/task1_edge_pinch_diverse_collection_v2/ \
      --manifest /root/smolvla-eval-prep/results/controller/task1_edge_pinch_diverse_collection_v2/manifest.json \
      --input-files task1_edge_pinch_seed950001_az90_so0.03.hdf5,task1_edge_pinch_seed950025_az45_so0.05.hdf5 \
      --output-dir /root/smolvla-eval-prep/results/lerobot/_smoke_task1_edge_pinch_v2/ \
      --repo-id local/_smoke_task1_edge_pinch_v2 \
      --fps 20

Converting all 96 (SEPARATE, later, explicit confirmation) additionally
requires --all-96 --confirm-full-conversion instead of --input-files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

# Verified 2026-09-17 against
# /root/smolvla-training-prep/data/demos/libero36_task1_5_success_pilot_v1/
# manifest.json's "language_instruction" field (the ACTUAL frozen
# 160-episode training set's Task1 instruction) -- lowercase, NOT the
# BDDL file's capitalized ':language' text. Do not "correct" the casing.
LANGUAGE_INSTRUCTION = "pick up the akita black bowl and place it on the plate in the middle region"

# Copied verbatim from scripts/convert_libero_hdf5_to_lerobot.py so the
# output schema matches the existing LeRobot reading pipeline exactly.
STATE_NAMES = [
    "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",
    "ee_x", "ee_y", "ee_z",
    "ee_ori_0", "ee_ori_1", "ee_ori_2",
    "gripper_0", "gripper_1",
]
ACTION_NAMES = [
    "delta_x", "delta_y", "delta_z",
    "delta_rot_x", "delta_rot_y", "delta_rot_z",
    "gripper",
]

REQUIRED_V2_FIELDS = [
    "actions",
    "obs/agentview_rgb",
    "obs/eye_in_hand_rgb",
    "obs/joint_states",
    "obs/ee_pos",
    "obs/ee_ori",
    "obs/gripper_states",
]

FILENAME_RE = re.compile(
    r"^task1_edge_pinch_seed(?P<seed>\d+)_az(?P<az>-?\d+)_so(?P<so>[0-9.]+)\.hdf5$"
)

BLOCKED_BLIND_EVAL_SEEDS = frozenset(range(555101, 555106))
FLAGGED_HISTORICAL_COMPARISON_SEEDS = frozenset(range(555001, 555006))


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_v2_filename(filename: str) -> dict:
    """Pure parser, no filesystem access. Raises ValueError on an
    unrecognized filename -- never guesses seed/azimuth/standoff."""
    m = FILENAME_RE.match(filename)
    if not m:
        raise ValueError(f"filename does not match expected v2 pattern: {filename}")
    return {
        "seed": int(m.group("seed")),
        "azimuth_deg": float(m.group("az")),
        "standoff_m": float(m.group("so")),
    }


def create_features_dict() -> dict:
    """Identical to the legacy converter's create_dataset()'s features
    dict (image shapes/dtype, state/action dims and names) so LeRobot
    readers built against the existing 160-episode dataset work unchanged
    against this one."""
    return {
        "observation.images.agentview": {
            "dtype": "image", "shape": (128, 128, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "image", "shape": (128, 128, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {"dtype": "float32", "shape": (15,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
    }


def build_state_vector(joint, ee_pos, ee_ori, gripper) -> np.ndarray:
    """Pure numpy assembly, order-matched to STATE_NAMES:
    joint(7) + ee_pos(3) + ee_ori(3) + gripper(2) = 15. Raises ValueError
    if the assembled vector is not exactly (15,) -- mirrors the legacy
    converter's per-frame shape check."""
    state = np.concatenate(
        [
            np.asarray(joint, dtype=np.float32),
            np.asarray(ee_pos, dtype=np.float32),
            np.asarray(ee_ori, dtype=np.float32),
            np.asarray(gripper, dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    if state.shape != (15,):
        raise ValueError(f"assembled state shape is {state.shape}, expected (15,)")
    return state


def validate_v2_demo_fields(demo, filename: str) -> int:
    """Adapted from the legacy converter's validate_demo(), but reads a
    SINGLE v2 demo_0 group (not one of many demo_N groups). Field names
    are identical to the legacy converter's expectations -- only the
    calling convention (one file = one demo) differs. Returns the
    validated frame count. Raises KeyError/ValueError, mirroring the
    legacy converter's error style, on any problem."""
    for key in REQUIRED_V2_FIELDS:
        if key not in demo:
            raise KeyError(f"{filename}: missing field {key}")

    expected_length = demo["actions"].shape[0]
    for key in REQUIRED_V2_FIELDS[1:]:
        actual_length = demo[key].shape[0]
        if actual_length != expected_length:
            raise ValueError(
                f"{filename}: {key} has {actual_length} frames, expected {expected_length}"
            )

    if demo["actions"].shape[1:] != (7,):
        raise ValueError(f"{filename}: expected action shape (*, 7), got {demo['actions'].shape}")
    if demo["obs/agentview_rgb"].shape[1:] != (128, 128, 3):
        raise ValueError(
            f"{filename}: unexpected agentview shape {demo['obs/agentview_rgb'].shape}"
        )
    if demo["obs/eye_in_hand_rgb"].shape[1:] != (128, 128, 3):
        raise ValueError(
            f"{filename}: unexpected wrist image shape {demo['obs/eye_in_hand_rgb'].shape}"
        )
    return expected_length


def compute_expected_frame_total(manifest: dict, filenames: list) -> int:
    """Sums num_frames from the v2 manifest.json's kept_files entries for
    EXACTLY the given filenames (a subset for smoke tests, or all 96 for
    a full run) -- this is the self-consistent replacement for the legacy
    converter's hardcoded 5068-frame assertion. Raises KeyError if any
    requested filename is not present in the manifest (never silently
    skips an unaccounted-for file)."""
    by_filename = {
        os.path.basename(entry["path"]): entry["num_frames"]
        for entry in manifest.get("kept_files", [])
    }
    total = 0
    missing = []
    for fn in filenames:
        if fn not in by_filename:
            missing.append(fn)
            continue
        total += by_filename[fn]
    if missing:
        raise KeyError(f"filenames not found in manifest.json kept_files: {missing}")
    return total


def check_seed_not_blocked(seed: int) -> None:
    """Re-checks, at conversion time, that no blind-eval or historical-
    comparison seed is being converted -- defense in depth on top of the
    collection-time gate and the content-QC re-check."""
    if seed in BLOCKED_BLIND_EVAL_SEEDS:
        raise ValueError(f"seed {seed} is a BLOCKED blind-eval seed (555101-555105); refusing")
    if seed in FLAGGED_HISTORICAL_COMPARISON_SEEDS:
        raise ValueError(
            f"seed {seed} is a FLAGGED historical-comparison seed (555001-555005); refusing"
        )


def build_provenance_record(filename: str, parsed: dict, h5_attrs: dict, source_sha256: str,
                             lerobot_episode_index: int, num_frames: int) -> dict:
    """One row of the episode-level provenance manifest this script
    writes alongside the LeRobot dataset (mirrors data/manifests/README.md's
    convention: LeRobot episode_index <-> source HDF5 <-> seed/azimuth/
    standoff <-> sha256)."""
    return {
        "lerobot_episode_index": lerobot_episode_index,
        "source_filename": filename,
        "source_sha256": source_sha256,
        "seed": parsed["seed"],
        "azimuth_deg": parsed["azimuth_deg"],
        "standoff_m": parsed["standoff_m"],
        "num_frames": num_frames,
        "termination_reason": h5_attrs.get("termination_reason"),
        "env_check_success": bool(h5_attrs.get("env_check_success")),
        "controller_script_sha256": h5_attrs.get("controller_script_sha256"),
        "controller_source": h5_attrs.get("controller_source"),
        "generation_only": bool(h5_attrs.get("generation_only")),
        "not_a_model_evaluation_input": bool(h5_attrs.get("not_a_model_evaluation_input")),
        "language_instruction": LANGUAGE_INSTRUCTION,
    }


def resolve_input_files(args, manifest: dict) -> list:
    """GATE A: exactly one of --input-files or (--all-96 AND
    --confirm-full-conversion). Returns the resolved list of filenames
    (not full paths)."""
    if args.input_files and args.all_96:
        raise SystemExit("REFUSING: pass either --input-files or --all-96, not both.")
    if args.all_96:
        if not args.confirm_full_conversion:
            raise SystemExit(
                "REFUSING: --all-96 requires --confirm-full-conversion as a separate explicit "
                "flag. Converting all 96 episodes is a distinct, larger action from a smoke "
                "test and must be confirmed on its own."
            )
        filenames = sorted(
            fn for fn in os.listdir(args.input_dir) if fn.endswith(".hdf5")
        )
        if len(filenames) != 96:
            raise SystemExit(
                f"REFUSING: --all-96 expects exactly 96 .hdf5 files in --input-dir, found "
                f"{len(filenames)}. Not proceeding with an unexpected file count."
            )
        return filenames
    if not args.input_files:
        raise SystemExit("REFUSING: must pass either --input-files (smoke test) or --all-96 --confirm-full-conversion.")
    return [fn.strip() for fn in args.input_files.split(",") if fn.strip()]


def gate_b_verify_sha256(input_dir: str, filenames: list, manifest: dict) -> dict:
    """GATE B: cross-checks each file's on-disk SHA-256 against
    manifest.json's kept_files entries. Returns {filename: source_sha256}
    for files that pass; raises SystemExit listing any that don't."""
    by_filename = {
        os.path.basename(entry["path"]): entry["sha256"] for entry in manifest.get("kept_files", [])
    }
    verified = {}
    problems = []
    for fn in filenames:
        path = os.path.join(input_dir, fn)
        if not os.path.isfile(path):
            problems.append(f"{fn}: file not found at {path}")
            continue
        expected = by_filename.get(fn)
        if expected is None:
            problems.append(f"{fn}: not present in manifest.json kept_files")
            continue
        actual = sha256_of_file(path)
        if actual != expected:
            problems.append(f"{fn}: sha256 mismatch (expected {expected}, got {actual})")
            continue
        verified[fn] = actual
    if problems:
        raise SystemExit("REFUSING: GATE B sha256 verification failed:\n" + "\n".join(problems))
    return verified


def gate_c_check_output_dir(output_dir: str, overwrite: bool) -> None:
    """GATE C: output dir must not exist, or --overwrite required. Never
    deletes anything itself -- that is LeRobotDataset.create()'s/shutil's
    job downstream, gated by the same --overwrite flag the caller passed
    explicitly."""
    if os.path.exists(output_dir) and os.listdir(output_dir) and not overwrite:
        raise SystemExit(
            f"REFUSING: output dir {output_dir} already exists and is non-empty. "
            "Pass --overwrite to rebuild it (this will delete its current contents)."
        )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--manifest", required=True, help="v2 manifest.json path")
    p.add_argument("--input-files", default=None, help="comma-separated filenames (smoke-test path)")
    p.add_argument("--all-96", action="store_true", help="convert every .hdf5 in --input-dir")
    p.add_argument("--confirm-full-conversion", action="store_true")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.isfile(args.manifest):
        print(f"REFUSING: manifest not found: {args.manifest}", file=sys.stderr)
        return 1
    with open(args.manifest) as f:
        manifest = json.load(f)

    try:
        filenames = resolve_input_files(args, manifest)
        parsed_by_filename = {}
        for fn in filenames:
            parsed = parse_v2_filename(fn)
            check_seed_not_blocked(parsed["seed"])
            parsed_by_filename[fn] = parsed

        verified_sha256 = gate_b_verify_sha256(args.input_dir, filenames, manifest)
        expected_frame_total = compute_expected_frame_total(manifest, filenames)
        gate_c_check_output_dir(args.output_dir, args.overwrite)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (ValueError, KeyError) as exc:
        print(f"REFUSING: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(
        f"Resolved {len(filenames)} file(s) to convert. "
        f"Expected total frames (from manifest): {expected_frame_total}. "
        f"Language instruction: {LANGUAGE_INSTRUCTION!r}"
    )

    if args.dry_run:
        print("DRY RUN: gates A/B/C passed. No LeRobotDataset created, no files written.")
        return 0

    import h5py  # noqa: E402, WPS433 -- deferred: only needed on this GPU-execution path
    import shutil
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402, WPS433

    if os.path.exists(args.output_dir) and args.overwrite:
        shutil.rmtree(args.output_dir)
    os.makedirs(os.path.dirname(args.output_dir.rstrip("/")) or ".", exist_ok=True)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=create_features_dict(),
        root=args.output_dir,
        robot_type="panda",
        use_videos=False,
        image_writer_threads=4,
    )

    provenance_rows = []
    total_frames = 0

    for episode_index, fn in enumerate(filenames):
        parsed = parsed_by_filename[fn]
        path = os.path.join(args.input_dir, fn)
        with h5py.File(path, "r") as f:
            demo = f["data"]["demo_0"]
            h5_attrs = dict(demo.attrs)
            num_frames = validate_v2_demo_fields(demo, fn)

            for frame_index in range(num_frames):
                joint = np.asarray(demo["obs/joint_states"][frame_index], dtype=np.float32)
                ee_pos = np.asarray(demo["obs/ee_pos"][frame_index], dtype=np.float32)
                ee_ori = np.asarray(demo["obs/ee_ori"][frame_index], dtype=np.float32)
                gripper = np.asarray(demo["obs/gripper_states"][frame_index], dtype=np.float32)
                state = build_state_vector(joint, ee_pos, ee_ori, gripper)

                frame = {
                    "observation.images.agentview": np.asarray(
                        demo["obs/agentview_rgb"][frame_index], dtype=np.uint8
                    ),
                    "observation.images.wrist": np.asarray(
                        demo["obs/eye_in_hand_rgb"][frame_index], dtype=np.uint8
                    ),
                    "observation.state": state,
                    "action": np.asarray(demo["actions"][frame_index], dtype=np.float32),
                    "task": LANGUAGE_INSTRUCTION,
                }
                dataset.add_frame(frame)
                total_frames += 1

        dataset.save_episode(parallel_encoding=False)
        provenance_rows.append(
            build_provenance_record(
                fn, parsed, h5_attrs, verified_sha256[fn], episode_index, num_frames
            )
        )
        print(f"CONVERTED episode {episode_index + 1:03d}/{len(filenames)} ({fn}, {num_frames} frames)")

    if hasattr(dataset, "finalize"):
        dataset.finalize()

    if len(filenames) != len(provenance_rows):
        raise RuntimeError(
            f"INTERNAL: episode count mismatch after conversion: "
            f"{len(filenames)} requested vs {len(provenance_rows)} written"
        )
    if total_frames != expected_frame_total:
        raise RuntimeError(
            f"FRAME COUNT MISMATCH: converted {total_frames} frames but manifest.json "
            f"recorded {expected_frame_total} for these {len(filenames)} file(s). "
            "Refusing to treat this conversion as complete -- investigate before using this "
            "output."
        )

    provenance_path = os.path.join(args.output_dir, "episode_provenance.csv")
    with open(provenance_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(provenance_rows[0].keys()))
        writer.writeheader()
        writer.writerows(provenance_rows)

    qc_summary = {
        "converter_version": "convert_task1_edge_pinch_v2_to_lerobot",
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "repo_id": args.repo_id,
        "language_instruction": LANGUAGE_INSTRUCTION,
        "num_episodes_converted": len(filenames),
        "total_frames": total_frames,
        "expected_frame_total_from_manifest": expected_frame_total,
        "frame_count_matches_manifest": total_frames == expected_frame_total,
        "provenance_csv": provenance_path,
        "generation_only": True,
        "not_a_model_evaluation_input": True,
    }
    qc_path = os.path.join(args.output_dir, "conversion_qc.json")
    with open(qc_path, "w") as f:
        json.dump(qc_summary, f, indent=2)

    print(json.dumps(qc_summary, indent=2))
    print(f"CONVERSION_COMPLETE episodes={len(filenames)} frames={total_frames}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
