#!/usr/bin/env python3
"""task1_edge_pinch_v2_content_qc_v1.py

Read-only content-level QC for the v2 formal-collection HDF5 output
(task1_edge_pinch_diverse_collection_v2/). Opens every kept .hdf5 file in
h5py read ('r') mode only -- never writes, copies, converts, or deletes any
data file. The ONLY file this script writes is its own QC report JSON,
written to --qc-output-dir (default: the scripts directory, NOT the data
directory), so the data directory itself is never touched.

User-requested checks (2026-09-17), each implemented below:
  1. All 96 HDF5 files open/readable without exception.
  2. Per-file episode/frame count (data/demo_0 attrs num_samples, and each
     dataset's actual leading dimension, cross-checked against each other).
  3. agentview_rgb / eye_in_hand_rgb resolution and dtype.
  4. joint_states(7) / ee_pos(3) / ee_ori(3) / gripper_states(2) shapes.
  5. actions shape, numeric range, and NaN/Inf.
  6. termination_reason == 'success_early_termination'.
  7. env_check_success == True.
  8. Axis-angle continuity: NO explicit per-frame hemisphere-flip flag is
     persisted to the HDF5 by the current recorder (checked directly in
     task1_edge_pinch_diverse_collection_recorder_v1.py's
     EpisodeRecordingBuffer.capture() -- HemisphereContinuityConverter's
     'hemisphere_flipped_this_frame' result is computed but not stored).
     So this check is a DERIVED proxy, not a read of a stored marker:
     consecutive-frame ee_ori deltas are computed and any jump above
     CONTINUITY_JUMP_THRESHOLD_RAD is flagged as a suspected discontinuity
     that slipped past the intra-episode hemisphere-continuity fix. This
     limitation is reported explicitly in the output (see
     'axisangle_continuity_check_is_derived_not_a_stored_flag').
  9. Provenance: controller_script_sha256 / seed / azimuth_deg / standoff_m
     attrs present and internally consistent with the filename.
 10. Each file's on-disk SHA-256 matches the value recorded in manifest.json
     at run time (detects any post-hoc modification).
 11. No blind-eval seed (555101-555105) or historical-comparison seed
     (555001-555005) present, re-checked at the content level (not just
     trusting the collection-time gate).

Run (GPU, read-only against the real data):
  python3 task1_edge_pinch_v2_content_qc_v1.py \
      --data-dir /root/smolvla-eval-prep/results/controller/task1_edge_pinch_diverse_collection_v2/ \
      --manifest /root/smolvla-eval-prep/results/controller/task1_edge_pinch_diverse_collection_v2/manifest.json \
      --qc-output-dir /root/smolvla-eval-prep/scripts/

Exit code 0 always (this is a report generator, not a gate) unless a
hard I/O error prevents even opening the manifest / data dir, in which case
it exits 1. All per-file problems are recorded in the JSON report's
'anomalies' list instead of raising, so one bad file does not abort QC of
the other 95.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

BLOCKED_BLIND_EVAL_SEEDS = frozenset(range(555101, 555106))
FLAGGED_HISTORICAL_COMPARISON_SEEDS = frozenset(range(555001, 555006))

REQUIRED_OBS_DATASETS = {
    "agentview_rgb": None,      # resolution checked, not a fixed shape
    "eye_in_hand_rgb": None,
    "ee_pos": 3,
    "ee_ori": 3,
    "gripper_states": 2,
    "joint_states": 7,
}
ACTIONS_LAST_DIM = 7

CONTINUITY_JUMP_THRESHOLD_RAD = 1.0  # matches the negative-control threshold
# used in test_quat_axisangle_convert_v1.py's
# test_hemisphere_continuity_across_synthetic_sequence (raw_jumps > 1.0 rad
# is treated there as "a real, large discontinuity").


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def qc_one_file(path: str, manifest_sha256_by_path: dict) -> dict:
    """Returns a per-file QC record. Never raises -- any exception during
    inspection is captured into record['fatal_error'] so the caller can
    continue with the remaining files."""
    record: dict = {"path": path, "filename": os.path.basename(path), "problems": []}
    try:
        import h5py  # local import: only needed on the GPU execution path

        # -- 10. on-disk SHA-256 vs manifest --------------------------------
        on_disk_sha256 = sha256_of_file(path)
        record["sha256_on_disk"] = on_disk_sha256
        expected_sha256 = manifest_sha256_by_path.get(os.path.abspath(path))
        record["sha256_expected_from_manifest"] = expected_sha256
        if expected_sha256 is None:
            record["problems"].append("not_found_in_manifest_kept_files")
        elif expected_sha256 != on_disk_sha256:
            record["problems"].append("sha256_mismatch_vs_manifest")

        with h5py.File(path, "r") as f:
            # -- 1. openable is implicit (we're inside the with-block) -----
            if "data" not in f or "demo_0" not in f["data"]:
                record["problems"].append("missing_data/demo_0_group")
                record["fatal_error"] = None
                return record
            demo = f["data"]["demo_0"]
            attrs = dict(demo.attrs)
            record["attrs"] = {
                k: (v.item() if isinstance(v, np.generic) else v) for k, v in attrs.items()
            }

            # -- 9. provenance attrs present ---------------------------------
            for required_attr in (
                "seed", "azimuth_deg", "standoff_m", "controller_script_sha256",
                "termination_reason", "env_check_success", "num_samples", "steps_run",
            ):
                if required_attr not in attrs:
                    record["problems"].append(f"missing_attr:{required_attr}")

            seed = attrs.get("seed")
            azimuth_deg = attrs.get("azimuth_deg")
            standoff_m = attrs.get("standoff_m")

            # -- 11. blind-eval / historical-comparison seed re-check -------
            if seed is not None:
                if int(seed) in BLOCKED_BLIND_EVAL_SEEDS:
                    record["problems"].append("CONTAINS_BLOCKED_BLIND_EVAL_SEED")
                if int(seed) in FLAGGED_HISTORICAL_COMPARISON_SEEDS:
                    record["problems"].append("CONTAINS_FLAGGED_HISTORICAL_COMPARISON_SEED")

            # -- filename vs attrs consistency -------------------------------
            expected_filename = f"task1_edge_pinch_seed{seed}_az{int(azimuth_deg)}_so{standoff_m}.hdf5"
            if os.path.basename(path) != expected_filename:
                record["problems"].append(
                    f"filename_mismatch: expected {expected_filename}, got {os.path.basename(path)}"
                )

            # -- 6/7. termination_reason / env_check_success ----------------
            if attrs.get("termination_reason") != "success_early_termination":
                record["problems"].append(
                    f"termination_reason_not_success_early_termination: {attrs.get('termination_reason')}"
                )
            if attrs.get("env_check_success") is not True and not bool(attrs.get("env_check_success", False)):
                record["problems"].append(f"env_check_success_not_true: {attrs.get('env_check_success')}")

            # -- datasets present ---------------------------------------------
            if "obs" not in demo:
                record["problems"].append("missing_obs_group")
                return record
            obs = demo["obs"]
            for key in REQUIRED_OBS_DATASETS:
                if key not in obs:
                    record["problems"].append(f"missing_obs_dataset:{key}")
            if "actions" not in demo:
                record["problems"].append("missing_actions_dataset")
            if "dones" not in demo:
                record["problems"].append("missing_dones_dataset")

            # bail out of shape/value checks if the core datasets are absent
            if record["problems"] and any(
                p.startswith("missing_obs_dataset") or p == "missing_actions_dataset"
                for p in record["problems"]
            ):
                return record

            # -- 2. frame counts, cross-checked -------------------------------
            n_frames_by_dataset = {}
            for key in REQUIRED_OBS_DATASETS:
                n_frames_by_dataset[f"obs/{key}"] = obs[key].shape[0]
            n_frames_by_dataset["actions"] = demo["actions"].shape[0]
            n_frames_by_dataset["dones"] = demo["dones"].shape[0]
            record["n_frames_by_dataset"] = n_frames_by_dataset
            distinct_counts = set(n_frames_by_dataset.values())
            if len(distinct_counts) != 1:
                record["problems"].append(f"inconsistent_frame_counts_across_datasets: {n_frames_by_dataset}")
            num_frames = next(iter(distinct_counts)) if len(distinct_counts) == 1 else None
            record["num_frames"] = num_frames

            steps_run = attrs.get("steps_run")
            num_samples_attr = attrs.get("num_samples")
            if num_frames is not None and steps_run is not None:
                if num_frames != int(steps_run) + 1:
                    record["problems"].append(
                        f"num_frames_not_steps_run_plus_1: num_frames={num_frames}, steps_run={steps_run}"
                    )
            if num_samples_attr is not None and num_frames is not None and int(num_samples_attr) != num_frames:
                record["problems"].append(
                    f"num_samples_attr_mismatch: attr={num_samples_attr}, actual={num_frames}"
                )

            # -- 3. image resolution / dtype -----------------------------------
            for img_key in ("agentview_rgb", "eye_in_hand_rgb"):
                ds = obs[img_key]
                record[f"{img_key}_shape"] = list(ds.shape)
                record[f"{img_key}_dtype"] = str(ds.dtype)
                if ds.ndim != 4 or ds.shape[-1] != 3:
                    record["problems"].append(f"{img_key}_unexpected_shape: {ds.shape}")
                if ds.dtype != np.uint8:
                    record["problems"].append(f"{img_key}_unexpected_dtype: {ds.dtype} (expected uint8)")

            # -- 4. state field shapes -------------------------------------------
            for key, expected_last_dim in REQUIRED_OBS_DATASETS.items():
                if expected_last_dim is None:
                    continue
                ds = obs[key]
                if ds.ndim != 2 or ds.shape[-1] != expected_last_dim:
                    record["problems"].append(f"{key}_unexpected_shape: {ds.shape} (expected (T, {expected_last_dim}))")

            # -- 5. actions shape / range / NaN-Inf ------------------------------
            actions = demo["actions"][()]
            record["actions_shape"] = list(actions.shape)
            record["actions_dtype"] = str(actions.dtype)
            if actions.ndim != 2 or actions.shape[-1] != ACTIONS_LAST_DIM:
                record["problems"].append(f"actions_unexpected_shape: {actions.shape} (expected (T, {ACTIONS_LAST_DIM}))")
            if actions.size:
                record["actions_min"] = float(np.min(actions))
                record["actions_max"] = float(np.max(actions))
                n_nan = int(np.isnan(actions).sum())
                n_inf = int(np.isinf(actions).sum())
                if n_nan:
                    record["problems"].append(f"actions_contains_nan: {n_nan} values")
                if n_inf:
                    record["problems"].append(f"actions_contains_inf: {n_inf} values")

            # -- 8. ee_ori continuity (derived proxy check) ----------------------
            ee_ori = obs["ee_ori"][()]
            if ee_ori.shape[0] >= 2:
                deltas = np.linalg.norm(np.diff(ee_ori, axis=0), axis=1)
                max_jump = float(np.max(deltas))
                record["ee_ori_max_consecutive_frame_delta_rad"] = max_jump
                if max_jump > CONTINUITY_JUMP_THRESHOLD_RAD:
                    record["problems"].append(
                        f"suspected_hemisphere_discontinuity: max consecutive-frame ee_ori "
                        f"delta={max_jump:.4f} rad exceeds threshold={CONTINUITY_JUMP_THRESHOLD_RAD} rad"
                    )
            n_nan_ori = int(np.isnan(ee_ori).sum())
            if n_nan_ori:
                record["problems"].append(f"ee_ori_contains_nan: {n_nan_ori} values")

    except Exception as exc:  # noqa: BLE001 -- one bad file must not abort the whole QC run
        record["fatal_error"] = f"{type(exc).__name__}: {exc}"
        record["problems"].append("fatal_error_during_inspection")

    return record


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir", type=str,
        default="/root/smolvla-eval-prep/results/controller/task1_edge_pinch_diverse_collection_v2/",
    )
    p.add_argument(
        "--manifest", type=str,
        default="/root/smolvla-eval-prep/results/controller/task1_edge_pinch_diverse_collection_v2/manifest.json",
    )
    p.add_argument("--qc-output-dir", type=str, default="/root/smolvla-eval-prep/scripts/")
    args = p.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"REFUSING: data dir does not exist: {args.data_dir}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.manifest):
        print(f"REFUSING: manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    with open(args.manifest) as f:
        manifest = json.load(f)

    manifest_sha256_by_path = {
        os.path.abspath(entry["path"]): entry["sha256"] for entry in manifest.get("kept_files", [])
    }
    manifest_seeds = sorted({entry["seed"] for entry in manifest.get("kept_files", [])})

    hdf5_files = sorted(
        os.path.join(args.data_dir, fn) for fn in os.listdir(args.data_dir) if fn.endswith(".hdf5")
    )

    per_file_records = [qc_one_file(path, manifest_sha256_by_path) for path in hdf5_files]

    anomalies = [
        {"filename": r["filename"], "problems": r["problems"]}
        for r in per_file_records
        if r.get("problems") or r.get("fatal_error")
    ]

    resolutions_seen = sorted({
        tuple(r["agentview_rgb_shape"][1:3]) for r in per_file_records if "agentview_rgb_shape" in r
    })

    report = {
        "qc_script_version": "task1_edge_pinch_v2_content_qc_v1",
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_dir": args.data_dir,
        "manifest_path": args.manifest,
        "num_hdf5_files_found_on_disk": len(hdf5_files),
        "num_kept_files_in_manifest": len(manifest.get("kept_files", [])),
        "num_files_qc_inspected": len(per_file_records),
        "num_files_with_zero_problems": sum(1 for r in per_file_records if not r.get("problems")),
        "num_files_with_problems": len(anomalies),
        "num_files_with_fatal_error": sum(1 for r in per_file_records if r.get("fatal_error")),
        "all_seeds_in_manifest": manifest_seeds,
        "any_blocked_blind_eval_seed_present": any(s in BLOCKED_BLIND_EVAL_SEEDS for s in manifest_seeds),
        "any_flagged_historical_comparison_seed_present": any(
            s in FLAGGED_HISTORICAL_COMPARISON_SEEDS for s in manifest_seeds
        ),
        "agentview_rgb_resolutions_seen": [list(r) for r in resolutions_seen],
        "axisangle_continuity_check_is_derived_not_a_stored_flag": (
            "The HDF5 files do not persist a per-frame hemisphere-flip marker "
            "(HemisphereContinuityConverter.convert()'s 'hemisphere_flipped_this_frame' "
            "result is computed at capture time but not written to obs/ee_ori or any "
            "other dataset). This QC instead checks consecutive-frame ee_ori deltas "
            f"against a {CONTINUITY_JUMP_THRESHOLD_RAD} rad threshold as a proxy for "
            "discontinuity; see per-file 'ee_ori_max_consecutive_frame_delta_rad'."
        ),
        "anomalies": anomalies,
        "per_file_records": per_file_records,
    }

    os.makedirs(args.qc_output_dir, exist_ok=True)
    out_path = os.path.join(
        args.qc_output_dir,
        f"task1_edge_pinch_v2_content_qc_report_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.json",
    )
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"QC_REPORT_WRITTEN {out_path}")
    print(
        f"QC_SUMMARY files_found={report['num_hdf5_files_found_on_disk']} "
        f"files_clean={report['num_files_with_zero_problems']} "
        f"files_with_problems={report['num_files_with_problems']} "
        f"files_fatal_error={report['num_files_with_fatal_error']} "
        f"blind_eval_seed_present={report['any_blocked_blind_eval_seed_present']} "
        f"historical_seed_present={report['any_flagged_historical_comparison_seed_present']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
