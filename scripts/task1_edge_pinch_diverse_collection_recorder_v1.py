#!/usr/bin/env python3
"""task1_edge_pinch_diverse_collection_recorder_v1.py

Formal per-frame HDF5 recorder for the Task1 edge-pinch diverse-collection
effort. Wraps task1_edge_pinch_controller_v1.py's run_episode_edge_pinch()
via its additive obs_capture_callback hook (2026-09-17) to capture, per
step: agentview_rgb, eye_in_hand_rgb, ee_pos, ee_ori (axis-angle, converted
from robot0_eef_quat with intra-episode hemisphere continuity), gripper_
states, joint_states, actions, dones, rewards -- matching the EXISTING
frozen pilot HDF5's obs/* schema (verified read-only, see
claude/task1_pilot_hdf5_state_schema_correction_and_axisangle_evidence_
20260917.json), so the output is directly compatible with the same
downstream HDF5->LeRobot converter used for the 160-episode dataset.

Design reference: claude/task1_edge_pinch_recording_script_design_v1_
20260917.json. User-confirmed decisions (2026-09-17):
  1. State schema: joint_pos(7) + ee_pos(3) + ee_axisangle(3) + gripper(2)
     = 15-dim (Option A). The raw 84-dim MuJoCo `states` is NOT used for
     training; optionally retained as a separate debug/replay field only.
  2. Quaternion ordering is verified via --self-test (offline known-
     rotation checks; an optional --self-test-live single-episode check is
     supported but requires a live GPU env and is NOT run by this
     delivery), never assumed.
  3. Hemisphere continuity is applied WITHIN a single episode only;
     env.reset() at the start of each new episode implicitly starts a new,
     independent continuity window (see main loop: a fresh
     HemisphereContinuityConverter is created per episode).
  4. Output directory: /root/smolvla-eval-prep/results/controller/
     task1_edge_pinch_diverse_collection_v1/ (raw HDF5 + per-episode QC +
     provenance; kept SEPARATE from the training-data directory -- copying
     into the training data directory is a later, separately-confirmed
     step, never done automatically by this script).
  5. Only episodes whose (seed, azimuth, standoff) triple is present in
     the confirmed keep-list AND whose ACTUAL run this time reports
     termination_reason == 'success_early_termination' AND
     env_check_success is True are written as a kept episode. Every run
     (kept or not) is still recorded in the per-batch QC/provenance JSON,
     but a non-kept run's HDF5 group is either omitted or marked
     passed=False (see --keep-failed-episodes), never silently dropped
     from the record.

STATUS: CODE ONLY, per explicit user instruction (2026-09-17): "创建录制
脚本...暂不部署GPU、不采集正式数据、不转换、不训练". This script has NOT
been deployed to GPU, NOT executed against any live simulator, and has
produced NO HDF5 output. py_compile and offline unit tests (via
test_quat_axisangle_convert_v1.py and this project's existing
test_edge_pinch_controller_v1.py) are the only verification performed so
far.

CRITICAL LABELING CONSTRAINT (unchanged from the controller): all output
is stamped controller_source="privileged_simulator_pose",
generation_only=true, not_a_model_evaluation_input=true. Fold02 gate
status and Task1's own strict closed-loop SmolVLA evaluation are entirely
unaffected by anything in this file.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from quat_axisangle_convert_v1 import HemisphereContinuityConverter  # noqa: E402

DEFAULT_EDGE_CONTROLLER_SCRIPT = (
    "/root/smolvla-training-prep/scripts/task1_edge_pinch_controller_v1.py"
)
DEFAULT_BDDL_PATH = (
    "/root/smolvla-training-prep/data/libero_36/bddl/"
    "task_001_layout_1_akita_black_bowl_put_on_top_middle.bddl"
)
DEFAULT_TARGET_OBJECT = "akita_black_bowl_1"

# User-confirmed (2026-09-17): raw recordings + per-episode QC/provenance
# live here, SEPARATE from the training-data directory. Copying into the
# training-data directory is a later, separately-confirmed step.
DEFAULT_OUTPUT_DIR = (
    "/root/smolvla-eval-prep/results/controller/task1_edge_pinch_diverse_collection_v1"
)

RECORDING_SCRIPT_VERSION = "task1_edge_pinch_diverse_collection_recorder_v1"


# --------------------------------------------------------------------------
# Formal collection entry-point safety gate (2026-09-17, user-confirmed
# design: "正式采集入口的安全门禁"). This section defines the CONFIRMED
# 96-combination allowed set (from claude/task1_edge_pinch_
# newseed_prevalidation_950006_950010_results_20260917.json and claude/
# task1_edge_pinch_phase2_bulk_950011_950025_results_20260917.json), the
# blocked blind-eval seed range, and the gate functions main() uses BEFORE
# ever creating an env or writing a file. None of this runs any simulation
# by itself -- it is pure, offline-testable validation logic.
# --------------------------------------------------------------------------

# seed 950001-950025 (25 seeds: 5 original + 20 new, Phase1 950006-950010 +
# Phase2 950011-950025), azimuth in {90, 45} degrees, standoff in
# {0.03, 0.05} m -- 100 candidate combinations total.
COLLECTION_SEED_RANGE = range(950001, 950026)
COLLECTION_AZIMUTHS_DEG = (90.0, 45.0)
COLLECTION_STANDOFFS_M = (0.03, 0.05)

# The 4 combinations EXCLUDED from the 100 candidates (all at azimuth=45,
# standoff=0.03 -- 3 genuine DESCEND-phase stall aborts plus one ambiguous
# normal_completion case excluded per the existing "no forcing to meet
# quota" rule; see the two source docs above for each one's specific
# abort_reason). Combined result: 96/100 kept.
EXCLUDED_COMBINATIONS = frozenset({
    (950009, 45.0, 0.03),
    (950011, 45.0, 0.03),
    (950019, 45.0, 0.03),
    (950024, 45.0, 0.03),
})


def _build_allowed_combinations() -> frozenset:
    allowed = set()
    for seed in COLLECTION_SEED_RANGE:
        for az in COLLECTION_AZIMUTHS_DEG:
            for so in COLLECTION_STANDOFFS_M:
                combo = (seed, az, so)
                if combo not in EXCLUDED_COMBINATIONS:
                    allowed.add(combo)
    return frozenset(allowed)


ALLOWED_COMBINATIONS = _build_allowed_combinations()
assert len(ALLOWED_COMBINATIONS) == 96, (
    f"ALLOWED_COMBINATIONS must contain exactly the confirmed 96 combinations, got "
    f"{len(ALLOWED_COMBINATIONS)} -- this is a hard-coded provenance constant, a count "
    f"mismatch means the source data above was edited incorrectly."
)

# User-confirmed (2026-09-17): "自动拒绝盲评 seed 555101-555105" -- explicit,
# named block, checked BEFORE the general allowed-set membership check (which
# would also reject these, since they are outside COLLECTION_SEED_RANGE, but
# a named check gives a clearer refusal reason and is defense-in-depth
# against COLLECTION_SEED_RANGE ever being widened by mistake).
BLOCKED_BLIND_EVAL_SEEDS = frozenset(range(555101, 555106))

# 555001-555005 were reclassified (2026-09-17,
# claude/task1_edge_pinch_diverse_collection_design_v1_20260917.json) as a
# HISTORICAL COMPARISON set, not a valid blind holdout -- they were already
# used in an earlier "unseen initialization evaluation". Not in
# ALLOWED_COMBINATIONS (outside COLLECTION_SEED_RANGE) so they are already
# rejected by the general check; flagged here only so the refusal reason is
# specific rather than a generic "not in allowed set".
FLAGGED_HISTORICAL_COMPARISON_SEEDS = frozenset(range(555001, 555006))


def validate_keep_list_against_gate(combinations: list) -> dict:
    """Validates a keep-list's entries against the confirmed 96-combination
    allowed set. Returns {"accepted": [...], "rejected": [{"combo", "reason"}]}.
    Each entry must be a dict with 'seed', 'azimuth_deg', 'standoff_m' keys
    (matching load_keep_list's expected structure) or a 3-tuple/list.
    ACCEPTS NOTHING WHEN EMPTY: an empty input list produces an empty
    accepted list, not an error -- callers must separately decide whether
    zero accepted combinations is itself a refusal condition."""
    accepted, rejected = [], []
    for entry in combinations:
        if isinstance(entry, dict):
            seed = entry.get("seed")
            az = entry.get("azimuth_deg")
            so = entry.get("standoff_m")
        else:
            seed, az, so = entry
        try:
            seed_i = int(seed)
            az_f = float(az)
            so_f = float(so)
        except (TypeError, ValueError):
            rejected.append({"combo": entry, "reason": "malformed_entry_not_seed_azimuth_standoff"})
            continue

        if seed_i in BLOCKED_BLIND_EVAL_SEEDS:
            rejected.append({
                "combo": {"seed": seed_i, "azimuth_deg": az_f, "standoff_m": so_f},
                "reason": "blocked_blind_eval_seed_555101_555105",
            })
            continue
        if seed_i in FLAGGED_HISTORICAL_COMPARISON_SEEDS:
            rejected.append({
                "combo": {"seed": seed_i, "azimuth_deg": az_f, "standoff_m": so_f},
                "reason": "flagged_historical_comparison_seed_555001_555005_not_a_valid_collection_seed",
            })
            continue
        if (seed_i, az_f, so_f) not in ALLOWED_COMBINATIONS:
            rejected.append({
                "combo": {"seed": seed_i, "azimuth_deg": az_f, "standoff_m": so_f},
                "reason": "not_in_confirmed_96_combination_allowed_set",
            })
            continue
        accepted.append({"seed": seed_i, "azimuth_deg": az_f, "standoff_m": so_f})
    return {"accepted": accepted, "rejected": rejected}


def check_output_dir_state(output_dir: str, overwrite: bool) -> Optional[str]:
    """Returns None if collection may proceed writing into output_dir, else
    a human-readable refusal reason. A non-existent or empty directory is
    always fine (created if needed). A non-empty existing directory refuses
    UNLESS --overwrite was passed -- and even then, this function never
    deletes anything; --overwrite only means 'permit writing alongside
    existing files', matching the project's standing 'never delete files
    without explicit confirmation' rule."""
    if not os.path.exists(output_dir):
        return None
    if not os.path.isdir(output_dir):
        return f"output_dir_exists_but_is_not_a_directory: {output_dir}"
    existing = os.listdir(output_dir)
    if not existing:
        return None
    if overwrite:
        return None
    return (
        f"output_dir_not_empty ({len(existing)} entries found in {output_dir}) and --overwrite "
        f"was not passed. Pass --overwrite to write alongside existing files (nothing is ever "
        f"deleted automatically), or choose a different --output-dir."
    )


def validate_episode_completeness(buf: "EpisodeRecordingBuffer", result) -> Optional[str]:
    """Per-episode acceptance gate for the formal collection loop (distinct
    from record_one_combination's own 'kept' flag, which only checks the
    controller-level retention rule). Returns None if the episode is
    complete and eligible to be WRITTEN to the formal HDF5, else a
    human-readable reason it must be logged as a failure instead. Checks,
    in the user-confirmed order:
      1. env_check_success == True (result.success is True)
      2. termination_reason == 'success_early_termination'
      3. trajectory length and field completeness: zero missing_key_warnings
         AND buffered frame count equals result.steps_run + 1 exactly (the
         '+1' accounts for the single extra capture at env.reset(), BEFORE
         any real env.step() -- see run_episode_edge_pinch()'s
         obs_capture_callback invocation, invoked once at RESET plus once
         per real step; this invariant is asserted directly by
         test_edge_pinch_controller_v1.py's
         test_run_episode_edge_pinch_obs_capture_callback_invoked_reset_plus_every_step,
         '67 calls = 1 RESET + 66 steps'. A count that is NOT steps_run+1
         means frames were genuinely dropped/duplicated) AND at least 1
         frame was captured.
    """
    if result is None:
        return "no_result_object_exception_during_episode"
    if result.error is not None:
        return f"episode_error: {result.error}"
    if result.aborted:
        return f"episode_aborted: {result.abort_reason}"
    if result.termination_reason != "success_early_termination":
        return f"termination_reason_not_success_early_termination (got: {result.termination_reason})"
    if result.success is not True:
        return f"env_check_success_not_true (got: {result.success})"
    if buf.missing_key_warnings:
        return f"missing_obs_keys_during_capture: {buf.missing_key_warnings}"
    if buf.num_frames() == 0:
        return "zero_frames_captured"
    if buf.num_frames() != result.steps_run + 1:
        return (
            f"frame_count_mismatch: buffered {buf.num_frames()} frames but "
            f"expected result.steps_run+1={result.steps_run + 1} "
            f"(1 RESET capture + {result.steps_run} step captures); got "
            f"result.steps_run={result.steps_run} (frames silently dropped or duplicated)"
        )
    return None


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# --self-test: offline, known-rotation checks (no GPU/live env required)
# --------------------------------------------------------------------------


def _quat_xyzw_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = angle / 2.0
    xyz = axis * np.sin(half)
    return np.array([xyz[0], xyz[1], xyz[2], np.cos(half)])


def run_self_test_offline() -> dict:
    """Known-rotation checks for the (x,y,z,w) ordering assumption --
    NO live environment or GPU required. This is the gate the user
    required before any recording is permitted ("先实现--self-test,用已知
    单位旋转和π附近姿态验证顺序;通过后才允许录制"). It does not by itself
    prove the LIVE environment's robot0_eef_quat uses this convention --
    see run_self_test_live() for that (not executed by this delivery).
    """
    conv = HemisphereContinuityConverter()
    report: dict = {"self_test": "offline_known_rotation_checks", "checks": []}
    all_ok = True

    def check(name, cond, detail):
        nonlocal all_ok
        all_ok = all_ok and bool(cond)
        report["checks"].append({"name": name, "ok": bool(cond), "detail": detail})

    # Identity
    conv.reset()
    r = conv.convert(np.array([0.0, 0.0, 0.0, 1.0]))
    vec = np.asarray(r["axisangle_vec"])
    check("identity_zero_vector", np.allclose(vec, [0, 0, 0], atol=1e-6), vec.tolist())

    # 90 deg about Z
    conv.reset()
    q = _quat_xyzw_from_axis_angle([0, 0, 1], np.pi / 2)
    r = conv.convert(q)
    vec = np.asarray(r["axisangle_vec"])
    expected = np.array([0, 0, np.pi / 2])
    check("90deg_about_z", np.allclose(vec, expected, atol=1e-4), {"got": vec.tolist(), "expected": expected.tolist()})

    # 180 deg (pi) about an arbitrary axis -- magnitude must be pi
    conv.reset()
    axis = np.array([0.267, 0.535, 0.802])
    q = _quat_xyzw_from_axis_angle(axis, np.pi)
    r = conv.convert(q)
    vec = np.asarray(r["axisangle_vec"])
    norm = float(np.linalg.norm(vec))
    check("180deg_magnitude_pi", abs(norm - np.pi) < 1e-3, {"norm": norm})

    # Hemisphere continuity within one episode across a synthetic
    # sign-flipping sequence near angle=pi
    conv.reset()
    axis = np.array([0.1, 0.2, 0.9])
    axis = axis / np.linalg.norm(axis)
    angles = np.linspace(np.pi - 0.05, np.pi + 0.05, 11)
    raw = [_quat_xyzw_from_axis_angle(axis, a) for a in angles]
    flipped = [q if i % 2 == 0 else -q for i, q in enumerate(raw)]
    vecs = [np.asarray(conv.convert(q)["axisangle_vec"]) for q in flipped]
    jumps = [float(np.linalg.norm(vecs[i] - vecs[i - 1])) for i in range(1, len(vecs))]
    check("hemisphere_continuity_no_large_jump", max(jumps) < 0.2, {"max_jump": max(jumps)})

    report["source_backend"] = HemisphereContinuityConverter().convert(
        np.array([0.0, 0.0, 0.0, 1.0])
    )["source"]
    report["all_checks_passed"] = all_ok
    report["note"] = (
        "PASSED offline checks only verify the conversion MATH is self-consistent under "
        "the assumed convention; they do NOT by themselves prove robot0_eef_quat on the "
        "live GPU environment actually uses this convention. Run with --self-test-live "
        "(requires a live env, NOT executed by this delivery) before trusting recorded "
        "ee_ori values from an actual episode."
    )
    return report


def run_self_test_live(make_env_fn, edge_ctrl, base_ctrl) -> dict:
    """OPTIONAL live single-episode consistency check: runs ONE episode via
    the controller with obs_capture_callback wired to this script's own
    conversion path, then compares the resulting ee_ori row-norm statistics
    against the existing pilot HDF5's known obs/ee_ori statistics (row
    norms clustered in [3.1416, 3.1938], mean ~3.1767 -- see
    claude/task1_pilot_hdf5_state_schema_correction_and_axisangle_evidence_
    20260917.json). A large, systematic mismatch (e.g. norms clustered near
    0 instead of near pi) would indicate the (x,y,z,w) ordering assumption
    is wrong for this environment and recording must NOT proceed.

    Invoked only via --self-test-live plus a working GPU/libero
    environment (main() wires this up). Uses a single ALREADY-VALIDATED
    combination (seed=950001, azimuth=90deg, standoff=0.05m -- part of the
    confirmed 96/100 keep list, Phase2 bulk results) purely so the episode
    reliably reaches RELEASE and yields many frames; this call still writes
    NOTHING to disk -- it is read-only diagnostics on live obs data,
    exactly like the earlier obs-field and pilot-HDF5-schema probes in this
    project. `edge_ctrl` is the loaded task1_edge_pinch_controller_v1
    module (hosts run_episode_edge_pinch/EdgePinchConfig/
    compute_rim_local_points); `base_ctrl` is the loaded
    task1_privileged_pickplace_controller_v1 module (the `ctrl_module`
    helper argument those functions expect, per the existing project
    convention -- see edge_ctrl.DEFAULT_CONTROLLER_SCRIPT).
    """
    report: dict = {"self_test": "live_single_episode_consistency_check"}
    env = None
    try:
        env = make_env_fn()
        rim_local_points = edge_ctrl.compute_rim_local_points(
            make_env_fn, base_ctrl, DEFAULT_TARGET_OBJECT
        )
        cfg = edge_ctrl.EdgePinchConfig(
            orientation_control_verified=True, edge_azimuth_deg=90.0, standoff_m=0.05,
        )
        conv = HemisphereContinuityConverter()
        ee_ori_vecs = []

        def capture(step_count, phase, obs, action, done, info):
            quat = obs.get("robot0_eef_quat")
            if quat is not None:
                r = conv.convert(np.asarray(quat))
                ee_ori_vecs.append(np.asarray(r["axisangle_vec"]))

        result = edge_ctrl.run_episode_edge_pinch(
            env, cfg, base_ctrl, seed=950001, rim_local_points=rim_local_points,
            obs_capture_callback=capture,
        )
        report["episode_error"] = result.error
        report["episode_aborted"] = result.aborted
        report["episode_termination_reason"] = result.termination_reason
        report["episode_env_check_success"] = result.success
        norms = np.linalg.norm(np.array(ee_ori_vecs), axis=1) if ee_ori_vecs else np.array([])
        report["num_frames"] = len(ee_ori_vecs)
        report["row_norm_min"] = float(norms.min()) if norms.size else None
        report["row_norm_max"] = float(norms.max()) if norms.size else None
        report["row_norm_mean"] = float(norms.mean()) if norms.size else None
        report["expected_from_pilot_hdf5"] = {"min": 3.1416, "max": 3.1938, "mean": 3.1767}
        consistent = bool(norms.size and abs(float(norms.mean()) - 3.1767) < 0.5)
        report["consistent_with_pilot_convention"] = consistent
    except Exception as exc:  # noqa: BLE001 -- diagnostic-only, never crashes the caller
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            env.close()
    return report


# --------------------------------------------------------------------------
# Per-episode recording buffer
# --------------------------------------------------------------------------


class EpisodeRecordingBuffer:
    """Accumulates per-step obs/action/done/reward arrays via the
    controller's obs_capture_callback hook, converting robot0_eef_quat to
    axis-angle (with intra-episode hemisphere continuity) as it goes. One
    instance per episode -- create a fresh one (and a fresh
    HemisphereContinuityConverter) for every new episode; never reused
    across episodes (design decision, part1 open question 2)."""

    REQUIRED_OBS_KEYS = (
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot0_joint_pos",
    )

    def __init__(self):
        self.conv = HemisphereContinuityConverter()
        self.agentview_rgb: list = []
        self.eye_in_hand_rgb: list = []
        self.ee_pos: list = []
        self.ee_ori: list = []
        self.gripper_states: list = []
        self.joint_states: list = []
        self.actions: list = []
        self.dones: list = []
        self.phases: list = []
        self.step_counts: list = []
        self.missing_key_warnings: list = []

    def capture(self, step_count, phase, obs, action, done, info):
        missing = [k for k in self.REQUIRED_OBS_KEYS if k not in obs]
        if missing:
            self.missing_key_warnings.append({"step_count": step_count, "phase": phase, "missing": missing})
            return  # do not append partial/misaligned rows

        self.agentview_rgb.append(np.asarray(obs["agentview_image"]))
        self.eye_in_hand_rgb.append(np.asarray(obs["robot0_eye_in_hand_image"]))
        self.ee_pos.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float64))
        conv_result = self.conv.convert(np.asarray(obs["robot0_eef_quat"]))
        self.ee_ori.append(np.asarray(conv_result["axisangle_vec"], dtype=np.float64))
        self.gripper_states.append(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64))
        self.joint_states.append(np.asarray(obs["robot0_joint_pos"], dtype=np.float64))
        self.actions.append(
            np.zeros(7, dtype=np.float32) if action is None else np.asarray(action, dtype=np.float32)
        )
        self.dones.append(bool(done))
        self.phases.append(phase)
        self.step_counts.append(step_count)

    def num_frames(self) -> int:
        return len(self.step_counts)

    def as_arrays(self) -> dict:
        return {
            "obs/agentview_rgb": np.stack(self.agentview_rgb) if self.agentview_rgb else np.zeros((0,)),
            "obs/eye_in_hand_rgb": np.stack(self.eye_in_hand_rgb) if self.eye_in_hand_rgb else np.zeros((0,)),
            "obs/ee_pos": np.stack(self.ee_pos) if self.ee_pos else np.zeros((0, 3)),
            "obs/ee_ori": np.stack(self.ee_ori) if self.ee_ori else np.zeros((0, 3)),
            "obs/gripper_states": np.stack(self.gripper_states) if self.gripper_states else np.zeros((0, 2)),
            "obs/joint_states": np.stack(self.joint_states) if self.joint_states else np.zeros((0, 7)),
            "actions": np.stack(self.actions) if self.actions else np.zeros((0, 7), dtype=np.float32),
            "dones": np.array(self.dones, dtype=bool),
        }


def write_episode_to_hdf5(h5_group, buffer: EpisodeRecordingBuffer, episode_attrs: dict) -> None:
    """Writes one episode's buffered arrays into an h5py group, matching
    the existing frozen pilot HDF5's obs/* naming convention exactly.
    Callers are responsible for opening/closing the file and choosing the
    group name (demo_i). Not invoked by this delivery (no HDF5 is written
    by this script yet -- see module docstring)."""
    arrays = buffer.as_arrays()
    obs_grp = h5_group.create_group("obs")
    for key, arr in arrays.items():
        if key.startswith("obs/"):
            obs_grp.create_dataset(key.split("/", 1)[1], data=arr)
        else:
            h5_group.create_dataset(key, data=arr)
    for k, v in episode_attrs.items():
        h5_group.attrs[k] = v


def write_failure_log(output_dir: str, failure_records: list) -> str:
    """Writes ONLY a failure log (JSON) -- never an HDF5 group -- for
    episodes that did not pass validate_episode_completeness(). Per
    user-confirmed design: '失败 episode 只写入失败日志，不进入正式
    HDF5'. Returns the written path."""
    path = os.path.join(output_dir, "failure_log.json")
    payload = {
        "recording_script_version": RECORDING_SCRIPT_VERSION,
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_failures": len(failure_records),
        "failures": failure_records,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def write_manifest_and_qc(
    output_dir: str, kept_files: list, failure_records: list, gate_rejected: list,
) -> dict:
    """Writes manifest.json: sha256 of every kept HDF5 file, an aggregate
    QC summary (counts of kept/failed/gate-rejected), and provenance
    stamps. Per user-confirmed design: '采集完成后生成 manifest、SHA-256
    和汇总 QC'. kept_files is a list of {"path", "seed", "azimuth_deg",
    "standoff_m", "num_frames"} dicts for episodes actually written to
    disk. Returns the manifest dict (also written to output_dir/
    manifest.json)."""
    manifest = {
        "recording_script_version": RECORDING_SCRIPT_VERSION,
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "controller_source": "privileged_simulator_pose",
        "generation_only": True,
        "not_a_model_evaluation_input": True,
        "output_dir": output_dir,
        "qc_summary": {
            "num_kept": len(kept_files),
            "num_failed_episodes": len(failure_records),
            "num_gate_rejected_candidates": len(gate_rejected),
        },
        "kept_files": [],
        "failure_log_ref": "failure_log.json" if failure_records else None,
    }
    for entry in kept_files:
        sha = sha256_of_file(entry["path"])
        manifest["kept_files"].append({**entry, "sha256": sha})
    path = os.path.join(output_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


# --------------------------------------------------------------------------
# Keep-list filtering
# --------------------------------------------------------------------------


def load_keep_list(path: str) -> list:
    """Loads the confirmed (seed, azimuth_deg, standoff_m) combinations
    from Phase1/Phase2 validation results (96/100 combined, per
    claude/task1_edge_pinch_phase2_bulk_950011_950025_results_20260917.json
    and the newseed_prevalidation doc). This is a LIST OF CANDIDATES TO
    ATTEMPT recording for -- actual keep/discard for the HDF5 is decided
    per this script's OWN run's termination_reason/env_check_success, never
    by trusting this list's historical smoke-test outcome directly (see
    module docstring part5)."""
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "keep_list" in data:
        return data["keep_list"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unrecognized keep-list JSON structure in {path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true", help="Run offline known-rotation checks and exit.")
    p.add_argument(
        "--self-test-live", action="store_true",
        help="Also run the live single-episode consistency check (requires a working GPU/libero "
        "env; NOT executed by this delivery). Implies --self-test.",
    )
    p.add_argument("--keep-list-json", type=str, default=None, help="Path to the confirmed keep-list JSON.")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--edge-controller-script", type=str, default=DEFAULT_EDGE_CONTROLLER_SCRIPT)
    p.add_argument("--bddl-path", type=str, default=DEFAULT_BDDL_PATH)
    p.add_argument(
        "--dry-run", action="store_true",
        help="Do not write any HDF5 file; print a per-episode summary of what WOULD be written.",
    )
    p.add_argument(
        "--keep-failed-episodes", action="store_true",
        help="DEPRECATED/UNUSED as of the safety-gate design (2026-09-17): failed episodes are "
        "now ALWAYS logged to failure_log.json and NEVER written to the HDF5, per the "
        "user-confirmed retention rule. This flag is accepted but ignored, kept only so an "
        "old invocation does not fail with an unrecognized-argument error.",
    )
    p.add_argument(
        "--confirm-collection", action="store_true",
        help="Required to actually run the formal recording loop (create an env, step episodes, "
        "write HDF5/manifest/failure-log files). Without this flag, main() validates the "
        "keep-list against the gate and then refuses to proceed (exit code 3) -- the same "
        "safe-default behavior as before this flag existed.",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Permit writing into a non-empty --output-dir. Never deletes existing files -- "
        "only relaxes the 'directory must be empty' precondition.",
    )
    return p


def _load_controller_module(controller_script_path: str):
    """Identical importlib pattern to the controller's own
    _load_controller_module / to task1_privileged_pickplace_controller_v1.py
    reuse elsewhere in this project (spec_from_file_location +
    sys.modules["edge_ctrl"]=m before exec_module, required for dataclass
    introspection)."""
    spec = importlib.util.spec_from_file_location("edge_ctrl", controller_script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load controller module from {controller_script_path}")
    m = importlib.util.module_from_spec(spec)
    sys.modules["edge_ctrl"] = m
    spec.loader.exec_module(m)
    return m


def _make_env_fn(bddl_path: str):
    def _make():
        from libero.libero.envs import OffScreenRenderEnv  # noqa: WPS433 -- GPU-only import

        return OffScreenRenderEnv(
            bddl_file_name=bddl_path, camera_heights=128, camera_widths=128,
            horizon=1000, use_camera_obs=True,
        )
    return _make


def record_one_combination(
    edge_ctrl, base_ctrl, env, cfg_kwargs: dict, seed: int, azimuth_deg: float,
    standoff_m: float, rim_local_points: dict,
) -> dict:
    """Runs exactly one episode for one (seed, azimuth, standoff) via
    run_episode_edge_pinch(), buffering per-step data through
    obs_capture_callback. Returns a dict with the buffer, the
    EdgePinchEpisodeResult, and the per-run QC summary -- never writes
    anything to disk itself (the caller decides keep/discard and dry-run
    vs write). Any exception from run_episode_edge_pinch (e.g. the existing
    RuntimeError paths for environment_failure_done_true, stall/timeout
    aborts propagate as exceptions per the controller's own design) is
    caught here and reported as a failed run, never silently swallowed."""
    cfg = edge_ctrl.EdgePinchConfig(edge_azimuth_deg=azimuth_deg, standoff_m=standoff_m, **cfg_kwargs)
    buf = EpisodeRecordingBuffer()
    run_started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        result = edge_ctrl.run_episode_edge_pinch(
            env, cfg, base_ctrl, seed=seed, rim_local_points=rim_local_points,
            obs_capture_callback=buf.capture,
        )
        # Retention rule (design part3 / user-confirmed, 2026-09-17):
        # kept ONLY if this run's OWN result says success_early_termination
        # AND env_check_success (result.success) is True -- never inferred
        # from the historical Phase1/Phase2 smoke-test outcome for the same
        # (seed, azimuth, standoff), since those runs recorded no data.
        kept = (
            not result.aborted
            and result.error is None
            and result.termination_reason == "success_early_termination"
            and result.success is True
        )
        return {
            "seed": seed, "azimuth_deg": azimuth_deg, "standoff_m": standoff_m,
            "buffer": buf, "result": result, "kept": bool(kept), "error": None,
            "run_started_utc": run_started_utc, "num_frames": buf.num_frames(),
            "missing_key_warnings": buf.missing_key_warnings,
        }
    except Exception as exc:  # noqa: BLE001 -- a failed run is data, not a crash of the batch
        return {
            "seed": seed, "azimuth_deg": azimuth_deg, "standoff_m": standoff_m,
            "buffer": buf, "result": None, "kept": False,
            "error": f"{type(exc).__name__}: {exc}",
            "run_started_utc": run_started_utc, "num_frames": buf.num_frames(),
            "missing_key_warnings": buf.missing_key_warnings,
        }


def main(argv: Optional[list] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.self_test or args.self_test_live:
        report = run_self_test_offline()
        print(json.dumps(report, indent=2))
        if not report["all_checks_passed"]:
            print("SELF_TEST_OFFLINE_FAILED", file=sys.stderr)
            return 1
        print("SELF_TEST_OFFLINE_OK")

        if args.self_test_live:
            # 2026-09-17 (user-confirmed): actually run the live single-
            # episode consistency check on GPU. Still writes NOTHING to
            # disk -- read-only diagnostics only, same class of operation
            # as the earlier obs-field / pilot-HDF5-schema probes.
            try:
                edge_ctrl = _load_controller_module(args.edge_controller_script)
                base_ctrl = edge_ctrl._load_controller_module(edge_ctrl.DEFAULT_CONTROLLER_SCRIPT)
                live_report = run_self_test_live(
                    _make_env_fn(args.bddl_path), edge_ctrl, base_ctrl
                )
            except Exception as exc:  # noqa: BLE001 -- report, never crash on a diagnostic
                live_report = {"error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps(live_report, indent=2))
            if live_report.get("error"):
                print("SELF_TEST_LIVE_FAILED", file=sys.stderr)
                return 1
            if not live_report.get("consistent_with_pilot_convention"):
                print(
                    "SELF_TEST_LIVE_INCONSISTENT: row-norm statistics do not match the pilot "
                    "HDF5's obs/ee_ori convention -- DO NOT proceed to recording; the (x,y,z,w) "
                    "ordering assumption needs review before this script may be used further.",
                    file=sys.stderr,
                )
                return 1
            print("SELF_TEST_LIVE_OK")
        return 0

    if not args.keep_list_json:
        print("ERROR: --keep-list-json is required for recording (or use --self-test).", file=sys.stderr)
        return 1

    combinations = load_keep_list(args.keep_list_json)
    print(
        f"Loaded {len(combinations)} candidate combinations from {args.keep_list_json}. "
        f"output_dir={args.output_dir} dry_run={args.dry_run} confirm_collection={args.confirm_collection}"
    )

    # GATE 1: keep-list entries must be a subset of the confirmed 96-
    # combination allowed set; blind-eval seeds 555101-555105 (and the
    # flagged 555001-555005 historical-comparison seeds) are explicitly
    # rejected. This runs BEFORE any env is created, regardless of
    # --confirm-collection.
    gate = validate_keep_list_against_gate(combinations)
    print(json.dumps({"gate_validation": gate}, indent=2, default=str))
    if gate["rejected"]:
        print(
            f"REFUSING TO EXECUTE: {len(gate['rejected'])} of {len(combinations)} keep-list "
            f"entries were rejected by the collection safety gate (see gate_validation.rejected "
            f"above for reasons). Remove or correct them and retry -- NOTHING has been written.",
            file=sys.stderr,
        )
        return 4
    if not gate["accepted"]:
        print("REFUSING TO EXECUTE: keep-list is empty after gate validation. Nothing to record.", file=sys.stderr)
        return 4

    # GATE 2: explicit user confirmation required to proceed past this
    # point. Without it, the gate validation above still ran (so a dry
    # keep-list check is possible without --confirm-collection), but no
    # env is created and nothing is written -- same safe-default posture
    # as before this gate design existed.
    if not args.confirm_collection:
        print(
            "REFUSING TO EXECUTE: keep-list passed the safety gate "
            f"({len(gate['accepted'])} combination(s) accepted), but --confirm-collection was not "
            "passed. This is the required explicit confirmation to create an env and run the "
            "formal recording loop. Per standing project instruction, GPU deployment/testing of "
            "this gate itself does NOT constitute that confirmation -- it must be given "
            "separately for an actual collection run.",
            file=sys.stderr,
        )
        return 3

    # GATE 3: output directory must be empty, or --overwrite must be passed.
    # Never deletes anything.
    dir_reason = check_output_dir_state(args.output_dir, args.overwrite)
    if dir_reason is not None:
        print(f"REFUSING TO EXECUTE: {dir_reason}", file=sys.stderr)
        return 5

    if args.dry_run:
        print(
            f"DRY RUN: would attempt to record {len(gate['accepted'])} accepted combination(s) "
            f"into {args.output_dir}. No env created, no files written."
        )
        return 0

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        edge_ctrl = _load_controller_module(args.edge_controller_script)
        base_ctrl = edge_ctrl._load_controller_module(edge_ctrl.DEFAULT_CONTROLLER_SCRIPT)
    except Exception as exc:  # noqa: BLE001
        print(f"REFUSING TO EXECUTE: could not load controller module(s): {type(exc).__name__}: {exc}", file=sys.stderr)
        return 6

    import h5py  # noqa: E402, WPS433 -- deferred: only needed on this GPU-execution path

    make_env_fn = _make_env_fn(args.bddl_path)
    env = make_env_fn()
    kept_files: list = []
    failure_records: list = []
    try:
        rim_local_points = edge_ctrl.compute_rim_local_points(make_env_fn, base_ctrl, DEFAULT_TARGET_OBJECT)
        for combo in gate["accepted"]:
            seed, azimuth_deg, standoff_m = combo["seed"], combo["azimuth_deg"], combo["standoff_m"]
            run_out = record_one_combination(
                edge_ctrl, base_ctrl, env, {"orientation_control_verified": True},
                seed, azimuth_deg, standoff_m, rim_local_points,
            )
            buf, result = run_out["buffer"], run_out["result"]
            fail_reason = validate_episode_completeness(buf, result)
            if fail_reason is not None:
                failure_records.append({
                    "seed": seed, "azimuth_deg": azimuth_deg, "standoff_m": standoff_m,
                    "reason": fail_reason, "run_error": run_out["error"],
                    "run_started_utc": run_out["run_started_utc"], "num_frames": run_out["num_frames"],
                })
                continue

            episode_attrs = {
                "seed": seed, "azimuth_deg": azimuth_deg, "standoff_m": standoff_m,
                "num_samples": buf.num_frames(),
                "termination_reason": result.termination_reason,
                "env_check_success": bool(result.success),
                "grasp_object_offset_m": json.dumps(result.grasp_object_offset_m),
                "steps_run": result.steps_run,
                "controller_source": "privileged_simulator_pose",
                "generation_only": True,
                "not_a_model_evaluation_input": True,
                "controller_script_sha256": sha256_of_file(args.edge_controller_script),
                "recording_script_version": RECORDING_SCRIPT_VERSION,
                "recorded_at_utc": run_out["run_started_utc"],
            }
            episode_filename = f"task1_edge_pinch_seed{seed}_az{int(azimuth_deg)}_so{standoff_m}.hdf5"
            episode_path = os.path.join(args.output_dir, episode_filename)
            with h5py.File(episode_path, "w") as f:
                data_grp = f.create_group("data")
                demo_grp = data_grp.create_group("demo_0")
                write_episode_to_hdf5(demo_grp, buf, episode_attrs)
            kept_files.append({
                "path": episode_path, "seed": seed, "azimuth_deg": azimuth_deg,
                "standoff_m": standoff_m, "num_frames": buf.num_frames(),
            })
            print(f"WROTE {episode_path} ({buf.num_frames()} frames)")
    finally:
        env.close()

    failure_log_path = write_failure_log(args.output_dir, failure_records) if failure_records else None
    manifest = write_manifest_and_qc(args.output_dir, kept_files, failure_records, gate["rejected"])
    print(json.dumps({"manifest_summary": manifest["qc_summary"], "failure_log_path": failure_log_path}, indent=2))
    print(f"COLLECTION_RUN_COMPLETE kept={len(kept_files)} failed={len(failure_records)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
