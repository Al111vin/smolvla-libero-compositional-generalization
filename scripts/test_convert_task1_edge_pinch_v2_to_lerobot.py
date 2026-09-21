"""test_convert_task1_edge_pinch_v2_to_lerobot.py

Offline logic tests for convert_task1_edge_pinch_v2_to_lerobot.py's pure
functions and safety gates. No h5py, no lerobot, no GPU required for the
functions tested here -- they are h5py/lerobot-free by design (the module
under test defers those imports to inside main()). Covers:
  - filename parsing (seed/azimuth/standoff)
  - 15-dim state vector assembly order and shape validation
  - per-demo field/shape validation (v2 field names)
  - expected-frame-total computation against a fake manifest.json
  - blind-eval / historical-comparison seed blocking
  - GATE A (--input-files vs --all-96/--confirm-full-conversion exclusivity)
  - GATE B (sha256 cross-check against manifest.json)
  - GATE C (output dir must not exist or --overwrite required)

Run: python3 test_convert_task1_edge_pinch_v2_to_lerobot.py
Expects: ALL_CONVERTER_UNIT_TESTS_OK printed at the end.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile

import numpy as np

import convert_task1_edge_pinch_v2_to_lerobot as conv


# ---------------------------------------------------------------------------
# filename parsing
# ---------------------------------------------------------------------------
def test_parse_v2_filename_valid():
    parsed = conv.parse_v2_filename("task1_edge_pinch_seed950001_az90_so0.03.hdf5")
    assert parsed == {"seed": 950001, "azimuth_deg": 90.0, "standoff_m": 0.03}, parsed


def test_parse_v2_filename_45_degree():
    parsed = conv.parse_v2_filename("task1_edge_pinch_seed950025_az45_so0.05.hdf5")
    assert parsed == {"seed": 950025, "azimuth_deg": 45.0, "standoff_m": 0.05}, parsed


def test_parse_v2_filename_rejects_unrecognized():
    try:
        conv.parse_v2_filename("not_a_v2_file.hdf5")
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# state vector assembly
# ---------------------------------------------------------------------------
def test_build_state_vector_order_and_shape():
    joint = np.arange(7, dtype=np.float32)
    ee_pos = np.array([100.0, 101.0, 102.0], dtype=np.float32)
    ee_ori = np.array([200.0, 201.0, 202.0], dtype=np.float32)
    gripper = np.array([300.0, 301.0], dtype=np.float32)
    state = conv.build_state_vector(joint, ee_pos, ee_ori, gripper)
    assert state.shape == (15,)
    assert state.dtype == np.float32
    expected = np.concatenate([joint, ee_pos, ee_ori, gripper])
    assert np.allclose(state, expected)
    # order sanity: state[7:10] must be ee_pos, not ee_ori
    assert np.allclose(state[7:10], ee_pos)
    assert np.allclose(state[10:13], ee_ori)
    assert np.allclose(state[13:15], gripper)


def test_build_state_vector_wrong_input_size_raises():
    try:
        conv.build_state_vector(np.zeros(6), np.zeros(3), np.zeros(3), np.zeros(2))
        assert False, "expected ValueError for a 14-dim result"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# create_features_dict
# ---------------------------------------------------------------------------
def test_create_features_dict_matches_legacy_schema():
    feats = conv.create_features_dict()
    assert feats["observation.state"]["shape"] == (15,)
    assert feats["observation.state"]["names"] == conv.STATE_NAMES
    assert feats["action"]["shape"] == (7,)
    assert feats["action"]["names"] == conv.ACTION_NAMES
    assert feats["observation.images.agentview"]["shape"] == (128, 128, 3)
    assert feats["observation.images.wrist"]["shape"] == (128, 128, 3)


# ---------------------------------------------------------------------------
# validate_v2_demo_fields (using a plain dict-of-numpy-arrays as the "demo")
# ---------------------------------------------------------------------------
def _make_demo_dict(n_frames=10):
    return {
        "actions": np.zeros((n_frames, 7), dtype=np.float32),
        "obs/agentview_rgb": np.zeros((n_frames, 128, 128, 3), dtype=np.uint8),
        "obs/eye_in_hand_rgb": np.zeros((n_frames, 128, 128, 3), dtype=np.uint8),
        "obs/joint_states": np.zeros((n_frames, 7), dtype=np.float64),
        "obs/ee_pos": np.zeros((n_frames, 3), dtype=np.float64),
        "obs/ee_ori": np.zeros((n_frames, 3), dtype=np.float64),
        "obs/gripper_states": np.zeros((n_frames, 2), dtype=np.float64),
    }


def test_validate_v2_demo_fields_clean():
    demo = _make_demo_dict(10)
    n = conv.validate_v2_demo_fields(demo, "f.hdf5")
    assert n == 10


def test_validate_v2_demo_fields_missing_field_raises():
    demo = _make_demo_dict(10)
    del demo["obs/ee_ori"]
    try:
        conv.validate_v2_demo_fields(demo, "f.hdf5")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_validate_v2_demo_fields_inconsistent_length_raises():
    demo = _make_demo_dict(10)
    demo["obs/ee_pos"] = np.zeros((9, 3))
    try:
        conv.validate_v2_demo_fields(demo, "f.hdf5")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_validate_v2_demo_fields_wrong_image_shape_raises():
    demo = _make_demo_dict(10)
    demo["obs/agentview_rgb"] = np.zeros((10, 64, 64, 3), dtype=np.uint8)
    try:
        conv.validate_v2_demo_fields(demo, "f.hdf5")
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# compute_expected_frame_total
# ---------------------------------------------------------------------------
def test_compute_expected_frame_total_sums_requested_subset_only():
    manifest = {
        "kept_files": [
            {"path": "/x/task1_edge_pinch_seed950001_az90_so0.03.hdf5", "num_frames": 100},
            {"path": "/x/task1_edge_pinch_seed950002_az90_so0.03.hdf5", "num_frames": 105},
            {"path": "/x/task1_edge_pinch_seed950003_az90_so0.03.hdf5", "num_frames": 200},
        ]
    }
    total = conv.compute_expected_frame_total(
        manifest,
        ["task1_edge_pinch_seed950001_az90_so0.03.hdf5", "task1_edge_pinch_seed950003_az90_so0.03.hdf5"],
    )
    assert total == 300, total  # 100 + 200, NOT including seed950002


def test_compute_expected_frame_total_raises_on_missing_filename():
    manifest = {"kept_files": [{"path": "/x/a.hdf5", "num_frames": 100}]}
    try:
        conv.compute_expected_frame_total(manifest, ["a.hdf5", "does_not_exist.hdf5"])
        assert False, "expected KeyError"
    except KeyError:
        pass


# ---------------------------------------------------------------------------
# seed blocking
# ---------------------------------------------------------------------------
def test_check_seed_not_blocked_allows_normal_seed():
    conv.check_seed_not_blocked(950001)  # must not raise


def test_check_seed_not_blocked_rejects_blind_eval_seed():
    try:
        conv.check_seed_not_blocked(555103)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_check_seed_not_blocked_rejects_historical_seed():
    try:
        conv.check_seed_not_blocked(555003)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# GATE A: resolve_input_files
# ---------------------------------------------------------------------------
def _args(**overrides):
    base = dict(
        input_dir="/tmp/nonexistent", manifest="/tmp/nonexistent.json",
        input_files=None, all_96=False, confirm_full_conversion=False,
        output_dir="/tmp/out", repo_id="local/x", fps=20, overwrite=False, dry_run=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_gate_a_input_files_path_ok():
    filenames = conv.resolve_input_files(_args(input_files="a.hdf5, b.hdf5"), manifest={})
    assert filenames == ["a.hdf5", "b.hdf5"], filenames


def test_gate_a_neither_flag_refuses():
    try:
        conv.resolve_input_files(_args(), manifest={})
        assert False, "expected SystemExit"
    except SystemExit:
        pass


def test_gate_a_both_flags_refuses():
    try:
        conv.resolve_input_files(_args(input_files="a.hdf5", all_96=True), manifest={})
        assert False, "expected SystemExit"
    except SystemExit:
        pass


def test_gate_a_all96_without_confirm_refuses():
    try:
        conv.resolve_input_files(_args(all_96=True, confirm_full_conversion=False), manifest={})
        assert False, "expected SystemExit"
    except SystemExit:
        pass


def test_gate_a_all96_with_confirm_but_wrong_file_count_refuses():
    d = tempfile.mkdtemp()
    for i in range(5):  # only 5, not 96
        open(os.path.join(d, f"task1_edge_pinch_seed95000{i}_az90_so0.03.hdf5"), "w").close()
    try:
        conv.resolve_input_files(
            _args(input_dir=d, all_96=True, confirm_full_conversion=True), manifest={}
        )
        assert False, "expected SystemExit"
    except SystemExit:
        pass


# ---------------------------------------------------------------------------
# GATE B: sha256 cross-check
# ---------------------------------------------------------------------------
def test_gate_b_accepts_matching_sha256():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "a.hdf5")
    with open(path, "wb") as f:
        f.write(b"hello")
    expected_sha = hashlib.sha256(b"hello").hexdigest()
    manifest = {"kept_files": [{"path": "/anywhere/a.hdf5", "sha256": expected_sha}]}
    verified = conv.gate_b_verify_sha256(d, ["a.hdf5"], manifest)
    assert verified == {"a.hdf5": expected_sha}


def test_gate_b_rejects_mismatched_sha256():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "a.hdf5")
    with open(path, "wb") as f:
        f.write(b"hello")
    manifest = {"kept_files": [{"path": "/anywhere/a.hdf5", "sha256": "0" * 64}]}
    try:
        conv.gate_b_verify_sha256(d, ["a.hdf5"], manifest)
        assert False, "expected SystemExit"
    except SystemExit:
        pass


def test_gate_b_rejects_file_not_in_manifest():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "a.hdf5")
    with open(path, "wb") as f:
        f.write(b"hello")
    manifest = {"kept_files": []}
    try:
        conv.gate_b_verify_sha256(d, ["a.hdf5"], manifest)
        assert False, "expected SystemExit"
    except SystemExit:
        pass


def test_gate_b_rejects_missing_file_on_disk():
    d = tempfile.mkdtemp()
    manifest = {"kept_files": [{"path": "/anywhere/a.hdf5", "sha256": "0" * 64}]}
    try:
        conv.gate_b_verify_sha256(d, ["a.hdf5"], manifest)  # a.hdf5 never written
        assert False, "expected SystemExit"
    except SystemExit:
        pass


# ---------------------------------------------------------------------------
# GATE C: output dir state
# ---------------------------------------------------------------------------
def test_gate_c_nonexistent_dir_ok():
    conv.gate_c_check_output_dir("/tmp/does/not/exist/at/all", overwrite=False)  # must not raise


def test_gate_c_empty_dir_ok():
    d = tempfile.mkdtemp()
    conv.gate_c_check_output_dir(d, overwrite=False)  # must not raise (empty)


def test_gate_c_nonempty_dir_refused_without_overwrite():
    d = tempfile.mkdtemp()
    open(os.path.join(d, "existing.txt"), "w").close()
    try:
        conv.gate_c_check_output_dir(d, overwrite=False)
        assert False, "expected SystemExit"
    except SystemExit:
        pass


def test_gate_c_nonempty_dir_ok_with_overwrite():
    d = tempfile.mkdtemp()
    open(os.path.join(d, "existing.txt"), "w").close()
    conv.gate_c_check_output_dir(d, overwrite=True)  # must not raise


# ---------------------------------------------------------------------------
# provenance record
# ---------------------------------------------------------------------------
def test_build_provenance_record_fields():
    parsed = {"seed": 950001, "azimuth_deg": 90.0, "standoff_m": 0.03}
    h5_attrs = {
        "termination_reason": "success_early_termination", "env_check_success": True,
        "controller_script_sha256": "deadbeef", "controller_source": "privileged_simulator_pose",
        "generation_only": True, "not_a_model_evaluation_input": True,
    }
    rec = conv.build_provenance_record("f.hdf5", parsed, h5_attrs, "abc123", 0, 100)
    assert rec["lerobot_episode_index"] == 0
    assert rec["seed"] == 950001
    assert rec["language_instruction"] == conv.LANGUAGE_INSTRUCTION
    assert rec["env_check_success"] is True


def test_language_instruction_is_lowercase_and_verified_string():
    # Regression guard: this exact string was verified 2026-09-17 against
    # the frozen 160-episode training set's Task1 manifest.json. Catches
    # an accidental "correction" back to the BDDL's capitalized text.
    assert conv.LANGUAGE_INSTRUCTION == (
        "pick up the akita black bowl and place it on the plate in the middle region"
    )
    assert conv.LANGUAGE_INSTRUCTION[0] == "p"  # lowercase 'p', not 'P'


if __name__ == "__main__":
    tests = [
        test_parse_v2_filename_valid,
        test_parse_v2_filename_45_degree,
        test_parse_v2_filename_rejects_unrecognized,
        test_build_state_vector_order_and_shape,
        test_build_state_vector_wrong_input_size_raises,
        test_create_features_dict_matches_legacy_schema,
        test_validate_v2_demo_fields_clean,
        test_validate_v2_demo_fields_missing_field_raises,
        test_validate_v2_demo_fields_inconsistent_length_raises,
        test_validate_v2_demo_fields_wrong_image_shape_raises,
        test_compute_expected_frame_total_sums_requested_subset_only,
        test_compute_expected_frame_total_raises_on_missing_filename,
        test_check_seed_not_blocked_allows_normal_seed,
        test_check_seed_not_blocked_rejects_blind_eval_seed,
        test_check_seed_not_blocked_rejects_historical_seed,
        test_gate_a_input_files_path_ok,
        test_gate_a_neither_flag_refuses,
        test_gate_a_both_flags_refuses,
        test_gate_a_all96_without_confirm_refuses,
        test_gate_a_all96_with_confirm_but_wrong_file_count_refuses,
        test_gate_b_accepts_matching_sha256,
        test_gate_b_rejects_mismatched_sha256,
        test_gate_b_rejects_file_not_in_manifest,
        test_gate_b_rejects_missing_file_on_disk,
        test_gate_c_nonexistent_dir_ok,
        test_gate_c_empty_dir_ok,
        test_gate_c_nonempty_dir_refused_without_overwrite,
        test_gate_c_nonempty_dir_ok_with_overwrite,
        test_build_provenance_record_fields,
        test_language_instruction_is_lowercase_and_verified_string,
    ]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print("ALL_CONVERTER_UNIT_TESTS_OK")
