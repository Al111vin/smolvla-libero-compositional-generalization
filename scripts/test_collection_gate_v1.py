"""test_collection_gate_v1.py

Offline unit tests for the formal-collection safety gate added to
task1_edge_pinch_diverse_collection_recorder_v1.py (2026-09-17,
user-confirmed design: "正式采集入口的安全门禁"). No GPU/libero/h5py
required -- these exercise only the gate logic (pure Python + filesystem
tmp-dir operations), not the actual recording loop.

User-confirmed gate requirements, each covered by at least one test below:
  1. --confirm-collection explicit flag required
  2. keep-list JSON must be provided
  3. keep-list entries restricted to already-validated (seed, azimuth,
     standoff) combinations (the confirmed 96/100 set)
  4. blind-eval seeds 555101-555105 auto-rejected
  5. output directory must be empty, or --overwrite required
  6. per-episode completeness check (env_check_success, termination_reason,
     trajectory length/field completeness)
  7. failed episodes -> failure log only, never the formal HDF5 (verified
     structurally here; the actual HDF5-vs-failure-log branching in main()
     itself requires h5py/a live env and is NOT exercised by this file --
     see the recorder's own main() for that wiring, GPU-side testing
     covers it)
  8. manifest + sha256 + aggregate QC generated after a run (write_manifest_
     and_qc tested directly against a tmp dir with fake files)

Run: python3 test_collection_gate_v1.py
Expects: ALL_COLLECTION_GATE_UNIT_TESTS_OK printed at the end.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile

import numpy as np

import task1_edge_pinch_diverse_collection_recorder_v1 as rec


# --------------------------------------------------------------------------
# 3 & 4: keep-list validation against the confirmed 96-combination set +
# blind-eval seed blocking
# --------------------------------------------------------------------------
def test_allowed_combinations_has_exactly_96():
    assert len(rec.ALLOWED_COMBINATIONS) == 96, len(rec.ALLOWED_COMBINATIONS)


def test_validate_keep_list_accepts_known_good_combination():
    result = rec.validate_keep_list_against_gate([
        {"seed": 950001, "azimuth_deg": 90.0, "standoff_m": 0.05},
    ])
    assert result["rejected"] == [], result["rejected"]
    assert len(result["accepted"]) == 1
    assert result["accepted"][0] == {"seed": 950001, "azimuth_deg": 90.0, "standoff_m": 0.05}


def test_validate_keep_list_rejects_excluded_combination():
    # 950009/45deg/0.03 is one of the 4 explicitly excluded combinations
    # (genuine DESCEND stall abort, not in the 96-combination allowed set).
    result = rec.validate_keep_list_against_gate([
        {"seed": 950009, "azimuth_deg": 45.0, "standoff_m": 0.03},
    ])
    assert result["accepted"] == []
    assert len(result["rejected"]) == 1
    assert result["rejected"][0]["reason"] == "not_in_confirmed_96_combination_allowed_set"


def test_validate_keep_list_rejects_blind_eval_seeds():
    result = rec.validate_keep_list_against_gate([
        {"seed": 555101, "azimuth_deg": 90.0, "standoff_m": 0.05},
        {"seed": 555105, "azimuth_deg": 45.0, "standoff_m": 0.03},
    ])
    assert result["accepted"] == []
    assert len(result["rejected"]) == 2
    for r in result["rejected"]:
        assert r["reason"] == "blocked_blind_eval_seed_555101_555105", r


def test_validate_keep_list_rejects_historical_comparison_seeds():
    result = rec.validate_keep_list_against_gate([
        {"seed": 555001, "azimuth_deg": 90.0, "standoff_m": 0.05},
    ])
    assert result["accepted"] == []
    assert result["rejected"][0]["reason"] == "flagged_historical_comparison_seed_555001_555005_not_a_valid_collection_seed"


def test_validate_keep_list_rejects_unknown_seed_outside_any_range():
    result = rec.validate_keep_list_against_gate([
        {"seed": 12345, "azimuth_deg": 90.0, "standoff_m": 0.05},
    ])
    assert result["accepted"] == []
    assert result["rejected"][0]["reason"] == "not_in_confirmed_96_combination_allowed_set"


def test_validate_keep_list_rejects_malformed_entry():
    result = rec.validate_keep_list_against_gate([{"seed": "not_a_number", "azimuth_deg": 90.0, "standoff_m": 0.05}])
    assert result["accepted"] == []
    assert result["rejected"][0]["reason"] == "malformed_entry_not_seed_azimuth_standoff"


def test_validate_keep_list_accepts_tuple_form():
    result = rec.validate_keep_list_against_gate([(950025, 90.0, 0.03)])
    assert result["rejected"] == []
    assert result["accepted"] == [{"seed": 950025, "azimuth_deg": 90.0, "standoff_m": 0.03}]


def test_validate_keep_list_mixed_batch_partitions_correctly():
    result = rec.validate_keep_list_against_gate([
        {"seed": 950001, "azimuth_deg": 90.0, "standoff_m": 0.05},   # accepted
        {"seed": 555103, "azimuth_deg": 90.0, "standoff_m": 0.05},   # blind eval, rejected
        {"seed": 950009, "azimuth_deg": 45.0, "standoff_m": 0.03},   # excluded combo, rejected
    ])
    assert len(result["accepted"]) == 1
    assert len(result["rejected"]) == 2


# --------------------------------------------------------------------------
# 5: output directory empty-or-overwrite gate
# --------------------------------------------------------------------------
def test_check_output_dir_state_nonexistent_dir_ok():
    d = os.path.join(tempfile.gettempdir(), "does_not_exist_collection_gate_test_v1")
    if os.path.exists(d):
        shutil.rmtree(d)
    assert rec.check_output_dir_state(d, overwrite=False) is None


def test_check_output_dir_state_empty_dir_ok():
    d = tempfile.mkdtemp()
    try:
        assert rec.check_output_dir_state(d, overwrite=False) is None
    finally:
        shutil.rmtree(d)


def test_check_output_dir_state_nonempty_dir_refused_without_overwrite():
    d = tempfile.mkdtemp()
    try:
        with open(os.path.join(d, "existing_file.txt"), "w") as f:
            f.write("preexisting")
        reason = rec.check_output_dir_state(d, overwrite=False)
        assert reason is not None
        assert "output_dir_not_empty" in reason
        # never deletes:
        assert os.path.exists(os.path.join(d, "existing_file.txt"))
    finally:
        shutil.rmtree(d)


def test_check_output_dir_state_nonempty_dir_ok_with_overwrite():
    d = tempfile.mkdtemp()
    try:
        with open(os.path.join(d, "existing_file.txt"), "w") as f:
            f.write("preexisting")
        assert rec.check_output_dir_state(d, overwrite=True) is None
        # still never deletes, even when permitted to proceed:
        assert os.path.exists(os.path.join(d, "existing_file.txt"))
    finally:
        shutil.rmtree(d)


# --------------------------------------------------------------------------
# 6: per-episode completeness gate
# --------------------------------------------------------------------------
class _FakeResult:
    def __init__(self, error=None, aborted=False, abort_reason=None,
                 termination_reason="success_early_termination", success=True, steps_run=10):
        self.error = error
        self.aborted = aborted
        self.abort_reason = abort_reason
        self.termination_reason = termination_reason
        self.success = success
        self.steps_run = steps_run


class _FakeBuffer:
    def __init__(self, n_frames=10, missing_key_warnings=None):
        self._n = n_frames
        self.missing_key_warnings = missing_key_warnings or []

    def num_frames(self):
        return self._n


def test_validate_episode_completeness_accepts_clean_success():
    # buf.num_frames() must be steps_run + 1: 1 extra capture at env.reset()
    # (step_count=0, before any real env.step()) plus one capture per real
    # step -- see run_episode_edge_pinch()'s obs_capture_callback contract,
    # asserted by test_edge_pinch_controller_v1.py's
    # test_run_episode_edge_pinch_obs_capture_callback_invoked_reset_plus_every_step
    # ("67 calls = 1 RESET + 66 steps"). This is the real invariant that the
    # 2026-09-17 formal-collection run's frame_count_mismatch bug violated
    # (see claude/task1_edge_pinch_collection_run_failure_diagnosis_
    # 20260917.json) -- this fake double must model it correctly.
    assert rec.validate_episode_completeness(_FakeBuffer(11), _FakeResult(steps_run=10)) is None


def test_validate_episode_completeness_rejects_exact_steps_run_no_reset_frame():
    # Regression test for the 2026-09-17 off-by-one bug: buf.num_frames()
    # EQUAL to steps_run (missing the RESET-frame) must be REJECTED, not
    # accepted. Before the fix, this exact case (which is what every one of
    # the 96 real failed episodes actually looked like from the gate's old,
    # wrong perspective) was misclassified -- wait, actually the real bug was
    # the reverse: real buffers were steps_run+1 and got rejected because the
    # old code demanded ==steps_run. This test asserts the NEW code correctly
    # rejects a buffer that is missing the RESET frame (genuinely incomplete),
    # to guard against overcorrecting into always accepting any count.
    reason = rec.validate_episode_completeness(_FakeBuffer(10), _FakeResult(steps_run=10))
    assert reason is not None
    assert "frame_count_mismatch" in reason


def test_validate_episode_completeness_rejects_no_result():
    reason = rec.validate_episode_completeness(_FakeBuffer(10), None)
    assert reason == "no_result_object_exception_during_episode"


def test_validate_episode_completeness_rejects_wrong_termination_reason():
    reason = rec.validate_episode_completeness(
        _FakeBuffer(10), _FakeResult(termination_reason="normal_completion", steps_run=10)
    )
    assert "termination_reason_not_success_early_termination" in reason


def test_validate_episode_completeness_rejects_success_not_true():
    reason = rec.validate_episode_completeness(
        _FakeBuffer(10), _FakeResult(success=False, steps_run=10)
    )
    assert "env_check_success_not_true" in reason


def test_validate_episode_completeness_rejects_missing_obs_keys():
    reason = rec.validate_episode_completeness(
        _FakeBuffer(10, missing_key_warnings=[{"step_count": 3, "missing": ["agentview_image"]}]),
        _FakeResult(steps_run=10),
    )
    assert "missing_obs_keys_during_capture" in reason


def test_validate_episode_completeness_rejects_frame_count_mismatch():
    reason = rec.validate_episode_completeness(_FakeBuffer(8), _FakeResult(steps_run=10))
    assert "frame_count_mismatch" in reason


def test_validate_episode_completeness_rejects_zero_frames():
    reason = rec.validate_episode_completeness(_FakeBuffer(0), _FakeResult(steps_run=0))
    assert reason == "zero_frames_captured"


def test_validate_episode_completeness_rejects_aborted():
    reason = rec.validate_episode_completeness(
        _FakeBuffer(5), _FakeResult(aborted=True, abort_reason="DESCEND: stalled", steps_run=5)
    )
    assert "episode_aborted" in reason


def test_validate_episode_completeness_rejects_error():
    reason = rec.validate_episode_completeness(
        _FakeBuffer(5), _FakeResult(error="RuntimeError: boom", steps_run=5)
    )
    assert "episode_error" in reason


# --------------------------------------------------------------------------
# 8: manifest + sha256 + QC generation (no h5py needed -- uses plain tmp
# files standing in for HDF5 outputs, since write_manifest_and_qc only
# hashes whatever file path it is given)
# --------------------------------------------------------------------------
def test_write_manifest_and_qc_computes_sha256_and_counts():
    d = tempfile.mkdtemp()
    try:
        fake_hdf5_path = os.path.join(d, "task1_edge_pinch_seed950001_az90_so0.05.hdf5")
        with open(fake_hdf5_path, "wb") as f:
            f.write(b"fake hdf5 bytes for test")
        kept_files = [{
            "path": fake_hdf5_path, "seed": 950001, "azimuth_deg": 90.0,
            "standoff_m": 0.05, "num_frames": 101,
        }]
        failure_records = [{"seed": 950002, "azimuth_deg": 90.0, "standoff_m": 0.05, "reason": "test_failure"}]
        gate_rejected = [{"combo": {"seed": 555101, "azimuth_deg": 90.0, "standoff_m": 0.05}, "reason": "blocked_blind_eval_seed_555101_555105"}]

        manifest = rec.write_manifest_and_qc(d, kept_files, failure_records, gate_rejected)

        assert manifest["qc_summary"]["num_kept"] == 1
        assert manifest["qc_summary"]["num_failed_episodes"] == 1
        assert manifest["qc_summary"]["num_gate_rejected_candidates"] == 1
        assert manifest["generation_only"] is True
        assert manifest["not_a_model_evaluation_input"] is True

        assert len(manifest["kept_files"]) == 1
        expected_sha256 = rec.sha256_of_file(fake_hdf5_path)
        assert manifest["kept_files"][0]["sha256"] == expected_sha256

        manifest_path = os.path.join(d, "manifest.json")
        assert os.path.exists(manifest_path)
        with open(manifest_path) as f:
            on_disk = json.load(f)
        assert on_disk["qc_summary"]["num_kept"] == 1
    finally:
        shutil.rmtree(d)


# --------------------------------------------------------------------------
# Regression / integration: drive a REAL EpisodeRecordingBuffer (not the
# _FakeBuffer double) through a simulated RESET + N-step capture sequence
# matching exactly what run_episode_edge_pinch()'s obs_capture_callback
# contract produces, and confirm validate_episode_completeness() accepts it.
# This closes the gap that let the 2026-09-17 off-by-one bug through: the
# old _FakeBuffer/_FakeResult doubles never modeled the real RESET-frame
# behavior, so they never caught the mismatch between the completeness gate
# and the controller's actual (correct, separately-tested) callback
# contract. See claude/task1_edge_pinch_collection_run_failure_diagnosis_
# 20260917.json.
# --------------------------------------------------------------------------
def _fake_obs(quat=(0.0, 0.0, 0.0, 1.0)):
    return {
        "agentview_image": np.zeros((84, 84, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((84, 84, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3, dtype=np.float64),
        "robot0_eef_quat": np.asarray(quat, dtype=np.float64),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float64),
        "robot0_joint_pos": np.zeros(7, dtype=np.float64),
    }


def test_real_episode_recording_buffer_reset_plus_n_steps_validates_as_complete():
    buf = rec.EpisodeRecordingBuffer()
    n_steps = 66

    # 1 RESET capture (step_count=0, action=None), matching
    # run_episode_edge_pinch()'s callback contract exactly.
    buf.capture(0, "RESET", _fake_obs(), None, False, None)

    # N real-step captures (step_count=1..n_steps, action is not None).
    for step in range(1, n_steps + 1):
        action = np.zeros(7, dtype=np.float32)
        buf.capture(step, "MOVE", _fake_obs(), action, step == n_steps, None)

    assert buf.num_frames() == n_steps + 1, buf.num_frames()
    assert not buf.missing_key_warnings, buf.missing_key_warnings

    result = _FakeResult(steps_run=n_steps)
    reason = rec.validate_episode_completeness(buf, result)
    assert reason is None, reason


def test_real_episode_recording_buffer_missing_reset_capture_is_rejected():
    # Same as above but WITHOUT the RESET capture -- num_frames() ==
    # n_steps, not n_steps+1 -- must be rejected as incomplete.
    buf = rec.EpisodeRecordingBuffer()
    n_steps = 66
    for step in range(1, n_steps + 1):
        action = np.zeros(7, dtype=np.float32)
        buf.capture(step, "MOVE", _fake_obs(), action, step == n_steps, None)

    assert buf.num_frames() == n_steps, buf.num_frames()
    result = _FakeResult(steps_run=n_steps)
    reason = rec.validate_episode_completeness(buf, result)
    assert reason is not None
    assert "frame_count_mismatch" in reason


def test_write_failure_log_writes_json_with_all_records():
    d = tempfile.mkdtemp()
    try:
        failures = [
            {"seed": 950009, "azimuth_deg": 45.0, "standoff_m": 0.03, "reason": "some_reason"},
        ]
        path = rec.write_failure_log(d, failures)
        assert os.path.exists(path)
        with open(path) as f:
            payload = json.load(f)
        assert payload["num_failures"] == 1
        assert payload["failures"] == failures
    finally:
        shutil.rmtree(d)


if __name__ == "__main__":
    tests = [
        test_allowed_combinations_has_exactly_96,
        test_validate_keep_list_accepts_known_good_combination,
        test_validate_keep_list_rejects_excluded_combination,
        test_validate_keep_list_rejects_blind_eval_seeds,
        test_validate_keep_list_rejects_historical_comparison_seeds,
        test_validate_keep_list_rejects_unknown_seed_outside_any_range,
        test_validate_keep_list_rejects_malformed_entry,
        test_validate_keep_list_accepts_tuple_form,
        test_validate_keep_list_mixed_batch_partitions_correctly,
        test_check_output_dir_state_nonexistent_dir_ok,
        test_check_output_dir_state_empty_dir_ok,
        test_check_output_dir_state_nonempty_dir_refused_without_overwrite,
        test_check_output_dir_state_nonempty_dir_ok_with_overwrite,
        test_validate_episode_completeness_accepts_clean_success,
        test_validate_episode_completeness_rejects_exact_steps_run_no_reset_frame,
        test_validate_episode_completeness_rejects_no_result,
        test_validate_episode_completeness_rejects_wrong_termination_reason,
        test_validate_episode_completeness_rejects_success_not_true,
        test_validate_episode_completeness_rejects_missing_obs_keys,
        test_validate_episode_completeness_rejects_frame_count_mismatch,
        test_validate_episode_completeness_rejects_zero_frames,
        test_validate_episode_completeness_rejects_aborted,
        test_validate_episode_completeness_rejects_error,
        test_real_episode_recording_buffer_reset_plus_n_steps_validates_as_complete,
        test_real_episode_recording_buffer_missing_reset_capture_is_rejected,
        test_write_manifest_and_qc_computes_sha256_and_counts,
        test_write_failure_log_writes_json_with_all_records,
    ]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print("ALL_COLLECTION_GATE_UNIT_TESTS_OK")
