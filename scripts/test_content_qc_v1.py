"""test_content_qc_v1.py

Offline logic test for task1_edge_pinch_v2_content_qc_v1.py. No real h5py
or real GPU data required: injects a minimal fake 'h5py' module into
sys.modules that mimics just enough of the h5py.File/Group/Dataset API
for qc_one_file() to exercise its real logic against synthetic in-memory
arrays built to match the EXACT schema written by
task1_edge_pinch_diverse_collection_recorder_v1.py's main() (data/demo_0
group, obs/ subgroup, attrs) -- see write_episode_to_hdf5() and the
episode_attrs dict built in main() for the ground truth being mirrored
here. This validates the QC script's shape/attr/continuity/sha256 logic
BEFORE it is deployed to GPU and run read-only against the real 96 files.

Run: python3 test_content_qc_v1.py
Expects: ALL_CONTENT_QC_UNIT_TESTS_OK printed at the end.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

import numpy as np


# ---------------------------------------------------------------------------
# Minimal fake h5py, injected into sys.modules BEFORE importing the module
# under test (which does `import h5py` lazily inside qc_one_file()).
# ---------------------------------------------------------------------------
class _FakeDataset:
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    @property
    def shape(self):
        return self._arr.shape

    @property
    def dtype(self):
        return self._arr.dtype

    @property
    def ndim(self):
        return self._arr.ndim

    def __getitem__(self, key):
        # supports ds[()] (full read) used by the module under test
        return self._arr[key] if key != () else self._arr


class _FakeGroup:
    def __init__(self, contents: dict, attrs: dict | None = None):
        self._contents = contents
        self.attrs = attrs or {}

    def __contains__(self, key):
        return key in self._contents

    def __getitem__(self, key):
        return self._contents[key]


_REGISTRY: dict = {}  # path -> root FakeGroup, set by each test before invoking qc_one_file


class _FakeFile:
    def __init__(self, path, mode):
        assert mode == "r", "QC script must open read-only"
        self._root = _REGISTRY[path]

    def __enter__(self):
        return self._root

    def __exit__(self, *exc):
        return False

    def __contains__(self, key):
        return key in self._root._contents

    def __getitem__(self, key):
        return self._root[key]


class _FakeH5pyModule:
    File = _FakeFile


sys.modules["h5py"] = _FakeH5pyModule()  # must precede the `import h5py` inside qc_one_file

import task1_edge_pinch_v2_content_qc_v1 as qc  # noqa: E402


def _make_clean_episode(seed=950001, azimuth_deg=90.0, standoff_m=0.03, n_frames=100, steps_run=99):
    rng = np.random.default_rng(seed)
    agentview = np.zeros((n_frames, 84, 84, 3), dtype=np.uint8)
    eye_in_hand = np.zeros((n_frames, 84, 84, 3), dtype=np.uint8)
    ee_pos = rng.normal(size=(n_frames, 3))
    # smooth, small-step ee_ori sequence -- no discontinuity
    ee_ori = np.cumsum(rng.normal(scale=0.01, size=(n_frames, 3)), axis=0)
    gripper = rng.normal(size=(n_frames, 2))
    joints = rng.normal(size=(n_frames, 7))
    actions = rng.uniform(-1, 1, size=(n_frames, 7)).astype(np.float32)
    dones = np.zeros(n_frames, dtype=bool)
    dones[-1] = True

    obs = _FakeGroup({
        "agentview_rgb": _FakeDataset(agentview),
        "eye_in_hand_rgb": _FakeDataset(eye_in_hand),
        "ee_pos": _FakeDataset(ee_pos),
        "ee_ori": _FakeDataset(ee_ori),
        "gripper_states": _FakeDataset(gripper),
        "joint_states": _FakeDataset(joints),
    })
    attrs = {
        "seed": seed, "azimuth_deg": azimuth_deg, "standoff_m": standoff_m,
        "num_samples": n_frames, "termination_reason": "success_early_termination",
        "env_check_success": True, "steps_run": steps_run,
        "controller_script_sha256": "deadbeef" * 8,
        "recording_script_version": "task1_edge_pinch_diverse_collection_recorder_v1",
    }
    demo = _FakeGroup({
        "obs": obs,
        "actions": _FakeDataset(actions),
        "dones": _FakeDataset(dones),
    }, attrs=attrs)
    data = _FakeGroup({"demo_0": demo})
    root = _FakeGroup({"data": data})
    return root


def _register_and_write_dummy_file(root, seed=950001, azimuth_deg=90.0, standoff_m=0.03, content=b"dummy"):
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, f"task1_edge_pinch_seed{seed}_az{int(azimuth_deg)}_so{standoff_m}.hdf5")
    with open(path, "wb") as f:
        f.write(content)
    _REGISTRY[path] = root
    return path


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_clean_episode_has_zero_problems():
    root = _make_clean_episode()
    path = _register_and_write_dummy_file(root)
    manifest_sha = {os.path.abspath(path): _sha256_bytes(b"dummy")}
    record = qc.qc_one_file(path, manifest_sha)
    assert record["problems"] == [], record["problems"]
    assert record["num_frames"] == 100
    assert "fatal_error" not in record


def test_sha256_mismatch_detected():
    root = _make_clean_episode()
    path = _register_and_write_dummy_file(root)
    manifest_sha = {os.path.abspath(path): "0" * 64}  # deliberately wrong
    record = qc.qc_one_file(path, manifest_sha)
    assert "sha256_mismatch_vs_manifest" in record["problems"]


def test_wrong_termination_reason_detected():
    root = _make_clean_episode()
    root._contents["data"]._contents["demo_0"].attrs["termination_reason"] = "normal_completion"
    path = _register_and_write_dummy_file(root)
    manifest_sha = {os.path.abspath(path): _sha256_bytes(b"dummy")}
    record = qc.qc_one_file(path, manifest_sha)
    assert any("termination_reason_not_success_early_termination" in p for p in record["problems"])


def test_env_check_success_false_detected():
    root = _make_clean_episode()
    root._contents["data"]._contents["demo_0"].attrs["env_check_success"] = False
    path = _register_and_write_dummy_file(root)
    manifest_sha = {os.path.abspath(path): _sha256_bytes(b"dummy")}
    record = qc.qc_one_file(path, manifest_sha)
    assert any("env_check_success_not_true" in p for p in record["problems"])


def test_frame_count_invariant_violation_detected():
    root = _make_clean_episode(n_frames=100, steps_run=100)  # should be steps_run+1=101, not 100
    path = _register_and_write_dummy_file(root)
    manifest_sha = {os.path.abspath(path): _sha256_bytes(b"dummy")}
    record = qc.qc_one_file(path, manifest_sha)
    assert any("num_frames_not_steps_run_plus_1" in p for p in record["problems"])


def test_nan_in_actions_detected():
    root = _make_clean_episode()
    actions_arr = root._contents["data"]._contents["demo_0"]._contents["actions"]._arr
    actions_arr[0, 0] = np.nan
    path = _register_and_write_dummy_file(root)
    manifest_sha = {os.path.abspath(path): _sha256_bytes(b"dummy")}
    record = qc.qc_one_file(path, manifest_sha)
    assert any("actions_contains_nan" in p for p in record["problems"])


def test_inf_in_actions_detected():
    root = _make_clean_episode()
    actions_arr = root._contents["data"]._contents["demo_0"]._contents["actions"]._arr
    actions_arr[0, 0] = np.inf
    path = _register_and_write_dummy_file(root)
    manifest_sha = {os.path.abspath(path): _sha256_bytes(b"dummy")}
    record = qc.qc_one_file(path, manifest_sha)
    assert any("actions_contains_inf" in p for p in record["problems"])


def test_hemisphere_discontinuity_jump_detected():
    root = _make_clean_episode()
    ee_ori_arr = root._contents["data"]._contents["demo_0"]._contents["obs"]._contents["ee_ori"]._arr
    ee_ori_arr[50] += 3.0  # inject a large artificial jump
    path = _register_and_write_dummy_file(root)
    manifest_sha = {os.path.abspath(path): _sha256_bytes(b"dummy")}
    record = qc.qc_one_file(path, manifest_sha)
    assert any("suspected_hemisphere_discontinuity" in p for p in record["problems"])
    assert record["ee_ori_max_consecutive_frame_delta_rad"] > qc.CONTINUITY_JUMP_THRESHOLD_RAD


def test_blind_eval_seed_flagged():
    root = _make_clean_episode(seed=555103)
    path = _register_and_write_dummy_file(root, seed=555103)
    manifest_sha = {os.path.abspath(path): _sha256_bytes(b"dummy")}
    record = qc.qc_one_file(path, manifest_sha)
    assert "CONTAINS_BLOCKED_BLIND_EVAL_SEED" in record["problems"]


def test_historical_comparison_seed_flagged():
    root = _make_clean_episode(seed=555003)
    path = _register_and_write_dummy_file(root, seed=555003)
    manifest_sha = {os.path.abspath(path): _sha256_bytes(b"dummy")}
    record = qc.qc_one_file(path, manifest_sha)
    assert "CONTAINS_FLAGGED_HISTORICAL_COMPARISON_SEED" in record["problems"]


def test_wrong_image_dtype_detected():
    root = _make_clean_episode()
    obs = root._contents["data"]._contents["demo_0"]._contents["obs"]
    bad = obs._contents["agentview_rgb"]._arr.astype(np.float32)
    obs._contents["agentview_rgb"] = _FakeDataset(bad)
    path = _register_and_write_dummy_file(root)
    manifest_sha = {os.path.abspath(path): _sha256_bytes(b"dummy")}
    record = qc.qc_one_file(path, manifest_sha)
    assert any("agentview_rgb_unexpected_dtype" in p for p in record["problems"])


def test_inconsistent_frame_counts_detected():
    root = _make_clean_episode(n_frames=100, steps_run=99)
    obs = root._contents["data"]._contents["demo_0"]._contents["obs"]
    truncated = obs._contents["ee_pos"]._arr[:90]
    obs._contents["ee_pos"] = _FakeDataset(truncated)
    path = _register_and_write_dummy_file(root)
    manifest_sha = {os.path.abspath(path): _sha256_bytes(b"dummy")}
    record = qc.qc_one_file(path, manifest_sha)
    assert any("inconsistent_frame_counts_across_datasets" in p for p in record["problems"])


def test_missing_path_not_in_manifest_detected():
    root = _make_clean_episode()
    path = _register_and_write_dummy_file(root)
    record = qc.qc_one_file(path, {})  # empty manifest map
    assert "not_found_in_manifest_kept_files" in record["problems"]


def test_fatal_error_does_not_raise_and_is_captured():
    # Register nothing for this path -> _FakeFile.__init__ will KeyError
    # inside qc_one_file's try/except, must be captured, not propagated.
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "unregistered.hdf5")
    with open(path, "wb") as f:
        f.write(b"x")
    record = qc.qc_one_file(path, {})
    assert "fatal_error" in record
    assert "fatal_error_during_inspection" in record["problems"]


if __name__ == "__main__":
    tests = [
        test_clean_episode_has_zero_problems,
        test_sha256_mismatch_detected,
        test_wrong_termination_reason_detected,
        test_env_check_success_false_detected,
        test_frame_count_invariant_violation_detected,
        test_nan_in_actions_detected,
        test_inf_in_actions_detected,
        test_hemisphere_discontinuity_jump_detected,
        test_blind_eval_seed_flagged,
        test_historical_comparison_seed_flagged,
        test_wrong_image_dtype_detected,
        test_inconsistent_frame_counts_detected,
        test_missing_path_not_in_manifest_detected,
        test_fatal_error_does_not_raise_and_is_captured,
    ]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print("ALL_CONTENT_QC_UNIT_TESTS_OK")
