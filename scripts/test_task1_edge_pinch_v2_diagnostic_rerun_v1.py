#!/usr/bin/env python3
"""Local (no-GPU/torch/libero) tests for task1_edge_pinch_v2_diagnostic_rerun_v1.py.

Same fake-injection strategy as test_task1_edge_pinch_v2_blind_eval_v1.py:
a fake 'torch' module is installed into sys.modules before any code path
that does `import torch` (frame(), select_action call site) runs, and a
fake 'PIL' Image module stands in for real Pillow so keyframe saving can
be tested without the real library installed locally.
"""
from __future__ import annotations

import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


class _FakeImageTensor:
    """Stands in for a torch.Tensor through the frame() permute().float()/255.0
    chain AND through the post(raw).detach().cpu() / np.asarray(...) chain
    used for actions -- never used numerically, only needs to not crash."""

    def __init__(self, arr):
        self._arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def permute(self, *dims):
        return self

    def float(self):
        return self

    def __truediv__(self, other):
        return self

    def __array__(self, dtype=None):
        import numpy as np
        return np.asarray(self._arr, dtype=dtype)


def _install_fake_torch():
    if "torch" in sys.modules and getattr(sys.modules["torch"], "_is_fake_for_tests", False):
        return sys.modules["torch"]

    fake = types.ModuleType("torch")
    fake._is_fake_for_tests = True

    class _InferenceModeCtx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def as_tensor(x):
        return _FakeImageTensor(x)

    def from_numpy(x):
        import numpy as np
        return np.asarray(x)

    fake.as_tensor = as_tensor
    fake.from_numpy = from_numpy
    fake.inference_mode = lambda: _InferenceModeCtx()
    fake.device = lambda *a, **k: "cpu"

    class _Cuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def empty_cache():
            return None

    fake.cuda = _Cuda()

    class _FakeTorchTensorMethods:
        pass

    sys.modules["torch"] = fake
    return fake


def _install_fake_pil():
    saved = []

    fake_pil = types.ModuleType("PIL")
    fake_image_mod = types.ModuleType("PIL.Image")

    class _FakeImg:
        def __init__(self, arr):
            self._arr = arr

        def save(self, path):
            saved.append(str(path))
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"FAKE_PNG")

    def fromarray(arr):
        return _FakeImg(arr)

    fake_image_mod.fromarray = fromarray
    fake_pil.Image = fake_image_mod
    sys.modules["PIL"] = fake_pil
    sys.modules["PIL.Image"] = fake_image_mod
    return fake_image_mod, saved


_install_fake_torch()


def _make_torch_like_action(values):
    return _FakeImageTensor(values)


class _FakeEnv:
    """step_outcomes: list of (reward, done) for successive .step() calls
    (both wait-phase and control-phase steps consume from the same queue).
    success_at_step_indices: 1-based count of .step() calls after which
    check_success() should return True (checked AFTER incrementing)."""

    def __init__(self, step_outcomes, success_at_step_indices=(), bowl_positions=None, plate_pos=(1.0, 1.0, 0.0)):
        self._outcomes = list(step_outcomes)
        self._success_at = set(success_at_step_indices)
        self._step_count = 0
        self.closed = False
        self._bowl_positions = list(bowl_positions) if bowl_positions is not None else None
        self._plate_pos = list(plate_pos)

    def _bowl_pos_for(self, idx):
        if self._bowl_positions is None:
            return [0.0, 0.0, 0.0]
        i = min(idx, len(self._bowl_positions) - 1)
        return list(self._bowl_positions[i])

    def _obs(self):
        import numpy as np
        return {
            "agentview_image": np.zeros((8, 8, 3), dtype="uint8"),
            "robot0_eye_in_hand_image": np.zeros((8, 8, 3), dtype="uint8"),
            "robot0_joint_pos": np.zeros(7, dtype="float32"),
            "robot0_eef_pos": np.zeros(3, dtype="float32"),
            "robot0_eef_quat": np.array([0, 0, 0, 1], dtype="float32"),
            "robot0_gripper_qpos": np.array([0.02 + 0.001 * self._step_count, -0.02 - 0.001 * self._step_count], dtype="float32"),
            "akita_black_bowl_1_pos": np.asarray(self._bowl_pos_for(self._step_count), dtype="float64"),
            "plate_1_pos": np.asarray(self._plate_pos, dtype="float64"),
        }

    def reset(self):
        self._step_count = 0
        return self._obs()

    def step(self, action):
        reward, done = self._outcomes[self._step_count] if self._step_count < len(self._outcomes) else (0.0, False)
        self._step_count += 1
        return self._obs(), reward, done, {}

    def check_success(self):
        return self._step_count in self._success_at

    def close(self):
        self.closed = True


class _FakePolicy:
    def __init__(self, action_out=None):
        self.reset_calls = 0
        self.select_action_calls = 0
        self.config = types.SimpleNamespace(n_action_steps=None)
        self._action_out = action_out if action_out is not None else [0.0] * 7

    def reset(self):
        self.reset_calls += 1

    def select_action(self, batch):
        self.select_action_calls += 1
        return _make_torch_like_action(list(self._action_out))


def identity_pre(x):
    return x


def identity_post(x):
    return x


class TestDiagnosticRerun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_fake_torch()
        cls.image_mod, cls.saved_paths = _install_fake_pil()
        cls.mod = importlib.import_module("task1_edge_pinch_v2_diagnostic_rerun_v1")

    def setUp(self):
        self.saved_paths.clear()

    # -- pure helpers --

    def test_parse_pair(self):
        self.assertEqual(self.mod.parse_pair("2500:555103"), (2500, 555103))
        self.assertEqual(self.mod.parse_pair("3000:555201"), (3000, 555201))

    def test_gate_a_modes(self):
        ns = types.SimpleNamespace(smoke_test=True, confirm_full_run=False)
        self.assertEqual(self.mod.gate_a_resolve_mode(ns), "smoke_test")
        ns = types.SimpleNamespace(smoke_test=False, confirm_full_run=True)
        self.assertEqual(self.mod.gate_a_resolve_mode(ns), "full_run")
        ns = types.SimpleNamespace(smoke_test=True, confirm_full_run=True)
        with self.assertRaises(ValueError):
            self.mod.gate_a_resolve_mode(ns)
        ns = types.SimpleNamespace(smoke_test=False, confirm_full_run=False)
        with self.assertRaises(ValueError):
            self.mod.gate_a_resolve_mode(ns)

    def test_gate_c_output_dir(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            nonexist = d / "nope"
            self.mod.gate_c_check_output_dir(nonexist, overwrite=False)  # no raise
            (d / "f.txt").write_text("x")
            with self.assertRaises(FileExistsError):
                self.mod.gate_c_check_output_dir(d, overwrite=False)
            self.mod.gate_c_check_output_dir(d, overwrite=True)  # no raise

    def test_blocked_seed_rejected_in_validate_seed_reused_from_blind_eval(self):
        with self.assertRaises(ValueError):
            self.mod.validate_seed(555001)  # blocked historical
        with self.assertRaises(ValueError):
            self.mod.validate_seed(950005)  # training collection seed
        self.mod.validate_seed(555103)  # allowed batch1 seed, must not raise
        self.mod.validate_seed(555201)  # allowed batch2 seed, must not raise

    # -- run_one_diagnostic_rollout --

    def test_success_path_records_object_pose_and_flags(self):
        env = _FakeEnv(
            step_outcomes=[(0.0, False)] * 5,
            success_at_step_indices={3},
            bowl_positions=[[0, 0, 0.9]] * 10,
            plate_pos=(0.3, 0.3, 0.85),
        )
        policy = _FakePolicy()
        with tempfile.TemporaryDirectory() as d:
            kf_dir = Path(d) / "kf"
            result = self.mod.run_one_diagnostic_rollout(
                policy, identity_pre, identity_post, lambda: env,
                seed=555103, lang="lang", n_action_steps=25, wait_steps=2, max_steps=10,
                keyframe_dir=kf_dir, png_module=self.image_mod,
            )
        self.assertTrue(result["success"])
        self.assertTrue(result["object_pose_diagnostic_only"])
        self.assertTrue(result["object_pose_not_used_as_model_input"])
        self.assertAlmostEqual(
            result["final_bowl_plate_distance_m"],
            ((0.3) ** 2 + (0.3) ** 2 + (0.85 - 0.9) ** 2) ** 0.5,
            places=6,
        )
        self.assertEqual(result["bowl_displacement_from_reset_m"], 0.0)  # bowl never moves in this fixture

    def test_keyframe_saved_at_step_zero_and_every_interval_and_final(self):
        # 120 successful control steps (no success, no done) -> keyframes expected at
        # step_0000 (pre-loop), step_0050, step_0100, and a final step_0120_final
        env = _FakeEnv(step_outcomes=[(0.0, False)] * 130)
        policy = _FakePolicy()
        with tempfile.TemporaryDirectory() as d:
            kf_dir = Path(d) / "kf"
            result = self.mod.run_one_diagnostic_rollout(
                policy, identity_pre, identity_post, lambda: env,
                seed=555103, lang="lang", n_action_steps=25, wait_steps=0, max_steps=120,
                keyframe_dir=kf_dir, png_module=self.image_mod,
            )
            self.assertTrue(kf_dir.exists())
        self.assertEqual(result["steps_run"], 120)
        self.assertIn("step_0000.png", result["keyframes_saved"])
        self.assertIn("step_0050.png", result["keyframes_saved"])
        self.assertIn("step_0100.png", result["keyframes_saved"])
        self.assertIn("step_0120_final.png", result["keyframes_saved"])
        self.assertTrue(result["keyframes_are_png"])

    def test_gripper_sequence_length_matches_wait_plus_control_steps(self):
        env = _FakeEnv(step_outcomes=[(0.0, False)] * 10)
        policy = _FakePolicy()
        with tempfile.TemporaryDirectory() as d:
            result = self.mod.run_one_diagnostic_rollout(
                policy, identity_pre, identity_post, lambda: env,
                seed=555103, lang="lang", n_action_steps=25, wait_steps=3, max_steps=5,
                keyframe_dir=Path(d) / "kf", png_module=self.image_mod,
            )
        # 3 wait steps + 5 control steps = 8 gripper samples
        self.assertEqual(len(result["gripper_qpos_sequence"]), 8)
        self.assertEqual(result["steps_run"], 5)

    def test_wait_phase_done_short_circuits_no_keyframes_no_policy_calls(self):
        env = _FakeEnv(step_outcomes=[(0.0, True)])  # done on first (wait) step
        policy = _FakePolicy()
        with tempfile.TemporaryDirectory() as d:
            result = self.mod.run_one_diagnostic_rollout(
                policy, identity_pre, identity_post, lambda: env,
                seed=555103, lang="lang", n_action_steps=25, wait_steps=1, max_steps=5,
                keyframe_dir=Path(d) / "kf", png_module=self.image_mod,
            )
        self.assertEqual(result["termination_reason"], "wait_phase_done")
        self.assertEqual(policy.reset_calls, 0)
        self.assertEqual(policy.select_action_calls, 0)
        self.assertEqual(result["keyframes_saved"], [])

    def test_invalid_action_stops_early_and_still_records_final_pose(self):
        env = _FakeEnv(step_outcomes=[(0.0, False)] * 5)
        policy = _FakePolicy(action_out=[0.0] * 6)  # wrong shape -> invalid
        with tempfile.TemporaryDirectory() as d:
            result = self.mod.run_one_diagnostic_rollout(
                policy, identity_pre, identity_post, lambda: env,
                seed=555103, lang="lang", n_action_steps=25, wait_steps=0, max_steps=5,
                keyframe_dir=Path(d) / "kf", png_module=self.image_mod,
            )
        self.assertTrue(result["invalid_action_detected"])
        self.assertEqual(result["termination_reason"], "invalid_action")
        self.assertIn("final_bowl_plate_distance_m", result)

    def test_env_closed_even_on_exception(self):
        class _RaisingEnv(_FakeEnv):
            def step(self, action):
                raise RuntimeError("boom")

        env = _RaisingEnv(step_outcomes=[])
        policy = _FakePolicy()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                self.mod.run_one_diagnostic_rollout(
                    policy, identity_pre, identity_post, lambda: env,
                    seed=555103, lang="lang", n_action_steps=25, wait_steps=1, max_steps=5,
                    keyframe_dir=Path(d) / "kf", png_module=self.image_mod,
                )
        self.assertTrue(env.closed)

    def test_blocked_seed_rejected_before_env_constructed(self):
        calls = {"n": 0}

        def env_fn():
            calls["n"] += 1
            return _FakeEnv(step_outcomes=[])

        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                self.mod.run_one_diagnostic_rollout(
                    _FakePolicy(), identity_pre, identity_post, env_fn,
                    seed=555001, lang="lang", n_action_steps=25, wait_steps=1, max_steps=5,
                    keyframe_dir=Path(d) / "kf", png_module=self.image_mod,
                )
        self.assertEqual(calls["n"], 0)

    def test_npy_fallback_when_no_png_module(self):
        env = _FakeEnv(step_outcomes=[(0.0, False)] * 5)
        policy = _FakePolicy()
        with tempfile.TemporaryDirectory() as d:
            result = self.mod.run_one_diagnostic_rollout(
                policy, identity_pre, identity_post, lambda: env,
                seed=555103, lang="lang", n_action_steps=25, wait_steps=0, max_steps=5,
                keyframe_dir=Path(d) / "kf", png_module=None,
            )
        self.assertFalse(result["keyframes_are_png"])
        self.assertTrue(all(k.endswith(".npy") for k in result["keyframes_saved"]))


if __name__ == "__main__":
    unittest.main()
