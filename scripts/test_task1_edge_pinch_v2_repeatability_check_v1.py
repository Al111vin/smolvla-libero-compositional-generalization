#!/usr/bin/env python3
"""Local (no-GPU/libero) tests for task1_edge_pinch_v2_repeatability_check_v1.py.
Same fake-torch-injection strategy as the other test files in this project."""
from __future__ import annotations

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


class _FakeImageTensor:
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
    fake.__version__ = "0.0.0-fake"

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

    def manual_seed(seed):
        return None

    fake.as_tensor = as_tensor
    fake.from_numpy = from_numpy
    fake.manual_seed = manual_seed
    fake.inference_mode = lambda: _InferenceModeCtx()
    fake.device = lambda *a, **k: "cpu"

    def use_deterministic_algorithms(flag, warn_only=False):
        return None

    fake.use_deterministic_algorithms = use_deterministic_algorithms

    class _Backends:
        class cudnn:
            benchmark = True
            deterministic = False

            @staticmethod
            def version():
                return 0

    fake.backends = _Backends()

    class _Cuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def empty_cache():
            return None

        @staticmethod
        def manual_seed_all(seed):
            return None

        @staticmethod
        def get_device_name(idx):
            return "fake-gpu"

    fake.cuda = _Cuda()

    sys.modules["torch"] = fake
    return fake


_install_fake_torch()


def _make_torch_like_action(values):
    return _FakeImageTensor(values)


class _FakeEnv:
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
            "robot0_gripper_qpos": np.array([0.02, -0.02], dtype="float32"),
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
        self._action_out = action_out if action_out is not None else [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.5]

    def reset(self):
        self.reset_calls += 1

    def select_action(self, batch):
        self.select_action_calls += 1
        return _make_torch_like_action(list(self._action_out))


def identity_pre(x):
    return x


def identity_post(x):
    return x


class TestRepeatabilityCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_fake_torch()
        cls.mod = importlib.import_module("task1_edge_pinch_v2_repeatability_check_v1")

    def test_parse_pair(self):
        self.assertEqual(self.mod.parse_pair("2500:555103"), (2500, 555103))
        self.assertEqual(self.mod.parse_pair("3000:555102"), (3000, 555102))

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
            self.mod.gate_c_check_output_dir(d / "nope", overwrite=False)
            (d / "f.txt").write_text("x")
            with self.assertRaises(FileExistsError):
                self.mod.gate_c_check_output_dir(d, overwrite=False)
            self.mod.gate_c_check_output_dir(d, overwrite=True)

    def test_apply_determinism_settings_disables_benchmark_and_records_report(self):
        report = self.mod.apply_determinism_settings()
        import torch
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertIn("torch_version", report)
        self.assertIn("cudnn_benchmark_after", report)
        self.assertFalse(report["cudnn_benchmark_after"])
        self.assertIn("use_deterministic_algorithms_requested", report)

    def test_blocked_seed_rejected_before_env_constructed(self):
        calls = {"n": 0}

        def env_fn():
            calls["n"] += 1
            return _FakeEnv(step_outcomes=[])

        with self.assertRaises(ValueError):
            self.mod.run_one_repeatability_rollout(
                _FakePolicy(), identity_pre, identity_post, env_fn,
                seed=555001, lang="lang", n_action_steps=25, wait_steps=1, max_steps=5,
            )
        self.assertEqual(calls["n"], 0)

    def test_action_snapshots_captured_at_expected_steps(self):
        env = _FakeEnv(step_outcomes=[(0.0, False)] * 450)
        policy = _FakePolicy()
        result = self.mod.run_one_repeatability_rollout(
            policy, identity_pre, identity_post, lambda: env,
            seed=555103, lang="lang", n_action_steps=25, wait_steps=0, max_steps=420,
        )
        self.assertEqual(result["steps_run"], 420)
        snaps = result["action_snapshots"]
        for expected_step in ["1", "50", "100", "200", "400"]:
            self.assertIn(expected_step, snaps)
            self.assertEqual(len(snaps[expected_step]), 7)
        self.assertIn("final_420", snaps)

    def test_success_path_records_object_pose(self):
        env = _FakeEnv(
            step_outcomes=[(0.0, False)] * 5,
            success_at_step_indices={3},
            bowl_positions=[[0, 0, 0.9]] * 10,
            plate_pos=(0.1, 0.1, 0.9),
        )
        policy = _FakePolicy()
        result = self.mod.run_one_repeatability_rollout(
            policy, identity_pre, identity_post, lambda: env,
            seed=555103, lang="lang", n_action_steps=25, wait_steps=1, max_steps=10,
        )
        self.assertTrue(result["success"])
        self.assertTrue(result["object_pose_diagnostic_only"])
        self.assertAlmostEqual(result["final_bowl_plate_distance_m"], (0.1**2 + 0.1**2) ** 0.5, places=6)

    def test_env_closed_even_on_exception(self):
        class _RaisingEnv(_FakeEnv):
            def step(self, action):
                raise RuntimeError("boom")

        env = _RaisingEnv(step_outcomes=[])
        policy = _FakePolicy()
        with self.assertRaises(RuntimeError):
            self.mod.run_one_repeatability_rollout(
                policy, identity_pre, identity_post, lambda: env,
                seed=555103, lang="lang", n_action_steps=25, wait_steps=1, max_steps=5,
            )
        self.assertTrue(env.closed)

    def test_wait_phase_done_short_circuits_no_snapshots_no_policy_calls(self):
        env = _FakeEnv(step_outcomes=[(0.0, True)])
        policy = _FakePolicy()
        result = self.mod.run_one_repeatability_rollout(
            policy, identity_pre, identity_post, lambda: env,
            seed=555103, lang="lang", n_action_steps=25, wait_steps=1, max_steps=5,
        )
        self.assertEqual(result["termination_reason"], "wait_phase_done")
        self.assertEqual(policy.reset_calls, 0)
        self.assertEqual(result["action_snapshots"], {})

    def test_seed_everything_calls_numpy_and_torch_seed_apis(self):
        # smoke check that it runs without error against both real numpy and
        # the fake torch module -- no assertion on internal RNG state since
        # the fake torch doesn't track it.
        self.mod.seed_everything(555103)


if __name__ == "__main__":
    unittest.main()
