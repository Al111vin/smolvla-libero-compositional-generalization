#!/usr/bin/env python3
"""Offline unit tests for task1_edge_pinch_v2_blind_eval_v1.py.

torch/libero are not installed in this local sandbox, so:
  - pure functions that don't need torch (qaxis, validate_seed,
    sha256_of_file, resolve_checkpoint_dir, the three gate functions,
    check_policy_api_compat) are tested directly against the real module.
  - run_one_blind_rollout() and frame() import torch internally, so a
    minimal fake torch module is injected into sys.modules before those
    tests run, mirroring this project's established pattern (fake h5py in
    test_content_qc_v1.py) for testing GPU-only code paths offline.

libero is never imported by anything under test here (make_env_fn's
closure defers the import until called, and no test calls that closure),
so no libero fake is needed.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def _install_fake_torch():
    """Installs a minimal fake `torch` module sufficient for frame() and
    run_one_blind_rollout()'s torch usage, then returns it. Idempotent --
    safe to call multiple times."""
    if "torch" in sys.modules and getattr(sys.modules["torch"], "_IS_FAKE_TORCH", False):
        return sys.modules["torch"]

    fake_torch = types.ModuleType("torch")
    fake_torch._IS_FAKE_TORCH = True

    class _FakeImageTensor:
        """Stands in for a torch.Tensor through the frame()
        permute().float()/255.0 chain. Never actually used numerically by
        any test (pre/post are test doubles), only needs to not crash."""

        def __init__(self, data):
            self._data = data

        def permute(self, *dims):
            return self

        def float(self):
            return self

        def __truediv__(self, other):
            return self

    def as_tensor(x):
        return _FakeImageTensor(x)

    def from_numpy(arr):
        return np.asarray(arr)

    class _InferenceMode:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def inference_mode():
        return _InferenceMode()

    class _FakeCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def empty_cache():
            pass

    fake_torch.as_tensor = as_tensor
    fake_torch.from_numpy = from_numpy
    fake_torch.inference_mode = inference_mode
    fake_torch.cuda = _FakeCuda()
    fake_torch.device = lambda *a, **k: "cpu"

    sys.modules["torch"] = fake_torch
    return fake_torch


class _FakeTorchAction(np.ndarray):
    """A numpy array that also exposes .detach()/.cpu(), standing in for a
    torch.Tensor action so the hasattr(processed, "detach") branch in
    run_one_blind_rollout is exercised."""

    def detach(self):
        return self

    def cpu(self):
        return self


def _make_torch_like_action(values):
    arr = np.asarray(values, dtype=np.float32).view(_FakeTorchAction)
    return arr


def _counting_callable(return_value_factory):
    calls = {"n": 0}

    def _fn(x):
        calls["n"] += 1
        return return_value_factory()

    _fn.calls = calls
    return _fn


class _FakeEnv:
    """Deterministic fake LIBERO env.

    step_outcomes: list of (obs, reward, done) tuples consumed in order,
    one per env.step() call (wait phase + control phase share the same
    queue). success_on_step_indices: 0-based step.step() call indices
    (across the whole episode, including wait steps) at which
    check_success() should return True.
    """

    def __init__(self, step_outcomes, success_on_step_indices=frozenset(), reset_obs=None):
        self._outcomes = list(step_outcomes)
        self._success_indices = set(success_on_step_indices)
        self._reset_obs = reset_obs if reset_obs is not None else _obs()
        self._step_idx = 0
        self.closed = False
        self.reset_called = 0
        self.step_calls = []

    def reset(self):
        self.reset_called += 1
        return self._reset_obs

    def step(self, action):
        self.step_calls.append(np.asarray(action).copy())
        obs, reward, done = self._outcomes[self._step_idx]
        self._step_idx += 1
        return obs, reward, done, {}

    def check_success(self):
        return (self._step_idx - 1) in self._success_indices

    def close(self):
        self.closed = True


def _obs():
    return {
        "robot0_joint_pos": np.zeros(7, dtype=np.float64),
        "robot0_eef_pos": np.zeros(3, dtype=np.float64),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float64),
        "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((128, 128, 3), dtype=np.uint8),
    }


class _FakePolicy:
    def __init__(self, compatible=True):
        self.reset_calls = 0
        self.select_action_calls = 0
        if compatible:
            self.config = types.SimpleNamespace(n_action_steps=None)

    def reset(self):
        self.reset_calls += 1

    def select_action(self, x):
        self.select_action_calls += 1
        return "raw_action_placeholder"


class TestPureHelpers(unittest.TestCase):
    def setUp(self):
        import task1_edge_pinch_v2_blind_eval_v1 as mod
        self.mod = importlib.reload(mod)

    def test_qaxis_identity_quaternion_returns_near_zero_vector(self):
        result = self.mod.qaxis([0.0, 0.0, 0.0, 1.0])
        self.assertTrue(np.allclose(result, np.zeros(3), atol=1e-6))

    def test_qaxis_90deg_about_x_has_expected_magnitude_and_axis(self):
        half = np.pi / 4
        q = [np.sin(half), 0.0, 0.0, np.cos(half)]  # 90deg about x
        result = self.mod.qaxis(q)
        self.assertAlmostEqual(float(np.linalg.norm(result)), np.pi / 2, places=5)
        self.assertAlmostEqual(float(result[0]), np.pi / 2, places=5)
        self.assertAlmostEqual(float(result[1]), 0.0, places=5)
        self.assertAlmostEqual(float(result[2]), 0.0, places=5)

    def test_validate_seed_allows_all_five_blind_eval_seeds(self):
        for seed in [555101, 555102, 555103, 555104, 555105]:
            self.mod.validate_seed(seed)  # must not raise

    def test_validate_seed_rejects_historical_555001_555005(self):
        for seed in [555001, 555002, 555003, 555004, 555005]:
            with self.assertRaises(ValueError):
                self.mod.validate_seed(seed)

    def test_validate_seed_rejects_training_collection_seeds(self):
        for seed in [950001, 950013, 950025]:
            with self.assertRaises(ValueError):
                self.mod.validate_seed(seed)

    def test_validate_seed_rejects_arbitrary_out_of_range_seed(self):
        with self.assertRaises(ValueError):
            self.mod.validate_seed(1)
        with self.assertRaises(ValueError):
            self.mod.validate_seed(555106)
        with self.assertRaises(ValueError):
            self.mod.validate_seed(555100)

    def test_validate_seed_allows_all_twenty_extended_batch2_seeds(self):
        for seed in range(555201, 555221):
            self.mod.validate_seed(seed)  # must not raise

    def test_validate_seed_rejects_gap_between_batch1_and_batch2(self):
        # 555106-555200 was never reviewed/allowed for either batch; must
        # stay rejected even though it sits between two allowed ranges.
        for seed in [555106, 555150, 555199, 555200]:
            with self.assertRaises(ValueError):
                self.mod.validate_seed(seed)

    def test_validate_seed_rejects_just_past_batch2_upper_bound(self):
        with self.assertRaises(ValueError):
            self.mod.validate_seed(555221)

    def test_batch1_seeds_still_allowed_after_batch2_addition(self):
        # Regression guard: extending ALLOWED_BLIND_EVAL_SEEDS must not
        # accidentally narrow or replace the original batch1 set.
        for seed in [555101, 555102, 555103, 555104, 555105]:
            self.mod.validate_seed(seed)

    def test_sha256_of_file_matches_hashlib_reference(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"task1 edge pinch blind eval test content")
            path = f.name
        try:
            expected = hashlib.sha256(open(path, "rb").read()).hexdigest()
            self.assertEqual(self.mod.sha256_of_file(path), expected)
        finally:
            os.unlink(path)

    def test_resolve_checkpoint_dir_missing_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                self.mod.resolve_checkpoint_dir(d, 500)

    def test_resolve_checkpoint_dir_found(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "000500" / "pretrained_model"
            target.mkdir(parents=True)
            resolved = self.mod.resolve_checkpoint_dir(d, 500)
            self.assertEqual(resolved, target)


class TestGates(unittest.TestCase):
    def setUp(self):
        import task1_edge_pinch_v2_blind_eval_v1 as mod
        self.mod = importlib.reload(mod)

    def _args(self, smoke_test=False, confirm_full_run=False):
        return types.SimpleNamespace(smoke_test=smoke_test, confirm_full_run=confirm_full_run)

    def test_gate_a_smoke_test_mode(self):
        self.assertEqual(self.mod.gate_a_resolve_mode(self._args(smoke_test=True)), "smoke_test")

    def test_gate_a_full_run_mode(self):
        self.assertEqual(
            self.mod.gate_a_resolve_mode(self._args(confirm_full_run=True)), "full_run"
        )

    def test_gate_a_both_flags_raises(self):
        with self.assertRaises(ValueError):
            self.mod.gate_a_resolve_mode(self._args(smoke_test=True, confirm_full_run=True))

    def test_gate_a_neither_flag_raises(self):
        with self.assertRaises(ValueError):
            self.mod.gate_a_resolve_mode(self._args())

    def test_gate_b_rejects_if_any_seed_invalid(self):
        with self.assertRaises(ValueError):
            self.mod.gate_b_validate_seeds([555101, 555001])  # one good, one blocked

    def test_gate_b_accepts_full_valid_list(self):
        self.mod.gate_b_validate_seeds([555101, 555102, 555103, 555104, 555105])  # no raise

    def test_gate_c_nonexistent_dir_ok(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "does_not_exist_yet"
            self.mod.gate_c_check_output_dir(target, overwrite=False)  # no raise

    def test_gate_c_empty_existing_dir_ok(self):
        with tempfile.TemporaryDirectory() as d:
            self.mod.gate_c_check_output_dir(Path(d), overwrite=False)  # no raise

    def test_gate_c_nonempty_without_overwrite_raises(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "existing.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                self.mod.gate_c_check_output_dir(Path(d), overwrite=False)

    def test_gate_c_nonempty_with_overwrite_ok(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "existing.json").write_text("{}")
            self.mod.gate_c_check_output_dir(Path(d), overwrite=True)  # no raise


class TestPolicyApiCompat(unittest.TestCase):
    def setUp(self):
        import task1_edge_pinch_v2_blind_eval_v1 as mod
        self.mod = importlib.reload(mod)

    def test_passes_for_compatible_fake_policy(self):
        self.mod.check_policy_api_compat(_FakePolicy(compatible=True))  # no raise

    def test_raises_if_reset_missing(self):
        policy = types.SimpleNamespace(
            config=types.SimpleNamespace(n_action_steps=None),
            select_action=lambda x: x,
        )  # deliberately no .reset attribute at all
        with self.assertRaises(RuntimeError):
            self.mod.check_policy_api_compat(policy)

    def test_raises_if_config_missing_n_action_steps(self):
        policy = _FakePolicy(compatible=True)
        policy.config = types.SimpleNamespace()  # no n_action_steps attr
        with self.assertRaises(RuntimeError):
            self.mod.check_policy_api_compat(policy)

    def test_raises_if_select_action_missing(self):
        policy = types.SimpleNamespace(
            config=types.SimpleNamespace(n_action_steps=None),
            reset=lambda: None,
        )  # deliberately no .select_action attribute at all
        with self.assertRaises(RuntimeError):
            self.mod.check_policy_api_compat(policy)


class TestRunOneBlindRollout(unittest.TestCase):
    def setUp(self):
        _install_fake_torch()
        import task1_edge_pinch_v2_blind_eval_v1 as mod
        self.mod = importlib.reload(mod)
        self.identity_pre = lambda x: x

    def test_post_called_exactly_once_per_control_step_not_twice(self):
        """Regression test for the double-post(raw)-call bug caught in
        review: post() must be invoked exactly once per control step, not
        once to check hasattr(...,"detach") and again to actually use the
        value."""
        env = _FakeEnv(
            step_outcomes=[(_obs(), 0.0, False)] * 3 + [(_obs(), 0.0, False)] * 5,
            success_on_step_indices=frozenset(),
        )
        policy = _FakePolicy(compatible=True)
        post = _counting_callable(lambda: _make_torch_like_action([0.0] * 7))

        result = self.mod.run_one_blind_rollout(
            policy, self.identity_pre, post, lambda: env,
            seed=555101, lang="pick up the akita black bowl and place it on the plate in the middle region",
            n_action_steps=25, wait_steps=3, max_steps=5,
        )
        # wait_steps=3 (no model/post calls) + max_steps=5 control steps
        # (none succeed, none invalid) -> exactly 5 post() calls.
        self.assertEqual(post.calls["n"], 5)
        self.assertEqual(result["steps_run"], 5)
        self.assertEqual(result["termination_reason"], "max_steps_reached")

    def test_post_called_exactly_once_when_post_returns_plain_ndarray(self):
        """Same regression guard, but exercising the branch where post()
        returns a plain numpy array with no .detach() (no double-call
        regardless of which branch is taken)."""
        env = _FakeEnv(step_outcomes=[(_obs(), 0.0, False)] * 3 + [(_obs(), 0.0, False)] * 2)
        policy = _FakePolicy(compatible=True)
        post = _counting_callable(lambda: np.zeros(7, dtype=np.float32))

        self.mod.run_one_blind_rollout(
            policy, self.identity_pre, post, lambda: env,
            seed=555102, lang="lang", n_action_steps=25, wait_steps=3, max_steps=2,
        )
        self.assertEqual(post.calls["n"], 2)

    def test_wait_phase_steps_not_counted_in_steps_run(self):
        env = _FakeEnv(step_outcomes=[(_obs(), 0.0, False)] * 10 + [(_obs(), 0.0, False)] * 4)
        policy = _FakePolicy(compatible=True)
        post = lambda raw: _make_torch_like_action([0.0] * 7)

        result = self.mod.run_one_blind_rollout(
            policy, self.identity_pre, post, lambda: env,
            seed=555103, lang="lang", n_action_steps=25, wait_steps=10, max_steps=4,
        )
        self.assertEqual(result["wait_steps"], 10)
        self.assertEqual(result["steps_run"], 4)  # only the 4 control steps, not 14
        self.assertEqual(len(env.step_calls), 14)  # but env.step() really was called 14 times total

    def test_wait_phase_zero_action_used(self):
        env = _FakeEnv(step_outcomes=[(_obs(), 0.0, False)] * 2 + [(_obs(), 0.0, False)] * 1)
        policy = _FakePolicy(compatible=True)
        post = lambda raw: _make_torch_like_action([1.0] * 7)

        self.mod.run_one_blind_rollout(
            policy, self.identity_pre, post, lambda: env,
            seed=555104, lang="lang", n_action_steps=25, wait_steps=2, max_steps=1,
        )
        # first 2 env.step() calls (the wait phase) must have used all-zero actions
        self.assertTrue(np.allclose(env.step_calls[0], np.zeros(7)))
        self.assertTrue(np.allclose(env.step_calls[1], np.zeros(7)))

    def test_wait_phase_done_short_circuits_before_any_policy_call(self):
        env = _FakeEnv(step_outcomes=[(_obs(), 0.0, True)])  # done on the very first (wait) step
        policy = _FakePolicy(compatible=True)
        post = _counting_callable(lambda: _make_torch_like_action([0.0] * 7))

        result = self.mod.run_one_blind_rollout(
            policy, self.identity_pre, post, lambda: env,
            seed=555105, lang="lang", n_action_steps=25, wait_steps=5, max_steps=100,
        )
        self.assertTrue(result["wait_terminated"])
        self.assertEqual(result["termination_reason"], "wait_phase_done")
        self.assertEqual(result["steps_run"], 0)
        self.assertEqual(policy.reset_calls, 0)  # policy.reset() never called
        self.assertEqual(post.calls["n"], 0)  # select_action/post never called

    def test_success_stops_rollout_early_and_records_reason(self):
        env = _FakeEnv(
            step_outcomes=[(_obs(), 0.0, False)]  # 1 wait step
            + [(_obs(), 0.0, False), (_obs(), 1.0, False), (_obs(), 0.0, False)],  # 3 control steps available
            success_on_step_indices=frozenset({2}),  # success on the 2nd control step (global index 2)
        )
        policy = _FakePolicy(compatible=True)
        post = lambda raw: _make_torch_like_action([0.0] * 7)

        result = self.mod.run_one_blind_rollout(
            policy, self.identity_pre, post, lambda: env,
            seed=555101, lang="lang", n_action_steps=25, wait_steps=1, max_steps=10,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["termination_reason"], "success_early_termination")
        self.assertEqual(result["steps_run"], 2)  # stopped after the 2nd control step, not all 3 available
        self.assertEqual(result["reward_max"], 1.0)

    def test_invalid_action_shape_flagged_and_stops(self):
        env = _FakeEnv(step_outcomes=[(_obs(), 0.0, False)])
        policy = _FakePolicy(compatible=True)
        post = lambda raw: _make_torch_like_action([0.0] * 6)  # wrong shape: 6, not 7

        result = self.mod.run_one_blind_rollout(
            policy, self.identity_pre, post, lambda: env,
            seed=555102, lang="lang", n_action_steps=25, wait_steps=0, max_steps=5,
        )
        self.assertTrue(result["invalid_action_detected"])
        self.assertEqual(result["termination_reason"], "invalid_action")
        self.assertEqual(result["steps_run"], 0)

    def test_invalid_action_nan_flagged(self):
        env = _FakeEnv(step_outcomes=[(_obs(), 0.0, False)])
        policy = _FakePolicy(compatible=True)
        post = lambda raw: _make_torch_like_action([float("nan")] * 7)

        result = self.mod.run_one_blind_rollout(
            policy, self.identity_pre, post, lambda: env,
            seed=555103, lang="lang", n_action_steps=25, wait_steps=0, max_steps=5,
        )
        self.assertTrue(result["invalid_action_detected"])

    def test_env_close_called_even_when_rollout_raises(self):
        class _RaisingEnv(_FakeEnv):
            def step(self, action):
                raise RuntimeError("simulated env failure")

        env = _RaisingEnv(step_outcomes=[])
        policy = _FakePolicy(compatible=True)
        post = lambda raw: _make_torch_like_action([0.0] * 7)

        with self.assertRaises(RuntimeError):
            self.mod.run_one_blind_rollout(
                policy, self.identity_pre, post, lambda: env,
                seed=555101, lang="lang", n_action_steps=25, wait_steps=1, max_steps=5,
            )
        self.assertTrue(env.closed)  # try/finally must still have closed it

    def test_blocked_seed_rejected_before_env_is_even_constructed(self):
        env_fn_calls = {"n": 0}

        def env_fn():
            env_fn_calls["n"] += 1
            return _FakeEnv(step_outcomes=[(_obs(), 0.0, False)] * 20)

        policy = _FakePolicy(compatible=True)
        post = lambda raw: _make_torch_like_action([0.0] * 7)

        with self.assertRaises(ValueError):
            self.mod.run_one_blind_rollout(
                policy, self.identity_pre, post, env_fn,
                seed=555001, lang="lang", n_action_steps=25, wait_steps=1, max_steps=5,
            )
        self.assertEqual(env_fn_calls["n"], 0)  # env must never be constructed for a blocked seed

    def test_policy_config_n_action_steps_set_before_rollout(self):
        env = _FakeEnv(step_outcomes=[(_obs(), 0.0, False)] * 1 + [(_obs(), 0.0, False)] * 2)
        policy = _FakePolicy(compatible=True)
        post = lambda raw: _make_torch_like_action([0.0] * 7)

        self.mod.run_one_blind_rollout(
            policy, self.identity_pre, post, lambda: env,
            seed=555104, lang="lang", n_action_steps=25, wait_steps=1, max_steps=2,
        )
        self.assertEqual(policy.config.n_action_steps, 25)

    def test_policy_reset_called_once_per_rollout(self):
        env = _FakeEnv(step_outcomes=[(_obs(), 0.0, False)] * 1 + [(_obs(), 0.0, False)] * 2)
        policy = _FakePolicy(compatible=True)
        post = lambda raw: _make_torch_like_action([0.0] * 7)

        self.mod.run_one_blind_rollout(
            policy, self.identity_pre, post, lambda: env,
            seed=555105, lang="lang", n_action_steps=25, wait_steps=1, max_steps=2,
        )
        self.assertEqual(policy.reset_calls, 1)


class TestFrame(unittest.TestCase):
    def setUp(self):
        _install_fake_torch()
        import task1_edge_pinch_v2_blind_eval_v1 as mod
        self.mod = importlib.reload(mod)

    def test_frame_does_not_require_or_touch_object_pose_keys(self):
        obs = _obs()
        obs["akita_black_bowl_1_pos"] = np.array([9.9, 9.9, 9.9])  # privileged field, must be ignored
        obs["plate_1_to_robot0_eef_pos"] = np.array([9.9, 9.9, 9.9])
        result = self.mod.frame(obs, "pick up the akita black bowl and place it on the plate in the middle region")
        self.assertIn("observation.state", result)
        self.assertIn("observation.images.agentview", result)
        self.assertIn("observation.images.wrist", result)
        self.assertEqual(result["task"], "pick up the akita black bowl and place it on the plate in the middle region")

    def test_frame_raises_if_a_required_proprioception_key_missing(self):
        obs = _obs()
        del obs["robot0_eef_quat"]
        with self.assertRaises(KeyError):
            self.mod.frame(obs, "lang")

    def test_frame_state_vector_is_15_dim_in_expected_order(self):
        obs = _obs()
        obs["robot0_joint_pos"] = np.arange(7, dtype=np.float64)
        obs["robot0_eef_pos"] = np.array([100.0, 200.0, 300.0])
        obs["robot0_gripper_qpos"] = np.array([9.0, 9.0])
        result = self.mod.frame(obs, "lang")
        state = np.asarray(result["observation.state"])
        self.assertEqual(state.shape, (15,))
        self.assertTrue(np.allclose(state[0:7], np.arange(7)))
        self.assertTrue(np.allclose(state[7:10], [100.0, 200.0, 300.0]))
        self.assertTrue(np.allclose(state[13:15], [9.0, 9.0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
