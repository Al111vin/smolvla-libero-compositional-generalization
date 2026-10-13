from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import replay_fresh_seed_init8_cpu_reset_v1 as runner  # noqa: E402


class FreshSeedInit8ResetSmokeTests(unittest.TestCase):
    def test_scope_is_exactly_the_registered_init8_pair(self) -> None:
        self.assertEqual(runner.selected_trace_keys(), (
            "single_task0_paired_8_r0_160k",
            "joint_task0_paired_8_r0_160k",
        ))

    def test_manifest_hash_is_canonical_and_self_excluding(self) -> None:
        record = {"schema_version": 1, "trace_keys": list(runner.TRACE_KEYS)}
        record["manifest_sha256"] = runner.canonical_sha(record, "manifest_sha256")
        self.assertEqual(runner.canonical_sha(record, "manifest_sha256"), record["manifest_sha256"])
        record["trace_keys"].reverse()
        self.assertNotEqual(runner.canonical_sha(record, "manifest_sha256"), record["manifest_sha256"])

    def test_build_manifest_pins_runner_helpers_schedules_and_tests(self) -> None:
        source = {"manifest_sha256": runner.REGISTRATION_MANIFEST_SHA}
        selected = {key: {"trace": {"key": key}, "row": {"init": 8}}
                    for key in runner.TRACE_KEYS}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            with (mock.patch.object(runner, "MANIFEST_PATH", path),
                  mock.patch.object(runner, "_source_rows", return_value=(source, selected))):
                manifest = runner.build_manifest(path)
            self.assertEqual(manifest["manifest_sha256"],
                             runner.canonical_sha(manifest, "manifest_sha256"))
            self.assertEqual(manifest["runner_sha256"], manifest["code_sha256"]["runner"])
            self.assertIn("unit_tests", manifest["code_sha256"])
            self.assertEqual(set(manifest["code_sha256"]), set(runner._code_paths()))

    def test_run_requires_explicit_execution_before_any_remote_path_access(self) -> None:
        with self.assertRaisesRegex(runner.ReplayError, "Explicit --authorize-execution"):
            runner.run(Path("/missing/manifest.json"), Path("/missing/output"), False)

    def test_execution_masks_all_cuda_devices(self) -> None:
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}):
            runner._force_cpu_only()
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")

    def test_worker_reseeds_before_constructing_a_fresh_environment_each_call(self) -> None:
        events: list[tuple[str, object]] = []
        env_instances: list[object] = []

        class FakeEnv:
            def __init__(self, **kwargs):
                events.append(("construct", dict(kwargs)))
                env_instances.append(self)

            def close(self):
                events.append(("close", id(self)))

        class Suite:
            def get_task(self, task_id):
                return types.SimpleNamespace(problem_folder="problem", bddl_file="task.bddl")

            def get_task_init_states(self, task_id):
                return list(range(20))

        class Benchmark:
            @staticmethod
            def get_benchmark(name):
                return lambda: Suite()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bddl = root / "problem" / "task.bddl"
            bddl.parent.mkdir()
            bddl.write_text("synthetic", encoding="utf-8")
            output = root / "out"
            output.mkdir()
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")

            libero_pkg = types.ModuleType("libero")
            libero_pkg.__path__ = []
            libero_mod = types.ModuleType("libero.libero")
            libero_mod.__path__ = []
            libero_mod.benchmark = Benchmark
            libero_mod.get_libero_path = lambda _: str(root)
            envs_mod = types.ModuleType("libero.libero.envs")
            envs_mod.__path__ = []
            wrapper_mod = types.ModuleType("libero.libero.envs.env_wrapper")
            wrapper_mod.ControlEnv = FakeEnv
            fake_numpy = types.ModuleType("numpy")
            fake_numpy.random = types.SimpleNamespace(
                seed=lambda seed: events.append(("numpy_seed", seed))
            )

            fake_rows = {
                key: {"trace": {"key": key}, "row": {"init": 8}}
                for key in runner.TRACE_KEYS
            }

            def replay(fake_env, initial_states, trace, row):
                events.append(("replay", trace["key"]))
                self.assertIs(fake_env, env_instances[-1])
                return {"key": trace["key"], "replay_success": True}

            with (
                mock.patch.object(runner, "OUTPUT_ROOT", output),
                mock.patch.object(runner, "MANIFEST_PATH", manifest),
                mock.patch.object(runner, "_load_run_manifest", return_value={}),
                mock.patch.object(runner, "_source_rows", return_value=({"manifest_sha256": "fake-source-sha"}, fake_rows)),
                mock.patch.object(runner, "require_gpu_idle", return_value={"utilization_percent": [0]}),
                mock.patch.object(runner.subprocess, "check_output", return_value="PID ARGS\n"),
                mock.patch.object(runner, "replay_trace", side_effect=replay),
                mock.patch.object(runner.random, "seed", side_effect=lambda seed: events.append(("python_seed", seed))),
                mock.patch.dict(sys.modules, {
                    "numpy": fake_numpy,
                    "libero": libero_pkg,
                    "libero.libero": libero_mod,
                    "libero.libero.envs": envs_mod,
                    "libero.libero.envs.env_wrapper": wrapper_mod,
                }),
            ):
                for key in runner.TRACE_KEYS:
                    runner.worker(key, output, manifest)

            self.assertEqual(len(env_instances), 2)
            self.assertIsNot(env_instances[0], env_instances[1])
            constructions = [i for i, item in enumerate(events) if item[0] == "construct"]
            for construction in constructions:
                prior = [item[0] for item in events[:construction]]
                self.assertEqual(prior[-2:], ["python_seed", "numpy_seed"])
            self.assertEqual(sorted(p.name for p in output.glob("*.json")), sorted(
                [f"{key}.json" for key in runner.TRACE_KEYS]
            ))
            self.assertEqual(list(output.glob("*.log")), [])
            for key in runner.TRACE_KEYS:
                data = json.loads((output / f"{key}.json").read_text())
                self.assertEqual(data["preconstruction_python_numpy_seed"], runner.ENVIRONMENT_SEED)
                self.assertEqual(data["policy_calls"], 0)
                self.assertEqual(data["new_rollouts"], 0)


if __name__ == "__main__":
    unittest.main()
