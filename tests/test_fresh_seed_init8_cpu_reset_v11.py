from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import replay_fresh_seed_init8_cpu_reset_v11 as runner  # noqa: E402
import replay_fresh_seed_init8_cpu_reset_v4 as engine  # noqa: E402


class V11RegistrationTests(unittest.TestCase):
    def test_new_unique_identity_paths_and_exact_scope(self) -> None:
        self.assertEqual(runner.EXPERIMENT_ID, "teacher_native_spatial_fresh_seed_init8_cpu_reset_v11_20261014")
        self.assertIn("_v11_", str(runner.MANIFEST_PATH))
        self.assertIn("_v11_", str(runner.OUTPUT_ROOT))
        self.assertEqual(runner._code_hashes()["runner"], engine.sha256_file(Path(runner.__file__).resolve()))
        self.assertEqual(runner.engine.selected_trace_keys(), (
            "single_task0_paired_8_r0_160k", "joint_task0_paired_8_r0_160k"))
        self.assertNotIn("prior_failed_runner_v9", runner._code_paths())
        self.assertNotIn("prior_failed_runner_v10", runner._code_paths())

    def test_builder_pins_fresh_v11_inventory_and_zero_compute(self) -> None:
        source = {"manifest_sha256": engine.REGISTRATION_MANIFEST_SHA}
        selected = {key: {"trace": {"key": key}, "row": {"init": 8}}
                    for key in engine.TRACE_KEYS}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            with (mock.patch.object(engine, "MANIFEST_PATH", path),
                  mock.patch.object(engine, "_source_rows", return_value=(source, selected))):
                payload = runner.engine.build_manifest(path)
            self.assertEqual(payload["experiment_id"], runner.EXPERIMENT_ID)
            self.assertEqual(payload["runner_sha256"], payload["code_sha256"]["runner"])
            self.assertEqual(set(payload["code_sha256"]), set(runner._code_paths()))
            self.assertEqual(payload["gpu_budget"], 0)
            self.assertEqual(payload["policy_calls"], 0)
            self.assertEqual(payload["new_rollouts"], 0)
            self.assertEqual(payload["training_updates"], 0)

    def test_no_execution_without_explicit_flag(self) -> None:
        with self.assertRaisesRegex(engine.ReplayError, "Explicit --authorize-execution"):
            runner.engine.run(Path("/missing"), Path("/missing"), False)


if __name__ == "__main__":
    unittest.main()
