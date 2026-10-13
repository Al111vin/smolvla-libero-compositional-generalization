from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import replay_fresh_seed_init8_cpu_reset_v5 as runner  # noqa: E402


class FreshSeedInit8ResetV5WrapperTests(unittest.TestCase):
    def test_new_paths_and_identity_are_independent(self) -> None:
        self.assertEqual(runner.EXPERIMENT_ID,
                         "teacher_native_spatial_fresh_seed_init8_cpu_reset_v5_20261013")
        self.assertIn("_v5_20261013", str(runner.MANIFEST_PATH))
        self.assertIn("_v5_20261013", str(runner.OUTPUT_ROOT))
        self.assertIn("_v5_20261013", str(runner.base.MANIFEST_PATH))
        self.assertIn("_v5_20261013", str(runner.base.OUTPUT_ROOT))

    def test_only_the_registered_pair_and_zero_gpu_budget_are_inherited(self) -> None:
        self.assertEqual(runner.base.selected_trace_keys(), (
            "single_task0_paired_8_r0_160k",
            "joint_task0_paired_8_r0_160k",
        ))
        self.assertEqual(runner.base.MAX_TOTAL_SECONDS, 300)
        self.assertEqual(runner.base.WORKER_TIMEOUT_SECONDS, 120)

    def test_complete_code_inventory_includes_both_runners_and_evaluator(self) -> None:
        paths = runner._code_paths()
        self.assertIn("base_runner", paths)
        self.assertIn("frozen_evaluator", paths)
        self.assertIn("unit_tests", paths)
        hashes = runner._code_hashes()
        self.assertEqual(set(paths), set(hashes))
        self.assertEqual(hashes["runner"], runner.base.sha256_file(Path(__file__).resolve().parents[1] / "scripts" / "replay_fresh_seed_init8_cpu_reset_v5.py"))
        self.assertEqual(hashes["frozen_evaluator"],
                         "89fd36a89dc45a56382219d4c3f6e5d12a3b1689abb3440ec29be346815af7cb")

    def test_v4_runner_is_not_modified_by_path_overlay(self) -> None:
        self.assertEqual(runner._code_hashes()["base_runner"],
                         "a14c1979cd18f8422167404f90415a0b1b964d90c03a128dd762355bf07e5b0f")


if __name__ == "__main__":
    unittest.main()
