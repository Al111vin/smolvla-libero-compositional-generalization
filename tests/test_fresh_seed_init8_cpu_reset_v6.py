from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import replay_fresh_seed_init8_cpu_reset_v6 as runner  # noqa: E402


class FreshSeedInit8ResetV6WrapperTests(unittest.TestCase):
    def test_new_paths_and_identity_are_independent(self) -> None:
        self.assertEqual(runner.EXPERIMENT_ID,
                         "teacher_native_spatial_fresh_seed_init8_cpu_reset_v6_20261013")
        self.assertIn("_v6_20261013", str(runner.MANIFEST_PATH))
        self.assertIn("_v6_20261013", str(runner.OUTPUT_ROOT))
        self.assertIn("_v6_20261013", str(runner.base.MANIFEST_PATH))
        self.assertIn("_v6_20261013", str(runner.base.OUTPUT_ROOT))
        self.assertIn("_v6_20261013", str(runner.base.base.MANIFEST_PATH))
        self.assertIn("_v6_20261013", str(runner.base.base.OUTPUT_ROOT))

    def test_trace_scope_and_execution_bounds_are_inherited(self) -> None:
        self.assertEqual(runner.base.base.selected_trace_keys(), (
            "single_task0_paired_8_r0_160k",
            "joint_task0_paired_8_r0_160k",
        ))
        self.assertEqual(runner.base.base.MAX_TOTAL_SECONDS, 300)
        self.assertEqual(runner.base.base.WORKER_TIMEOUT_SECONDS, 120)

    def test_inventory_pins_both_prior_wrappers_and_frozen_evaluator(self) -> None:
        paths = runner._code_paths()
        hashes = runner._code_hashes()
        self.assertEqual(set(paths), set(hashes))
        self.assertIn("base_runner_v5", paths)
        self.assertIn("base_runner", paths)
        self.assertIn("frozen_evaluator", paths)
        self.assertIn("unit_tests", paths)
        self.assertEqual(hashes["base_runner_v5"],
                         "fc062193eae93ef7348b96aad73de6467d34f65ffd5cbe73ed1dbf636671d954")
        self.assertEqual(hashes["base_runner"],
                         "a14c1979cd18f8422167404f90415a0b1b964d90c03a128dd762355bf07e5b0f")
        self.assertEqual(hashes["frozen_evaluator"],
                         "89fd36a89dc45a56382219d4c3f6e5d12a3b1689abb3440ec29be346815af7cb")


if __name__ == "__main__":
    unittest.main()
