import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Seed2002WrapperTests(unittest.TestCase):
    def test_single_wrapper_is_pinned_to_seed2002_config(self):
        source = (ROOT / "scripts/fresh_seed2002_task0_train_wrapper_v1.py").read_text()
        self.assertIn("teacher_native_spatial_task0_freshseed2002_40k_batch2_v1_20261014.json", source)
        self.assertIn("ae600eb68936841ffeb8fa24defe0fcbcd217dc7d8c6a6a508ece31f8d46133d", source)
        self.assertIn("cfg.seed != 2002", source)
        ast.parse(source)

    def test_joint_wrapper_pins_model_and_sampler_seed2002(self):
        source = (ROOT / "scripts/fresh_seed2002_joint4_train_wrapper_v1.py").read_text()
        self.assertIn("teacher_native_spatial_4task_freshseed2002_160k_batch2_v1_20261014.json", source)
        self.assertIn("3c0295b09055878045cd2c17a24dee41b6a397979e6523206f35d7f696da04c4", source)
        self.assertIn("cfg.seed != 2002", source)
        self.assertIn("seed=2002", source)
        self.assertIn("cfg.steps != BLOCKS * TASKS", source)
        self.assertIn("cfg.batch_size != 2", source)
        ast.parse(source)


if __name__ == "__main__":
    unittest.main()
