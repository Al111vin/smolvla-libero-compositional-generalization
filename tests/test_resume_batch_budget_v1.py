import sys
import unittest
from pathlib import Path
from itertools import islice
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from exposure_matched_blocks_v1 import balanced_batches
from resume_batch_budget_v1 import JointResumeBatchSampler, SingleResumeBatchSampler


class ResumeBudgetTests(unittest.TestCase):
    def setUp(self):
        self.pools = {t: list(range(t * 7, (t + 1) * 7)) for t in range(4)}

    def test_joint_full_suffix_and_finite_exhaustion(self):
        sampler = JointResumeBatchSampler(self.pools, restored_step=80000)
        expected = [b for _, _, b in islice(balanced_batches(
            self.pools, blocks=40000, batch_size=2, seed=1000), 80000, None)]
        actual = list(sampler)
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), len(sampler))
        self.assertEqual(actual, list(sampler))

    def test_joint_fail_closed(self):
        for options in ({"restored_step": 0}, {"restored_step": 80001},
                        {"restored_step": 80000, "total_steps": 320000},
                        {"restored_step": 80000, "seed": 1001}):
            with self.assertRaises(ValueError):
                JointResumeBatchSampler(self.pools, **options)
        for pools in ({0: [0]}, {**self.pools, 3: []},
                      {**self.pools, 3: [0]}, {**self.pools, 3: [-1]}):
            with self.assertRaises(ValueError):
                JointResumeBatchSampler(pools, restored_step=80000)

    def test_single_invalid_protocol_before_torch_import(self):
        for dataset, options in ((range(5067), {"restored_step": 20000}),
                                 (range(5068), {"restored_step": 0}),
                                 (range(5068), {"restored_step": 20000, "seed": 1000})):
            with self.assertRaises(ValueError):
                SingleResumeBatchSampler(dataset, **options)


if __name__ == "__main__":
    unittest.main()
