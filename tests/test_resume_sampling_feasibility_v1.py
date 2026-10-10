"""Synthetic CPU evidence only: not an actual dataset/resume integration test."""
import random
import sys
import unittest
from collections import Counter
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from exposure_matched_blocks_v1 import balanced_batches


class ResumeSamplingFeasibility(unittest.TestCase):
    def test_full_remaining_joint_budget_exact_suffix_and_exposure(self):
        pools = {t: list(range(t * 17, (t + 1) * 17)) for t in range(4)}
        def sequence():
            return balanced_batches(pools, blocks=40000, batch_size=2, seed=1000)
        uninterrupted = list(sequence())
        original_lr_arm = list(islice(sequence(), 80000, None))
        half_lr_arm = list(islice(sequence(), 80000, None))
        self.assertEqual(original_lr_arm, uninterrupted[80000:])
        self.assertEqual(original_lr_arm, half_lr_arm)
        self.assertEqual(len(original_lr_arm), 80000)
        self.assertEqual(Counter(t for _, t, _ in original_lr_arm),
                         {t: 20000 for t in range(4)})
        for offset in range(0, 80000, 4):
            self.assertEqual({t for _, t, _ in original_lr_arm[offset:offset + 4]},
                             set(range(4)))

    def test_sampler_does_not_consume_global_python_rng(self):
        before = random.getstate()
        list(balanced_batches({0: [0, 1], 1: [2, 3]},
                              blocks=100, batch_size=2, seed=1000))
        self.assertEqual(before, random.getstate())

    def test_naive_restart_is_not_continuation(self):
        pools = {t: list(range(t * 17, (t + 1) * 17)) for t in range(4)}
        seq = list(balanced_batches(pools, blocks=40, batch_size=2, seed=1000))
        self.assertNotEqual(seq[:80], seq[80:])


if __name__ == "__main__":
    unittest.main()
