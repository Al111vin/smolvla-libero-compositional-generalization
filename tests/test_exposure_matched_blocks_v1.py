import unittest
from collections import Counter
from scripts.exposure_matched_blocks_v1 import balanced_batches, baseline_lr_for_update


class ExposureTests(unittest.TestCase):
    def test_full_budget(self):
        pools = {t: list(range(t * 10, t * 10 + t + 3)) for t in range(4)}
        counts = Counter()
        lr_sums = Counter()
        reference = [float(k) / 40000 for k in range(40000)]
        seen = []
        for update, (block, task, batch) in enumerate(balanced_batches(
                pools, blocks=40000, batch_size=2, seed=12345)):
            self.assertEqual(len(batch), 2)
            self.assertTrue(set(batch) <= set(pools[task]))
            counts[task] += len(batch)
            lr_sums[task] += 2 * baseline_lr_for_update(reference, update, 4)
            seen.append(task)
            if update % 4 == 3:
                self.assertEqual(set(seen), set(pools))
                seen = []
            self.assertEqual(block, update // 4)
        self.assertEqual(update + 1, 160000)
        self.assertEqual(dict(counts), {t: 80000 for t in pools})
        self.assertEqual(len(set(lr_sums.values())), 1)

    def test_deterministic(self):
        kwargs = dict(blocks=5, batch_size=2, seed=7)
        pools = {0: [0, 1, 2], 1: [3, 4]}
        self.assertEqual(list(balanced_batches(pools, **kwargs)),
                         list(balanced_batches(pools, **kwargs)))

    def test_bounds(self):
        self.assertEqual(baseline_lr_for_update([0.1, 0.2], 7, 4), 0.2)
        with self.assertRaises(ValueError):
            baseline_lr_for_update([0.1, 0.2], 8, 4)
        with self.assertRaises(ValueError):
            list(balanced_batches({0: [1], 1: [1]}, blocks=1, batch_size=2, seed=1))


if __name__ == "__main__":
    unittest.main()
