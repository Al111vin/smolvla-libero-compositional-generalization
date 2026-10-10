"""Finite paired-resume batch budgets; no training launcher."""
from itertools import islice
from exposure_matched_blocks_v1 import balanced_batches


class JointResumeBatchSampler:
    batch_size = 2
    drop_last = True

    def __init__(self, pools, *, restored_step, expected_step=80000,
                 total_steps=160000, seed=1000):
        if (restored_step != expected_step or expected_step != 80000
                or total_steps != 160000 or seed != 1000):
            raise ValueError("unregistered resume boundary or budget")
        if tuple(sorted(pools)) != (0, 1, 2, 3):
            raise ValueError("registered four-task pools required")
        self.pools = {t: tuple(pools[t]) for t in sorted(pools)}
        flat = [i for p in self.pools.values() for i in p]
        if (any(not p for p in self.pools.values()) or len(set(flat)) != len(flat)
                or any(type(i) is not int or i < 0 for i in flat)):
            raise ValueError("empty, duplicate or invalid dataset indices")
        self.restored_step = restored_step
        self.total_steps = total_steps
        self.seed = seed

    def __len__(self):
        return self.total_steps - self.restored_step

    def __iter__(self):
        sequence = balanced_batches(self.pools, blocks=40000,
                                    batch_size=2, seed=self.seed)
        for _, _, batch in islice(sequence, self.restored_step, self.total_steps):
            yield batch


class SingleResumeBatchSampler:
    batch_size = 2
    drop_last = False

    def __init__(self, dataset, *, restored_step, expected_step=20000,
                 total_steps=40000, seed=3000):
        if (restored_step != expected_step or expected_step != 20000
                or total_steps != 40000 or seed != 3000 or len(dataset) != 5068):
            raise ValueError("unregistered single-task resume protocol")
        self.dataset = dataset
        self.remaining = total_steps - restored_step
        self.seed = seed

    def __len__(self):
        return self.remaining

    def __iter__(self):
        import torch
        from torch.utils.data import BatchSampler, RandomSampler
        generator = torch.Generator().manual_seed(self.seed)
        emitted = 0
        while emitted < self.remaining:
            sampler = BatchSampler(RandomSampler(self.dataset, generator=generator),
                                   batch_size=2, drop_last=False)
            for batch in sampler:
                yield batch
                emitted += 1
                if emitted == self.remaining:
                    return
