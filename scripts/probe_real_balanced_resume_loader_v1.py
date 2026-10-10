"""CPU real-data suffix and Accelerator probe, no model or training."""
import hashlib
import json
from collections import Counter
from itertools import islice
from pathlib import Path
import draccus
import torch
import lerobot.scripts.lerobot_train
from accelerate import Accelerator
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from torch.utils.data import DataLoader
from exposure_matched_blocks_v1 import balanced_batches


def main():
    path = Path("/root/smolvla-training-prep/results/training/teacher_native_spatial_4task_homogeneous_160k_batch2_v1_20261009/checkpoints/080000/pretrained_model/train_config.json")
    cfg = draccus.decode(TrainPipelineConfig, json.loads(path.read_text()))
    assert cfg.num_workers == 4 and not cfg.dataset.image_transforms.enable
    dataset = make_dataset(cfg)
    tasks = dataset.hf_dataset.data.column("task_index").to_pylist()
    pools = {t: [] for t in range(4)}
    for index, task in enumerate(tasks):
        pools[int(task)].append(index)
    assert [len(pools[t]) for t in range(4)] == [5068, 6707, 5882, 5052]
    def sequence():
        return balanced_batches(pools, blocks=40000, batch_size=2, seed=1000)
    suffix = list(islice(sequence(), 80000, None))
    assert suffix == list(sequence())[80000:]
    assert suffix == list(islice(sequence(), 80000, None))
    assert Counter(t for _, t, _ in suffix) == {t: 20000 for t in range(4)}
    for offset in range(0, len(suffix), 4):
        assert {t for _, t, _ in suffix[offset:offset+4]} == set(range(4))
    assert all(len(batch) == 2 and all(int(tasks[i]) == t for i in batch)
               for _, t, batch in suffix)
    class Batches:
        batch_size = 2
        drop_last = True
        def __len__(self):
            return len(suffix)
        def __iter__(self):
            return (batch for _, _, batch in suffix)
    accelerator = Accelerator(cpu=True)
    assert accelerator.num_processes == 1
    state = torch.get_rng_state().clone()
    arms = []
    for _ in range(2):
        loader = DataLoader(dataset, batch_sampler=Batches(), num_workers=4,
                            generator=torch.Generator().manual_seed(2000),
                            prefetch_factor=2, pin_memory=False)
        loader = accelerator.prepare(loader)
        iterator = iter(loader)
        arms.append([next(iterator) for _ in range(8)])
        del iterator, loader
        assert torch.equal(state, torch.get_rng_state())
    fields = set()
    for offset, (left, right) in enumerate(zip(*arms)):
        assert left.keys() == right.keys()
        assert left["task_index"].tolist() == [suffix[offset][1]] * 2
        assert left["index"].tolist() == suffix[offset][2]
        for key in left:
            fields.add(key)
            assert torch.equal(left[key], right[key]) if isinstance(left[key], torch.Tensor) else left[key] == right[key]
    print(json.dumps({"status": "real_balanced_suffix_accelerator_cpu_passed",
                      "remaining_batches": len(suffix), "updates_per_task": 20000,
                      "sample_draws_per_task": 40000, "decoded_batches_per_arm": 8,
                      "fields_compared": sorted(fields), "main_rng_preserved": True,
                      "suffix_sha256": hashlib.sha256(json.dumps(suffix, separators=(",", ":")).encode()).hexdigest(),
                      "gpu_or_policy_used": False}))


if __name__ == "__main__":
    main()
