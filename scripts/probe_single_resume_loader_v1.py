"""New registered single-task sampling, not historical cursor reconstruction."""
import hashlib
import json
from pathlib import Path
import draccus
import torch
import lerobot.scripts.lerobot_train
from accelerate import Accelerator
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from torch.utils.data import DataLoader, RandomSampler, BatchSampler


def main():
    path = Path("/root/smolvla-training-prep/results/training/teacher_native_spatial_task0_single_current_recipe_40k_batch2_v1/checkpoints/020000/pretrained_model/train_config.json")
    cfg = draccus.decode(TrainPipelineConfig, json.loads(path.read_text()))
    assert cfg.num_workers == 4 and not cfg.dataset.image_transforms.enable
    dataset = make_dataset(cfg)
    tasks = dataset.hf_dataset.data.column("task_index").to_pylist()
    assert len(dataset) == 5068 and set(tasks) == {0}
    state = torch.get_rng_state().clone()
    def sequence():
        generator = torch.Generator().manual_seed(3000)
        batches = []
        while len(batches) < 20000:
            batches.extend(BatchSampler(RandomSampler(dataset, generator=generator),
                                        batch_size=2, drop_last=False))
        return batches[:20000]
    batches = sequence()
    assert batches == sequence()
    assert len(batches) == 20000 and sum(map(len, batches)) == 40000
    assert all(len(b) == 2 and all(0 <= i < 5068 for i in b) for b in batches)
    assert torch.equal(state, torch.get_rng_state())
    class Batches:
        batch_size = 2
        drop_last = False
        def __len__(self):
            return len(batches)
        def __iter__(self):
            return iter(batches)
    accelerator = Accelerator(cpu=True)
    assert accelerator.num_processes == 1
    arms = []
    for _ in range(2):
        loader = accelerator.prepare(DataLoader(dataset, batch_sampler=Batches(),
                      num_workers=4, generator=torch.Generator().manual_seed(2000),
                      prefetch_factor=2, pin_memory=False))
        iterator = iter(loader)
        arms.append([next(iterator) for _ in range(8)])
        del iterator, loader
        assert torch.equal(state, torch.get_rng_state())
    fields = set()
    for offset, (left, right) in enumerate(zip(*arms)):
        assert left.keys() == right.keys()
        assert left["index"].tolist() == batches[offset]
        assert left["task_index"].tolist() == [0, 0]
        for key in left:
            fields.add(key)
            assert torch.equal(left[key], right[key]) if isinstance(left[key], torch.Tensor) else left[key] == right[key]
    print(json.dumps({"status": "new_single_resume_sampling_cpu_passed",
                      "remaining_batches": 20000, "sample_draws": 40000,
                      "sampling_seed": 3000, "worker_generator_seed": 2000,
                      "sequence_sha256": hashlib.sha256(json.dumps(batches,separators=(",", ":")).encode()).hexdigest(),
                      "decoded_batches_per_arm": 8, "fields_compared": sorted(fields),
                      "main_rng_preserved": True, "historical_cursor_reconstructed": False}))


if __name__ == "__main__":
    main()
