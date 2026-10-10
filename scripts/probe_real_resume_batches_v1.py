"""Bounded actual-data CPU probe, not a resume or training launcher."""
import json
from pathlib import Path
import torch
import draccus
import lerobot.scripts.lerobot_train  # registers SmolVLA
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from torch.utils.data import DataLoader


def main():
    path = Path("/root/smolvla-training-prep/results/training/teacher_native_spatial_4task_homogeneous_160k_batch2_v1_20261009/checkpoints/080000/pretrained_model/train_config.json")
    cfg = draccus.decode(TrainPipelineConfig, json.loads(path.read_text()))
    assert cfg.num_workers == 4 and not cfg.dataset.image_transforms.enable
    dataset = make_dataset(cfg)
    torch.manual_seed(1000)
    state = torch.get_rng_state().clone()
    arms = []
    for _ in range(2):
        loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4,
                            generator=torch.Generator().manual_seed(2000),
                            prefetch_factor=2, drop_last=False, pin_memory=False)
        iterator = iter(loader)
        batches = [next(iterator) for _ in range(8)]
        del iterator, loader
        assert torch.equal(state, torch.get_rng_state()), "main RNG changed"
        arms.append(batches)
    fields = set()
    for left, right in zip(*arms):
        assert left.keys() == right.keys()
        for key in left:
            fields.add(key)
            if isinstance(left[key], torch.Tensor):
                assert torch.equal(left[key], right[key]), key
            else:
                assert left[key] == right[key], key
    print(json.dumps({"status": "actual_data_eight_batches_equal_cpu_only",
                      "fields_compared": sorted(fields), "workers": 4,
                      "global_torch_rng_preserved": True,
                      "homogeneous_sampler_or_accelerator_tested": False}))


if __name__ == "__main__":
    main()
