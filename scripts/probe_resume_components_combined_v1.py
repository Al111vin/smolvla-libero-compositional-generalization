"""Actual CPU restore plus guarded real loader, without policy updates."""
import json
from pathlib import Path
import draccus
import torch
import lerobot.scripts.lerobot_train as trainer
from accelerate import Accelerator
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from torch.utils.data import DataLoader
from resume_batch_budget_v1 import JointResumeBatchSampler
from resume_iterator_integration_v1 import guarded_resume_iterator
import probe_full_training_restore_v2 as restore_probe


original_restore = trainer.load_training_state


def restore_and_check_loader(checkpoint, optimizer, scheduler):
    result = original_restore(checkpoint, optimizer, scheduler)
    return result


def main():
    # Probe main restores real CPU policy/optimizer, then applies half-LR and
    # verifies saved moments. Intercept only its full-restore call.
    import sys
    checkpoint = Path(sys.argv[sys.argv.index("--checkpoint") + 1])
    cfg = draccus.decode(TrainPipelineConfig, json.loads(
        (checkpoint / "pretrained_model/train_config.json").read_text()))
    assert cfg.steps == 160000
    trainer.load_training_state = restore_and_check_loader
    try:
        with guarded_resume_iterator(trainer, expected_step=80000, total_steps=160000):
            restore_probe.load_training_state = trainer.load_training_state
            restore_probe.main()
            dataset = make_dataset(cfg)
            values = dataset.hf_dataset.data.column("task_index").to_pylist()
            pools = {t: [] for t in range(4)}
            for i, t in enumerate(values):
                pools[int(t)].append(i)
            sampler = JointResumeBatchSampler(pools, restored_step=80000)
            expected = iter(sampler)
            accelerator = Accelerator(cpu=True)
            assert accelerator.num_processes == 1
            state = torch.get_rng_state().clone()
            loader = accelerator.prepare(DataLoader(dataset, batch_sampler=sampler,
                num_workers=4, generator=torch.Generator().manual_seed(2000),
                prefetch_factor=2, pin_memory=False))
            batches = trainer.cycle(loader)
            for _ in range(2):
                batch = next(batches)
                assert batch["index"].tolist() == next(expected)
                assert len(set(batch["task_index"].tolist())) == 1
            assert torch.equal(state, torch.get_rng_state())
            del batches, loader
            print(json.dumps({"status": "combined_actual_restore_guarded_loader_cpu_passed",
                              "decoded_batches": 2, "loader_budget": 80000,
                              "restored_policy_rng_preserved_by_loader": True,
                              "policy_forward_called": False, "optimizer_step_called": False}))
    finally:
        trainer.load_training_state = original_restore


if __name__ == "__main__":
    main()
