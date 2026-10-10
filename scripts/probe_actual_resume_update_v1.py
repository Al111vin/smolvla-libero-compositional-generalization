"""One transient CPU update through installed trainer; never saves weights."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import draccus
import torch
from accelerate import Accelerator
import lerobot.scripts.lerobot_train as trainer
from lerobot.configs.train import TrainPipelineConfig
from resume_batch_budget_v1 import SingleResumeBatchSampler, finite_training_batches
from resume_lr_guard_v1 import apply_post_restore_factor


def main():
    checkpoint = Path("/root/smolvla-training-prep/results/training/teacher_native_spatial_task0_single_current_recipe_40k_batch2_v1/checkpoints/020000")
    cfg = draccus.decode(TrainPipelineConfig, json.loads(
        (checkpoint / "pretrained_model/train_config.json").read_text()))
    cfg.resume = True
    cfg.policy.device = "cpu"
    cfg.policy.pretrained_path = checkpoint / "pretrained_model"
    assert cfg.steps == 40000 and cfg.batch_size == 2 and cfg.env is None
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    assert accelerator.num_processes == 1
    dataset = trainer.make_dataset(cfg)
    policy = trainer.make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    preprocessor, _ = trainer.make_pre_post_processors(
        policy_cfg=cfg.policy, pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "normalizer_processor": {"stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping},
            "rename_observations_processor": {"rename_map": cfg.rename_map}},
        postprocessor_overrides={"unnormalizer_processor": {
            "stats": dataset.meta.stats, "features": policy.config.output_features,
            "norm_map": policy.config.normalization_mapping}})
    optimizer = cfg.optimizer.build(policy.get_optim_params())
    scheduler = cfg.scheduler.build(optimizer, 40000)
    step, optimizer, scheduler = trainer.load_training_state(checkpoint, optimizer, scheduler)
    apply_post_restore_factor(optimizer, scheduler, restored_step=step,
        expected_step=20000, expected_lrs=[0.00005125], factor=0.5)
    loader = torch.utils.data.DataLoader(dataset,
        batch_sampler=SingleResumeBatchSampler(dataset, restored_step=step),
        num_workers=4, generator=torch.Generator().manual_seed(2000),
        prefetch_factor=2, pin_memory=False)
    policy, optimizer, loader, scheduler = accelerator.prepare(policy, optimizer, loader, scheduler)
    state = torch.get_rng_state().clone()
    iterator = finite_training_batches(loader, restored_step=step, total_steps=40000)
    batch = next(iterator)
    assert torch.equal(state, torch.get_rng_state())
    batch = preprocessor(batch)
    watched = next(p for p in policy.parameters() if p.requires_grad)
    before = watched.detach().clone()
    class CheckedScheduler:
        def step(self):
            if optimizer.step_was_skipped:
                raise RuntimeError("optimizer update skipped")
            scheduler.step()
    pre_lr = optimizer.param_groups[0]["lr"]
    assert pre_lr == 0.000025625
    metrics, _ = trainer.update_policy(SimpleNamespace(), policy, batch, optimizer,
        cfg.optimizer.grad_clip_norm, accelerator=accelerator, lr_scheduler=CheckedScheduler())
    assert math.isfinite(metrics.loss) and math.isfinite(metrics.grad_norm)
    assert not torch.equal(before, watched.detach())
    assert scheduler.last_epoch == 20001
    print(json.dumps({"status": "one_actual_cpu_resume_update_passed",
        "updates": 1, "pre_update_lr": pre_lr, "loss": metrics.loss,
        "grad_norm": metrics.grad_norm, "scheduler_epoch": scheduler.last_epoch,
        "watched_trainable_parameter_changed": True,
        "checkpoint_saved": False, "gpu_used": False}))


if __name__ == "__main__":
    main()
