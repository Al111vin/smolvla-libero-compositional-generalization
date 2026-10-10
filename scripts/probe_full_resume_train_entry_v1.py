"""Bounded installed train-function CPU probe; no checkpoint or GPU work."""
import json
import math
import sys
from pathlib import Path
import draccus
from accelerate import Accelerator
import lerobot.scripts.lerobot_train as trainer
from lerobot.configs.train import TrainPipelineConfig
from resume_training_hooks_v2 import registered_resume_hooks


class ProbeFinished(Exception):
    pass


def main():
    checkpoint = Path('/root/smolvla-training-prep/results/training/teacher_native_spatial_task0_single_current_recipe_40k_batch2_v1/checkpoints/020000')
    config_path = checkpoint / 'pretrained_model/train_config.json'
    output = Path('/tmp/smolvla_full_entry_cpu_probe_v1_20261010')
    if output.exists():
        raise FileExistsError(output)
    cfg = draccus.decode(TrainPipelineConfig, json.loads(config_path.read_text()))
    cfg.resume = True
    cfg.policy.device = 'cpu'
    cfg.policy.push_to_hub = False
    cfg.output_dir = output
    cfg.job_name = 'full_entry_cpu_probe_v1'
    cfg.wandb.enable = False
    cfg.save_checkpoint = False
    cfg.eval_freq = 0
    sys.argv = [sys.argv[0], f'--config_path={config_path}']
    accelerator = Accelerator(cpu=True, mixed_precision='no', step_scheduler_with_optimizer=False)
    if accelerator.num_processes != 1 or accelerator.device.type != 'cpu':
        raise RuntimeError('single CPU process required')
    updates = []
    original_update = trainer.update_policy
    original_save = trainer.save_checkpoint

    def forbidden_save(*args, **kwargs):
        raise RuntimeError('checkpoint save forbidden in probe')

    def one_update(*args, **kwargs):
        if updates:
            raise RuntimeError('second update forbidden')
        if cfg.checkpoint_path != checkpoint or cfg.policy.pretrained_path != checkpoint / 'pretrained_model':
            raise RuntimeError('validated checkpoint path mismatch')
        optimizer = args[3]
        pre_lr = optimizer.param_groups[0]['lr']
        if pre_lr != 0.000025625:
            raise RuntimeError('boundary LR mismatch')
        metrics, _ = original_update(*args, **kwargs)
        if not math.isfinite(metrics.loss) or not math.isfinite(metrics.grad_norm):
            raise RuntimeError('nonfinite update')
        updates.append(dict(loss=metrics.loss, grad_norm=metrics.grad_norm, pre_update_lr=pre_lr))
        raise ProbeFinished()

    trainer.update_policy = one_update
    trainer.save_checkpoint = forbidden_save
    try:
        with registered_resume_hooks(trainer, cfg, factor=0.5):
            try:
                trainer.train(cfg, accelerator=accelerator)
            except ProbeFinished:
                pass
            else:
                raise RuntimeError('train returned without controlled one-update stop')
        if len(updates) != 1:
            raise RuntimeError('missing update')
        if output.exists():
            raise RuntimeError('unexpected output directory creation')
        print(json.dumps(dict(status='full_installed_train_entry_cpu_v1_passed', updates=updates,
            gpu_used=False, checkpoint_saved=False, output_created=False,
            stop='controlled exception after first real update before save/eval branches')))
    finally:
        trainer.update_policy = original_update
        trainer.save_checkpoint = original_save


if __name__ == '__main__':
    main()
