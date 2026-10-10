"""Read-only entry contract. Not a launcher, reservation or budget grant."""
from pathlib import Path
from resume_path_guard_v1 import require_independent_absent_paths


def require_resume_entry_contract(cfg, accelerator, *, checkpoint, output, log, pid, exit_code):
    if type(accelerator.num_processes) is not int or accelerator.num_processes != 1:
        raise ValueError('registered exposure requires exactly one process')
    if not cfg.resume or cfg.steps not in (40000, 160000):
        raise ValueError('unregistered resume boundary')
    if cfg.batch_size != 2 or cfg.num_workers != 4 or cfg.env is not None:
        raise ValueError('registered batch/workers/no-eval contract required')
    if cfg.wandb.enable or cfg.policy.push_to_hub:
        raise ValueError('external training uploads forbidden')
    if cfg.dataset.streaming or cfg.dataset.image_transforms.enable:
        raise ValueError('registered offline/no-augmentation contract required')
    if Path(cfg.output_dir) != Path(output):
        raise ValueError('configured output differs from registered output')
    if Path(cfg.checkpoint_path) != Path(checkpoint):
        raise ValueError('validated checkpoint differs from registered checkpoint')
    if Path(cfg.policy.pretrained_path) != Path(checkpoint) / 'pretrained_model':
        raise ValueError('policy restore path differs from training checkpoint')
    return require_independent_absent_paths(checkpoint=checkpoint, output=output,
        log=log, pid=pid, exit_code=exit_code)
