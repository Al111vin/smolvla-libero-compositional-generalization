"""Hash-pinned actual one-update probe routed through registered hooks."""
from build_actual_resume_update_probe_v2 import build_source as build_v2


def build_source(source):
    source = build_v2(source)
    replacements = [
        ("    optimizer = cfg.optimizer.build(policy.get_optim_params())\n    scheduler = cfg.scheduler.build(optimizer, 40000)",
         "    optimizer, scheduler = trainer.make_optimizer_and_scheduler(cfg, policy)"),
        ("    apply_post_restore_factor(optimizer, scheduler, restored_step=step,\n        expected_step=20000, expected_lrs=[0.00005125], factor=0.5)\n", ""),
        ("    loader = torch.utils.data.DataLoader(dataset,\n        batch_sampler=SingleResumeBatchSampler(dataset, restored_step=step),\n        num_workers=4, generator=torch.Generator().manual_seed(2000),\n        prefetch_factor=2, pin_memory=False)",
         "    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True,\n        sampler=None, num_workers=4, prefetch_factor=2, pin_memory=False, drop_last=False)"),
        ("    iterator = finite_training_batches(loader, restored_step=step, total_steps=40000)",
         "    iterator = trainer.cycle(loader)"),
        ("one_actual_cpu_resume_update_v2_passed", "hooked_actual_cpu_resume_update_v4_passed"),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError("unexpected source integration site")
        source = source.replace(old, new)
    start = source.index('    accelerator = Accelerator(cpu=True, mixed_precision="no")')
    end = source.index('\n\n\nif __name__ == "__main__":')
    block = source[start:end]
    source = (source[:start] + "    from resume_training_hooks_v1 import registered_resume_hooks\n"
              + "    with registered_resume_hooks(trainer, cfg, factor=0.5):\n"
              + "\n".join("    " + line if line else line for line in block.split("\n"))
              + source[end:])
    compile(source, "hooked_resume_update_v4", "exec")
    return source
