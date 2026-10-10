"""Scoped registered resume hooks; not a launcher or budget authorization."""
from contextlib import contextmanager
from exposure_matched_blocks_v1 import build_block_lambda_scheduler, require_executed_optimizer_update
from resume_batch_budget_v1 import JointResumeBatchSampler, SingleResumeBatchSampler
from resume_iterator_integration_v1 import guarded_resume_iterator
from resume_lr_guard_v1 import apply_post_restore_factor


@contextmanager
def registered_resume_hooks(trainer, cfg, *, factor):
    import torch
    if factor not in (1.0, 0.5) or not cfg.resume or cfg.steps not in (40000, 160000):
        raise ValueError("unregistered resume condition")
    if cfg.batch_size != 2 or cfg.num_workers != 4 or cfg.env is not None:
        raise ValueError("registered batch/workers/no-eval condition required")
    if cfg.dataset.streaming or cfg.dataset.image_transforms.enable or cfg.policy.push_to_hub:
        raise ValueError("registered offline/no-augmentation/no-upload condition required")
    expected = 20000 if cfg.steps == 40000 else 80000
    original_factory = trainer.make_optimizer_and_scheduler
    original_restore = trainer.load_training_state
    original_update = trainer.update_policy
    original_loader_init = torch.utils.data.DataLoader.__init__
    loader_calls = []

    def factory(config, policy):
        if config is not cfg:
            raise ValueError("unexpected training configuration")
        if cfg.steps == 40000:
            return original_factory(config, policy)
        optimizer = cfg.optimizer.build(policy.get_optim_params())
        scheduler = build_block_lambda_scheduler(optimizer, cfg.scheduler, blocks=40000, task_count=4)
        return optimizer, scheduler

    def restore(*args, **kwargs):
        step, optimizer, scheduler = original_restore(*args, **kwargs)
        apply_post_restore_factor(optimizer, scheduler, restored_step=step,
            expected_step=expected, expected_lrs=[0.00005125], factor=factor)
        return step, optimizer, scheduler

    def loader_init(self, dataset, *args, **kwargs):
        if getattr(dataset, "repo_id", None) != cfg.dataset.repo_id:
            return original_loader_init(self, dataset, *args, **kwargs)
        if loader_calls or args or kwargs.get("batch_size") != 2 or kwargs.get("sampler") is not None:
            raise ValueError("unexpected or repeated target loader")
        if (kwargs.get("num_workers") != 4 or kwargs.get("prefetch_factor") != 2
                or kwargs.get("batch_sampler") is not None or kwargs.get("generator") is not None):
            raise ValueError("unexpected target worker or sampler configuration")
        values = dataset.hf_dataset.data.column("task_index").to_pylist()
        if cfg.steps == 40000:
            if set(values) != {0}:
                raise ValueError("single task mapping mismatch")
            sampler = SingleResumeBatchSampler(dataset, restored_step=expected)
        else:
            pools = {t: [] for t in range(4)}
            for index, task in enumerate(values):
                if int(task) not in pools:
                    raise ValueError("joint task mapping mismatch")
                pools[int(task)].append(index)
            if [len(pools[t]) for t in range(4)] != [5068, 6707, 5882, 5052]:
                raise ValueError("joint frame count mismatch")
            sampler = JointResumeBatchSampler(pools, restored_step=expected)
        for key in ("batch_size", "shuffle", "sampler", "drop_last"):
            kwargs.pop(key, None)
        kwargs["batch_sampler"] = sampler
        kwargs["generator"] = torch.Generator().manual_seed(2000)
        loader_calls.append(True)
        return original_loader_init(self, dataset, *args, **kwargs)

    def update(*args, **kwargs):
        optimizer = args[3]
        scheduler = kwargs.get("lr_scheduler")
        if scheduler is None:
            raise ValueError("scheduler required")
        class CheckedScheduler:
            def step(self):
                require_executed_optimizer_update(optimizer)
                scheduler.step()
        kwargs["lr_scheduler"] = CheckedScheduler()
        result = original_update(*args, **kwargs)
        require_executed_optimizer_update(optimizer)
        return result

    trainer.make_optimizer_and_scheduler = factory
    trainer.load_training_state = restore
    trainer.update_policy = update
    torch.utils.data.DataLoader.__init__ = loader_init
    try:
        with guarded_resume_iterator(trainer, expected_step=expected, total_steps=cfg.steps):
            yield
    finally:
        trainer.make_optimizer_and_scheduler = original_factory
        trainer.load_training_state = original_restore
        trainer.update_policy = original_update
        torch.utils.data.DataLoader.__init__ = original_loader_init
