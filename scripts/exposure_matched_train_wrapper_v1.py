"""Independent bounded four-task launcher. Importing does not start training."""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exposure_matched_blocks_v1 import (
    build_block_lambda_scheduler, install_homogeneous_loader_patch,
    require_executed_optimizer_update,
)

REPO = "local/libero_spatial_tasks0_3_native_v1"
BLOCKS = 40000
TASKS = 4


def install(trainer):
    original_factory = trainer.make_optimizer_and_scheduler
    original_update = trainer.update_policy
    signature = inspect.signature(original_update)
    completed = 0
    block_tasks = set()

    def factory(cfg, policy):
        if (cfg.dataset.repo_id != REPO or cfg.steps != BLOCKS * TASKS
                or cfg.batch_size != 2 or cfg.seed != 1000 or cfg.resume
                or cfg.env is not None or cfg.use_rabc):
            raise ValueError("training differs from registered finite experiment")
        if list(cfg.dataset.episodes) != list(range(200)):
            raise ValueError("expected all200 registered episodes")
        # Build directly: native160k scheduler must never be used.
        params = policy.get_optim_params() if cfg.use_policy_training_preset else policy.parameters()
        optimizer = cfg.optimizer.build(params)
        scheduler = build_block_lambda_scheduler(optimizer, cfg.scheduler,
                                                  blocks=BLOCKS, task_count=TASKS)
        return optimizer, scheduler

    def update(*args, **kwargs):
        nonlocal completed, block_tasks
        bound = signature.bind(*args, **kwargs)
        accelerator = bound.arguments["accelerator"]
        if accelerator.num_processes != 1 or accelerator.gradient_accumulation_steps != 1:
            raise ValueError("only single-process non-accumulated updates registered")
        if completed >= BLOCKS * TASKS:
            raise ValueError("finite update budget exhausted")
        values = bound.arguments["batch"]["task_index"].reshape(-1).tolist()
        if len(values) != 2 or values[0] != values[1] or values[0] not in range(TASKS):
            raise ValueError("non-homogeneous or invalid task batch")
        task = int(values[0])
        if task in block_tasks:
            raise ValueError("task repeated within balanced block")
        result = original_update(*args, **kwargs)
        require_executed_optimizer_update(bound.arguments["optimizer"])
        block_tasks.add(task)
        completed += 1
        if completed % TASKS == 0:
            if block_tasks != set(range(TASKS)):
                raise ValueError("incomplete task block")
            block_tasks = set()
        return result

    loader_restore = install_homogeneous_loader_patch(repo_id=REPO, blocks=BLOCKS, seed=1000)
    trainer.make_optimizer_and_scheduler = factory
    trainer.update_policy = update
    return original_factory, original_update, loader_restore


def main():
    import lerobot.scripts.lerobot_train as trainer
    install(trainer)
    trainer.train()


if __name__ == "__main__":
    main()
