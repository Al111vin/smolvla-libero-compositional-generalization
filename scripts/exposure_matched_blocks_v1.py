"""Finite homogeneous task batches; no trainer patch or GPU execution.

LR inputs must be captured BEFORE baseline optimizer.step, not from trainer
metrics (which are recorded AFTER scheduler.step).
"""
import random


def build_block_lambda_scheduler(optimizer, baseline_config, *, blocks, task_count):
    """Real PyTorch scheduler recognized by Accelerator.prepare.

    The facade below is an offline specification only. Use this factory for
    training so AMP-skipped optimizer updates also skip scheduler advancement.
    """
    from torch.optim.lr_scheduler import LambdaLR
    if blocks <= 0 or task_count <= 0:
        raise ValueError("positive finite budget required")
    baseline = baseline_config.build(optimizer, blocks)
    if not isinstance(baseline, LambdaLR) or baseline.last_epoch != 0:
        raise ValueError("expected freshly initialized LambdaLR baseline")

    def expand(fn):
        def lr_factor(completed_updates):
            if not 0 <= completed_updates <= blocks * task_count:
                raise ValueError("scheduler budget exhausted")
            return fn(completed_updates // task_count)
        return lr_factor

    return LambdaLR(optimizer, [expand(fn) for fn in baseline.lr_lambdas], last_epoch=-1)


def balanced_batches(indices_by_task, *, blocks, batch_size, seed):
    if blocks <= 0 or batch_size <= 0:
        raise ValueError("positive finite blocks and batch size required")
    tasks = sorted(indices_by_task)
    if not tasks or any(not indices_by_task[t] for t in tasks):
        raise ValueError("every task needs sample indices")
    flattened = [i for t in tasks for i in indices_by_task[t]]
    if len(set(flattened)) != len(flattened):
        raise ValueError("sample indices must be unique across task pools")
    rng = random.Random(seed)
    pools = {t: list(indices_by_task[t]) for t in tasks}
    cursors = {t: len(pools[t]) for t in tasks}
    for block in range(blocks):
        order = tasks.copy()
        rng.shuffle(order)
        for task in order:
            batch = []
            for _ in range(batch_size):
                if cursors[task] == len(pools[task]):
                    rng.shuffle(pools[task])
                    cursors[task] = 0
                batch.append(pools[task][cursors[task]])
                cursors[task] += 1
            yield block, task, batch


def baseline_lr_for_update(baseline_pre_step_lrs, update_index, task_count):
    if task_count <= 0 or update_index < 0:
        raise ValueError("invalid update index or task count")
    block = update_index // task_count
    if block >= len(baseline_pre_step_lrs):
        raise ValueError("update exceeds registered baseline LR budget")
    return baseline_pre_step_lrs[block]


class BlockSchedulerAdapter:
    """Single-process facade; call once AFTER each optimizer update.

    Construct the wrapped scheduler with the baseline block budget, not the
    expanded update budget. Accelerate integration remains separately tested.
    """
    def __init__(self, scheduler, *, task_count, blocks):
        if task_count <= 0 or blocks <= 0:
            raise ValueError("positive task count and blocks required")
        self.scheduler = scheduler
        self.optimizer = scheduler.optimizer
        self.task_count = task_count
        self.blocks = blocks
        self.completed_updates = 0

    def step(self):
        if self.completed_updates >= self.task_count * self.blocks:
            raise ValueError("scheduler budget exhausted")
        self.completed_updates += 1
        if self.completed_updates % self.task_count == 0:
            self.scheduler.step()

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def state_dict(self):
        return dict(task_count=self.task_count, blocks=self.blocks,
                    completed_updates=self.completed_updates,
                    baseline_scheduler=self.scheduler.state_dict())

    def load_state_dict(self, state):
        if state["task_count"] != self.task_count or state["blocks"] != self.blocks:
            raise ValueError("registered scheduler budget mismatch")
        updates = state["completed_updates"]
        if not isinstance(updates, int) or not 0 <= updates <= self.task_count * self.blocks:
            raise ValueError("invalid completed update count")
        if state["baseline_scheduler"]["last_epoch"] != updates // self.task_count:
            raise ValueError("baseline scheduler phase mismatch")
        self.scheduler.load_state_dict(state["baseline_scheduler"])
        self.completed_updates = updates
        # PyTorch scheduler.load_state_dict does not restore optimizer LR.
        lrs = self.get_last_lr()
        if len(self.optimizer.param_groups) != len(lrs):
            raise ValueError("optimizer LR group count mismatch")
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr
