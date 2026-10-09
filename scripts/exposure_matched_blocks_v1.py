"""Finite homogeneous task batches; no trainer patch or GPU execution.

LR inputs must be captured BEFORE baseline optimizer.step, not from trainer
metrics (which are recorded AFTER scheduler.step).
"""
import random


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
