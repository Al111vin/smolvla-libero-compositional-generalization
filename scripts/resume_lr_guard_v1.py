"""Post-restore LR intervention guard; no training entry point."""
import math


def apply_post_restore_factor(optimizer, scheduler, *, restored_step, expected_step,
                              expected_lrs, factor):
    if factor not in (1.0, 0.5):
        raise ValueError("unregistered LR factor")
    if restored_step != expected_step or scheduler.last_epoch != expected_step:
        raise ValueError("restored step/scheduler boundary mismatch")
    if getattr(scheduler, "_resume_lr_guard_applied", False):
        raise ValueError("intervention already applied")
    if scheduler.optimizer is not optimizer:
        raise ValueError("scheduler bound to a different optimizer")
    groups = optimizer.param_groups
    lrs = scheduler.get_last_lr()
    lambdas = scheduler.lr_lambdas
    if not len(groups) == len(lrs) == len(expected_lrs) == len(lambdas):
        raise ValueError("parameter-group/schedule count mismatch")
    if not groups or not all(callable(fn) for fn in lambdas):
        raise ValueError("empty groups or invalid scheduler functions")
    for group, actual, expected in zip(groups, lrs, expected_lrs):
        if not math.isfinite(expected) or expected <= 0:
            raise ValueError("invalid registered boundary LR")
        if actual != expected or group["lr"] != expected:
            raise ValueError("saved optimizer/scheduler LR mismatch")
    # Validate everything before mutation. Preserve base_lrs and optimizer moments.
    scheduler.lr_lambdas = [lambda step, fn=fn: factor * fn(step) for fn in lambdas]
    scaled = [lr * factor for lr in expected_lrs]
    for group, lr in zip(groups, scaled):
        group["lr"] = lr
    scheduler._last_lr = scaled
    scheduler._resume_lr_guard_applied = True
    return scaled
