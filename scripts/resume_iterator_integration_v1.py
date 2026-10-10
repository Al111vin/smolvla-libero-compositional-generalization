"""Scoped iterator integration only; no training launch or GPU budget."""
from contextlib import contextmanager
from resume_batch_budget_v1 import finite_training_batches


@contextmanager
def guarded_resume_iterator(trainer, *, expected_step, total_steps):
    if (expected_step, total_steps) not in ((20000, 40000), (80000, 160000)):
        raise ValueError("unregistered resume budget")
    original_restore = trainer.load_training_state
    original_cycle = trainer.cycle
    restored = []
    iterator_created = []

    def checked_restore(*args, **kwargs):
        if restored:
            raise RuntimeError("training state restored more than once")
        result = original_restore(*args, **kwargs)
        step, optimizer, scheduler = result
        if step != expected_step or scheduler is None or scheduler.last_epoch != step:
            raise RuntimeError("restored training boundary mismatch")
        restored.append(step)
        return result

    def checked_iterator(loader):
        if restored != [expected_step]:
            raise RuntimeError("iterator requested before verified restore")
        if iterator_created:
            raise RuntimeError("resume iterator cannot be recreated")
        iterator_created.append(True)
        return finite_training_batches(loader, restored_step=expected_step,
                                       total_steps=total_steps)

    trainer.load_training_state = checked_restore
    trainer.cycle = checked_iterator
    try:
        yield
    finally:
        trainer.load_training_state = original_restore
        trainer.cycle = original_cycle
