"""CPU full-restore integration, retaining the v2 tensor-equality audit."""
import inspect
import json
import probe_actual_optimizer_restore_v2 as probe
from lerobot.utils.train_utils import load_training_state
from resume_lr_guard_v1 import apply_post_restore_factor


def full_restore(optimizer, state_dir):
    # The v2 caller has already built its registered scheduler.
    scheduler = inspect.currentframe().f_back.f_locals["scheduler"]
    expected = {"020000": 20000, "080000": 80000}[state_dir.parent.name]
    assert state_dir.parent.name == f"{expected:06d}"
    step, restored_optimizer, restored_scheduler = load_training_state(
        state_dir.parent, optimizer, scheduler
    )
    assert restored_optimizer is optimizer and restored_scheduler is scheduler
    lrs = scheduler.get_last_lr()
    apply_post_restore_factor(optimizer, scheduler, restored_step=step,
                             expected_step=expected, expected_lrs=lrs, factor=1.0)
    print(json.dumps({"full_training_restore_step": step,
                      "control_factor": 1.0, "rng_restore_called": True,
                      "optimizer_step_called": False}))


probe.load_optimizer_state = full_restore
probe.main()
