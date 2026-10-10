"""CPU toy-optimizer integration probe; no actual policy/checkpoint restoration."""
import copy
import json
import torch

if "apply_post_restore_factor" not in globals():
    from resume_lr_guard_v1 import apply_post_restore_factor


def main():
    param = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([param], lr=0.0001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda k: 1 - k / 40000)
    for _ in range(10):
        param.grad = torch.ones_like(param)
        optimizer.step()
        scheduler.step()
    moments = copy.deepcopy(optimizer.state_dict()["state"])
    boundary = scheduler.get_last_lr()
    apply_post_restore_factor(optimizer, scheduler, restored_step=10,
                             expected_step=10, expected_lrs=boundary, factor=0.5)
    for key, state in moments.items():
        for name, value in state.items():
            actual = optimizer.state_dict()["state"][key][name]
            assert torch.equal(value, actual) if torch.is_tensor(value) else value == actual
    assert optimizer.param_groups[0]["lr"] == boundary[0] * 0.5
    for step in range(10, 40000):
        expected = 0.0001 * (1 - step / 40000) * 0.5
        assert optimizer.param_groups[0]["lr"] == expected
        optimizer.step()
        scheduler.step()
    assert scheduler.last_epoch == 40000 and optimizer.param_groups[0]["lr"] == 0
    print(json.dumps({"status": "torch_cpu_guard_probe_passed", "remaining_steps": 39990,
                      "moments_unchanged_by_intervention": True,
                      "actual_checkpoint_restored": False}))


if __name__ == "__main__":
    main()
