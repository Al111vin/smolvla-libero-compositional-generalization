"""Immutable interleaved schedule and fail-closed per-rollout protocol checks."""
import math
def schedule():
    rows = []
    for kind, index, repeat in [("paired", i, 0) for i in range(20)] + [
            ("extra_init3", 3, r) for r in range(1, 5)]:
        for model in ("40k", "80k"):
            rows.append({"model": model, "kind": kind, "init": index,
                         "repeat": repeat, "cli_seed": 12345 + index,
                         "key": f"{kind}_{index}_r{repeat}_{model}"})
    assert len(rows) == len({r["key"] for r in rows}) == 48
    return rows


def validate_protocol(row, protocol):
    assert protocol["init"] == row["init"]
    assert protocol["cli_seed"] == row["cli_seed"]
    assert protocol["effective_seed"] == 12345 + 2 * row["init"]
    assert protocol["environment_seed"] == 12351
    assert protocol["wait"] == 10 and protocol["max_steps"] == 300
    assert protocol["action_steps"] == 25


def environment_signature(capture):
    assert capture["images"] and capture["model_arrays"]
    return {k: capture[k] for k in ("model_arrays", "simulator_state", "images", "frame")}


def validate_rollout(row, summary, actions):
    assert summary["suite"] == "libero_spatial" and int(summary["task_id"]) == 0
    assert summary["init_source"] == "benchmark" and int(summary["init_index"]) == row["init"]
    assert int(summary["seed"]) == 12345 + 2 * row["init"]
    assert int(summary["wait_steps"]) == 10 and int(summary["n_action_steps"]) == 25
    assert summary["success"] in ("True", "False")
    steps = int(summary["steps"])
    assert 1 <= steps <= 300 and len(actions) == steps
    for i, action in enumerate(actions):
        assert int(action["step"]) == i
        for key in ["reward"] + [f"{prefix}_{j}" for prefix in
                ("raw_action", "processed_action", "applied_action") for j in range(7)] + [
                f"state_{j}" for j in range(15)]:
            assert math.isfinite(float(action[key]))
        for j in range(7):
            processed = float(action[f"processed_action_{j}"])
            applied = float(action[f"applied_action_{j}"])
            assert applied == max(-1.0, min(1.0, processed))
    total = float(summary["total_reward"])
    assert math.isfinite(total)
    assert abs(total - sum(float(a["reward"]) for a in actions)) < 1e-6
