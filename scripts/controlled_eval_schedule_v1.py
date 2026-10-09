"""Immutable interleaved schedule and fail-closed per-rollout protocol checks."""
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
