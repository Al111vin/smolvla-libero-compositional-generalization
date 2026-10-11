"""Immutable 120-rollout schedule for seed-2001 single/joint replication."""
from scaling_eval_schedule_v1 import schedule as four_task_schedule


def schedule():
    base = {(r["task_id"], r["kind"], r["init"], r["repeat"]): r
            for r in four_task_schedule()}
    cases = [("paired", i, 0) for i in range(20)] + [
        ("extra_init3", 3, r) for r in range(1, 5)]
    rows = []
    for kind, init, repeat in cases:
        # Pair task0 starts immediately; the remaining joint tasks follow.
        for model, task in [("single", 0), ("joint", 0),
                            ("joint", 1), ("joint", 2), ("joint", 3)]:
            source = base[(task, kind, init, repeat)]
            row = dict(source, model=model)
            row["key"] = f"{model}_{source['key']}"
            rows.append(row)
    assert len(rows) == len({r["key"] for r in rows}) == 120
    assert sum(r["model"] == "single" for r in rows) == 24
    assert sum(r["model"] == "joint" for r in rows) == 96
    return rows
