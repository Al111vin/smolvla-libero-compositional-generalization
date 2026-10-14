"""Frozen 24-rollout task1 control schedule derived from the validated V3 schedule."""
from scaling_eval_schedule_v1 import schedule as full_schedule


def schedule():
    rows = [row for row in full_schedule() if row["task_id"] == 1]
    if len(rows) != 24:
        raise ValueError(f"Expected exactly 24 task1 rows, got {len(rows)}")
    paired = [row for row in rows if row["kind"] == "paired"]
    init3 = [row for row in rows if row["init"] == 3]
    extras = [row for row in rows if row["kind"] == "extra_init3"]
    if len(paired) != 20 or len(init3) != 5 or len(extras) != 4:
        raise ValueError("Task1 schedule does not contain 20 paired plus four fixed-init3 repeats")
    if any(row["task_id"] != 1 or row["wait"] != 10 or row["max_steps"] != 300
           or row["action_steps"] != 25 for row in rows):
        raise ValueError("Task1 schedule protocol differs from the registered evaluator")
    if any(row["cli_seed"] != 12345 + row["init"]
           or row["effective_seed"] != 12345 + 2 * row["init"] for row in paired):
        raise ValueError("Paired task1 seeds differ from the registered seed mapping")
    if any(row["init"] != 3 or row["cli_seed"] != 12348
           or row["effective_seed"] != 12351 for row in extras):
        raise ValueError("Fixed-init3 repeat seeds differ from registration")
    return rows


def gate_task1(outcomes):
    rows = schedule()
    if set(outcomes) != {row["key"] for row in rows}:
        raise ValueError("Missing or unexpected task1 rollout outcomes")
    if any(type(value) is not bool for value in outcomes.values()):
        raise ValueError("Rollout outcomes must be explicit booleans")
    paired = sum(outcomes[row["key"]] for row in rows if row["kind"] == "paired")
    init3 = sum(outcomes[row["key"]] for row in rows if row["init"] == 3)
    return {"paired_successes": paired, "paired_total": 20,
            "init3_successes": init3, "init3_total": 5,
            "passed": paired >= 10 and init3 == 5,
            "automatic_next_stage": False, "fold02": "LOCKED"}


if __name__ == "__main__":
    import json
    print(json.dumps(schedule(), indent=2))
