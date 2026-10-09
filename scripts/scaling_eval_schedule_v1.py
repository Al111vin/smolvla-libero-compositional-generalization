"""Preregistered final-checkpoint schedule for four native Spatial tasks.

Independent of historical task0 schedules. No execution or adaptive sampling.
"""
import math


def schedule():
    rows = []
    cases = [("paired", i, 0) for i in range(20)] + [
        ("extra_init3", 3, r) for r in range(1, 5)
    ]
    for kind, index, repeat in cases:
        for task in range(4):
            rows.append({
                "task_id": task, "kind": kind, "init": index,
                "repeat": repeat, "cli_seed": 12345 + index,
                "effective_seed": 12345 + 2 * index,
                "environment_seed": 12351, "wait": 10,
                "max_steps": 300, "action_steps": 25,
                "key": f"task{task}_{kind}_{index}_r{repeat}_160k",
            })
    return rows


def validate_protocol(row, protocol):
    for key in ("task_id", "init", "cli_seed", "effective_seed",
                "environment_seed", "wait", "max_steps", "action_steps"):
        if protocol.get(key) != row[key]:
            raise ValueError(f"Protocol mismatch: {key}")


def validate_rollout(row, summary, actions):
    """Validate full action evidence, not only the evaluator success flag."""
    expected = {'suite': 'libero_spatial', 'init_source': 'benchmark'}
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f'Summary mismatch: {key}')
    for key, value in {'task_id': row['task_id'], 'init_index': row['init'],
                       'seed': row['effective_seed'], 'wait_steps': row['wait'],
                       'n_action_steps': row['action_steps']}.items():
        if int(summary[key]) != value:
            raise ValueError(f'Summary mismatch: {key}')
    if summary['success'] not in ('True', 'False'):
        raise ValueError('Invalid success flag')
    steps = int(summary['steps'])
    if not 1 <= steps <= row['max_steps'] or len(actions) != steps:
        raise ValueError('Action budget or row count mismatch')
    fields = ['reward'] + [f'{prefix}_{j}' for prefix in
        ('raw_action', 'processed_action', 'applied_action') for j in range(7)] + [
        f'state_{j}' for j in range(15)]
    rewards = []
    for index, action in enumerate(actions):
        if int(action['step']) != index:
            raise ValueError('Noncontiguous action steps')
        if any(not math.isfinite(float(action[key])) for key in fields):
            raise ValueError('Nonfinite action evidence')
        for j in range(7):
            if float(action[f'applied_action_{j}']) != max(-1.0, min(
                    1.0, float(action[f'processed_action_{j}']))):
                raise ValueError('Action clipping mismatch')
        rewards.append(float(action['reward']))
    total = float(summary['total_reward'])
    if not math.isfinite(total) or abs(total - sum(rewards)) >= 1e-6:
        raise ValueError('Reward total mismatch')
    success = summary['success'] == 'True'
    if success != any(reward > 0 for reward in rewards):
        raise ValueError('Success flag disagrees with sparse task reward')
    return success


def gate(outcomes):
    """Require the complete immutable schedule before calculating gates."""
    rows = schedule()
    if set(outcomes) != {r["key"] for r in rows}:
        raise ValueError("Missing or unexpected rollout outcomes")
    if any(type(value) is not bool for value in outcomes.values()):
        raise ValueError("Outcomes must be explicit booleans")
    result = {}
    for task in range(4):
        selected = [r for r in rows if r["task_id"] == task]
        paired = sum(outcomes[r["key"]] for r in selected if r["kind"] == "paired")
        init3 = sum(outcomes[r["key"]] for r in selected if r["init"] == 3)
        passed = paired >= (11 if task == 0 else 10)
        if task == 0:
            passed = passed and init3 == 5
        result[str(task)] = {"paired_successes": paired, "paired_total": 20,
                             "init3_successes": init3, "init3_total": 5,
                             "passed": passed}
    return {"tasks": result, "passed": all(r["passed"] for r in result.values()),
            "automatic_next_stage": False, "fold02": "locked"}
