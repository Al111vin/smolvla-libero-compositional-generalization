# V4 LIBERO Spatial LOCO Evaluation Protocol

This protocol was fixed before running either held-out target task.

## Frozen models

| Model label | Training tasks | Held-out task | Primary checkpoint |
|---|---|---:|---|
| `task6_holdout` | 0, 2, 3, 4, 7, 8, 9 | 6 | 90,000 |
| `task3_holdout` | 0, 2, 4, 6, 7, 8, 9 | 3 | 90,000 |

Both models use training seed 1000, batch size 8, 90,000 optimizer
steps, a 3,000-step warmup, and cosine decay through step 90,000.
The 30k and 60k checkpoints are recovery points only. The primary
result uses the preselected 90k checkpoint and is not selected using
held-out performance.

## Rollout matrix

Each cell uses all 50 official LIBERO benchmark initial states.

| Model | Task 3 | Task 6 |
|---|---|---|
| `task3_holdout` | held-out primary | seen control |
| `task6_holdout` | seen control | held-out primary |

The complete evaluation therefore contains 200 rollouts. Results from
the two held-out cells are the primary LOCO measurements. Seen-control
cells distinguish failure to generalize from general policy failure.

## Fixed rollout settings

- Suite: `libero_spatial`
- Initialization source: official benchmark states 0 through 49
- Policy actions per episode: 280
- Stabilization: 10 no-op steps before the policy, not counted in 280
- Stabilization action: `[0, 0, 0, 0, 0, 0, -1]`
- Action chunk execution: `n_action_steps=25`
- Success criterion: `env.check_success()` only
- Stop immediately on official success; timeout is failure
- Observation adapter: the project's validated two-camera, unflipped
  RGB input and 15-dimensional state adapter

The 25-action setting was selected earlier on task 0, which is a
training task in both folds, before either task 3 or task 6 was used as
a held-out target. No action horizon or checkpoint is tuned on either
held-out task.

The earlier task-0 horizon ablation used the legacy evaluator's
all-zero stabilization action. Its gripper component therefore differs
from the official action fixed above. The pre-target choice of 25 is
retained without re-tuning; it is treated as a training-task prior, not
as a same-protocol validation result.

## Determinism and resumability

For benchmark initial-state index `i`:

```text
environment seed = 1000 + i
policy seed      = 2000 + i
```

Python, NumPy, Torch, and CUDA policy RNG state is reset for every
episode. `policy.reset()` is also called at the beginning of every
episode. The same task/state/seed combinations are reused across both
models.

The evaluator appends one durable summary row after every rollout and
can resume without repeating completed task/state pairs. Compressed
per-step action traces are retained outside Git for diagnostics.

## Reporting

For every model/task cell, report successes out of 50, success rate,
and a 95% Wilson confidence interval. Report the equal-weight mean of
the two held-out success rates as the LOCO macro score. Seen-versus-
held-out comparisons use paired state-level outcomes. Because the
current experiment uses one training seed per fold, rollout intervals
do not measure training-seed uncertainty.
