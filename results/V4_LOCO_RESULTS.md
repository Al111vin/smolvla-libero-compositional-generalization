# V4 LIBERO Spatial LOCO Results

## Primary result

The frozen V4 SmolVLA policies achieved **0.0%** mean success on the two held-out compositional folds, versus **86.0%** on matched seen-task controls. The macro generalization gap was **86.0%**.

## Per-cell results

| Model | Task | Role | Success | 95% Wilson CI | Mean successful steps |
|---|---:|---|---:|---:|---:|
| `task3_holdout` | 3 | heldout | 0/50 (0.0%) | [0.0%, 7.1%] | — |
| `task3_holdout` | 6 | seen_control | 44/50 (88.0%) | [76.2%, 94.4%] | 107.86 |
| `task6_holdout` | 3 | seen_control | 42/50 (84.0%) | [71.5%, 91.7%] | 91.98 |
| `task6_holdout` | 6 | heldout | 0/50 (0.0%) | [0.0%, 7.1%] | — |

## Paired seen-versus-held-out comparison

For each target task, outcomes are paired by the same official benchmark initial-state index.

| Task | Seen-only successes | Held-out-only successes | Both fail | Gap | Exact McNemar p | Holm-adjusted p |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 42 | 0 | 8 | 84.0% | 4.547e-13 | 4.547e-13 |
| 6 | 44 | 0 | 6 | 88.0% | 1.137e-13 | 2.274e-13 |

## Interpretation

- Both held-out cells were 0/50, while their matched seen controls were 42/50 and 44/50.
- The strong seen controls show that the result is not explained by a generally broken policy or evaluation pipeline.
- Under these two audited target-role LOCO folds, the policy learned the training combinations but did not transfer that behavior to the held-out composition.
- Aggregating the two equally sized held-out cells gives 0/100; the descriptive pooled Wilson 95% upper bound is 3.7%.

## Frozen protocol and provenance

- Protocol: `v4_loco_official_v1`
- Evaluation commit: `a352a4b90aba1a3cb716f9d365bec1cfd139cbcd`
- Evaluator fingerprint: `f0b332bc668b3336fe552595540504c55a2954dfca35b0dd82aa1008e4e6542c`
- Checkpoint: preselected step 90,000 for both folds
- Official benchmark states: 0 through 49 per cell
- Policy horizon: 280 actions
- Stabilization: 10 steps with `[0,0,0,0,0,0,-1]`
- Action execution horizon: `n_action_steps=25`
- Success: `env.check_success()` only

## Limitations

- Each fold has one training seed. Confidence intervals describe benchmark-state outcomes, not variation across training runs.
- The 50 states per task are a fixed benchmark set; Wilson intervals and McNemar p-values are descriptive of state-level uncertainty under this protocol rather than a guarantee for an unrestricted task population.
- The conclusion applies to the two audited LIBERO-Spatial target-role folds (tasks 3 and 6), not to every possible form of compositional generalization.
- The action horizon was selected on task 0 before target evaluation, but that earlier ablation used the legacy all-zero stabilization action.
- Post-hoc diagnostics may explain the failures, but must not be used to replace or retune these frozen primary results.
