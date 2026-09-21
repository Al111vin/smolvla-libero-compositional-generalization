# LIBERO36 Joint32 Strict Benchmark — Final Conclusion

Date: 2026-09-21

## Evidence

The frozen feasible benchmark contains 32 tasks (36 logical task definitions minus physically infeasible Tasks 15, 17, 33, and 35). Each task was evaluated at four checkpoints (`030000`, `060000`, `090000`, and `last`) on 50 deterministic initial states: 6,400 closed-loop rollouts in total. All 32 task summaries and all 128 checkpoint groups are complete. No deterministic-algorithm errors were recorded.

Aggregate strict-protocol result: **0/6,400 successes; mean reward 0.0 at every checkpoint**.

## Interpretation

Under the registered strict protocol, this trained joint32 policy provides no positive evidence of compositional generalization on the frozen feasible benchmark. This is a negative result for this model/training/evaluation configuration; it does not establish that no training method can generalize on LIBERO. The evaluator completed all planned rollouts without deterministic-algorithm failures.

Task1 edge-pinch v2 remains a separate diagnostic track and did not pass its strict closed-loop gate. Fold 02 remains **LOCKED** under the pre-defined project rule. The joint32 aggregate does not override or unlock that gate.

## Reproducibility

The machine-readable aggregate is `summary.json` in this directory. Raw datasets, checkpoints, and rollout payloads remain outside GitHub; dataset artifacts are archived in the private Hugging Face dataset repository.
