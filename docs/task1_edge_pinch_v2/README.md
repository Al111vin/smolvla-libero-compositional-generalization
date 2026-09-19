# Task1 edge-pinch v2 evidence package

This package records the completed Task1 edge-pinch v2 data, training, and
closed-loop diagnosis work. It is evidence and reproducibility metadata only;
it does not change the official LIBERO benchmark or unlock Fold 02.

## Current decision

Task1 strict closed-loop evaluation remains **not passed**. Under the strict
determinism protocol, the batch1 evaluation (six checkpoints × seeds
555101–555105) produced 0/30 successes. Historical results from the older
non-strict protocol are retained only as historical observations.

Fold 02 remains locked.

## Completed artifacts

- 96 edge-pinch v2 privileged-controller demonstrations, QC passed.
- HDF5 → LeRobot conversion verified frame-by-frame (`max_abs_diff=0.0` for
  the action cross-check).
- 77/96 episodes used for the isolated Task1 fine-tune; validation episodes
  remain separate.
- Baseline and chunk-size-10 fine-tunes completed in independent output
  directories.
- Strict closed-loop evaluation, release-action capture, position-error
  analysis, and robustness diagnostics completed.

## Important interpretation boundaries

- `555207` must always be labeled with its checkpoint/group. The baseline
  chunk50 `3000:555207` result and the chunk10 `3000:555207` result are
  different experiments and cannot substitute for one another.
- `per_step_log` includes a reset frame; `action_log` does not. The correct
  mapping is `action_log_index = per_step_log_step - 1`.
- Gripper convention is `open ≈ -1`, `close ≈ +1`.
- Existing diagnosis supports a position/orientation-target problem more
  strongly than a simple release-timing or chunk-size problem, but does not
  establish causality.

## Data policy

Large HDF5, Parquet, checkpoints, videos, logs, and secrets are not stored in
GitHub. Lightweight scripts, manifests, QC summaries, and this README are
versioned here. The Hugging Face dataset repository is the canonical external
location for approved dataset artifacts.

See `task1_edge_pinch_v2_chatgpt_handoff_summary_20260919.md` for the full
handoff, known pitfalls, and the list of experiments that must not be repeated.
