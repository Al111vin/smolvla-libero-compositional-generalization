# Task1 pose-target intervention: phase 2 design

## Status

Design only. No training, evaluation, data mutation, or Fold 02 unlock is
performed by this document.

## Motivation

Strict batch1 evaluation remains 0/30. Existing action captures show that the
chunk10 `3000:555207` failure has a sustained gripper opening but a large
position/pose error. The current evidence therefore favors a grasp/pose-target
problem over a simple release-timing problem. This is a hypothesis, not a
causal result.

## What has already been ruled out

- No consistent improvement from inference `n_action_steps` values 1, 5, 10,
  25, or 50.
- No improvement from a chunk-size 50→10 training intervention.
- HDF5→LeRobot action conversion is lossless (`max_abs_diff=0.0`).
- The strict official Task1 result remains 0/30; Fold 02 stays locked.

## Implementation constraint

The installed SmolVLA trainer uses a uniform mean over valid time steps and
action dimensions. It has no phase-weight or action-channel-weight config.
Its standard data loader also has no weighted sampler hook. A pose/grasp
weighting experiment therefore requires an explicitly isolated custom training
entry point or a predeclared dataset-sampling transformation; it must not be
silently approximated by changing an unrelated config field.

## Proposed minimum experiment

1. Freeze the existing 77-episode train split and all validation episodes.
2. Define closure/lift windows from the recorded demonstration signals using a
   deterministic, documented rule. In the frozen 96-demo audit, stable close
   commands begin at frames 36–42 (median 37), reopen commands begin at
   frames 93–105 (median 97), and the close→reopen interval is 57–65 frames
   (median 60). A proposed first window is `[close-10, close+20]`, clipped to
   the episode, with a proposed 3× sampling multiplier. These are defaults for
   review, not an authorization to train.
3. Produce a new, independently named training variant that oversamples only
   those windows, while retaining the original action labels and images.
4. Keep base checkpoint, optimizer, seed, total updates, and evaluation
   protocol fixed.
5. Run the same strict closed-loop evaluation on a predeclared seed set.
6. Compare success, release distance, grasp/lift rate, and action stability;
   loss decrease alone is not a success criterion.

Before implementation, the window definition, sampling multiplier, output
directory, and strict evaluation seeds must be reviewed and confirmed.

## Synchronization rule

After the design is reviewed and each experiment is completed, synchronize the
lightweight code, README, manifests, and QC summaries to GitHub and Hugging
Face. Do not upload checkpoints, raw HDF5, Parquet, videos, logs, or secrets
without a separate explicit decision.
