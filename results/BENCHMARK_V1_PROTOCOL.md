# Transparent reconstructed benchmark v1

Status: **approved for use as a new benchmark (2026-09-22)**.

This protocol is the project's current reproducible evaluator for the LIBERO
Spatial task-0 sanity audit. It is a new benchmark and does not retroactively
replace or reinterpret historical V3 results.

## Frozen implementation

- Evaluator: `scripts/eval_v3_task0.py`
- SHA-256: `5fb67f7a535e842214c351d09d368b3d2884deb267023fb51e54d3f5d494a2b3`
- Suite/task: `libero_spatial`, task id `0`
- Environment: `OffScreenRenderEnv`, 128x128 agent-view and wrist images
- State: 15 floats in the order joint position (7), end-effector position (3),
  quaternion-to-axis-angle orientation (3), gripper position (2)
- Action: 7-dimensional OSC action, clipped to `[-1, 1]` immediately before
  `env.step`

## Initialization and determinism

The benchmark initialization is explicit and must not be inferred from output
directory names:

1. Construct the environment.
2. Reset and install the selected benchmark or HDF5 initial state.
3. Apply the configured zero-action wait period.
4. Set `numpy`, PyTorch, and CUDA seeds to `seed_base + initialization_index`.
5. Reset the policy and begin closed-loop control.

For a benchmark sweep, callers pass one fixed `--seed` base (normally
`12345`); the evaluator adds the initialization index exactly once. Passing
`12345 + index` is invalid because it double-applies the index.

Default benchmark settings are `wait_steps=10`, `max_steps=300`, and the
checkpoint's configured `n_action_steps` unless explicitly overridden.

## Success and outputs

A rollout is successful when the environment reports success or cumulative
reward is positive. Each run writes a summary CSV and a per-step action CSV
containing raw, post-processed, and applied actions, plus the effective seed,
initialization source, checkpoint, wait steps, action horizon, and step count.

## Scope and gates

Benchmark v1 is for protocol calibration and the joint32 audit. It does not
unlock Fold 02, does not overwrite legacy results, and does not claim that the
historical V3 14/20 result has been reproduced. A positive-control mismatch is
recorded as a protocol limitation until the historical evaluator/runtime or a
known-good checkpoint under that exact protocol is recovered.
