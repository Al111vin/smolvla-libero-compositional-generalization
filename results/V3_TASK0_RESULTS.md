# SmolVLA V3 Task-0 Results

## Setup

- Base policy: `lerobot/smolvla_base`
- Training data: 50 task-0 demonstrations converted to LeRobot format
- Training: 10,000 steps
- Checkpoint: `010000`
- Action steps: 50
- Maximum rollout length: 300

## Result

Evaluation from the HDF5 demonstration states produced:

- 36 successes in 50 rollouts
- 72% success rate
- 84.67 mean steps among successful rollouts
- 81.5 median steps among successful rollouts
- Failed demos: 7, 8, 9, 10, 15, 17, 24, 29, 31, 32, 33, 36, 42, 46

Each rollout used seed `12345 + demo_index` and zero stabilization steps.

## Important limitation

This is an in-distribution HDF5-state result, not a standard
current-version LIBERO benchmark result.

The current benchmark initialization failed at 300 steps. Diagnostic
tests showed that the checkpoint predicts the converted training frame
well (action MAE 0.0531), but prediction error rises to 0.2584 on the
reconstructed benchmark frame. None of the current 50 benchmark states
exactly matches the recorded HDF5 state.

The HDF5 metadata references an older Chiliocosm / robosuite
environment. The dataset and current LIBERO installation therefore
have an environment-version mismatch.
