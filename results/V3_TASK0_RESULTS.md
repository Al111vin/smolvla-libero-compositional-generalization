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

## Official benchmark result

Evaluation on the first 20 fixed LIBERO benchmark initial states:

- 9 successes in 20 rollouts
- 45% success rate
- 88.22 mean steps among successful rollouts
- 88 median steps among successful rollouts
- Successful states: 0, 2, 4, 6, 12, 16, 17, 18, 19
- Failed states: 1, 3, 5, 7, 8, 9, 10, 11, 13, 14, 15

Each rollout used 10 stabilization steps, 50 action steps, and seed
`12345 + init_index`.

## Interpretation

The 72% HDF5-state score measures performance on demonstration
initial states and is therefore an in-distribution result. The 45%
benchmark score measures generalization to LIBERO's fixed benchmark
initial states.

A previous one-off benchmark init-0 rollout failed, while the
standardized fixed-seed init-0 rollout succeeded. Single-rollout
results are therefore sensitive to SmolVLA's stochastic action
sampling.

The checkpoint predicts the converted training frame accurately
(action MAE 0.0531), while prediction error rises to 0.2584 on one
reconstructed benchmark frame. This indicates an observation
distribution shift. HDF5 metadata also references older Chiliocosm /
robosuite paths, but state mismatch alone does not prove a version
incompatibility.
