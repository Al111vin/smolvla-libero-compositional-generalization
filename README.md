# SmolVLA LIBERO Compositional Generalization

This project studies the compositional generalization and failure diagnosis of SmolVLA in LIBERO robotic manipulation environments.

## Environment

- Ubuntu 22.04
- Python 3.12.11
- PyTorch 2.9.1 + CUDA 12.8
- GPU: NVIDIA GeForce RTX 5090
- LeRobot 0.5.1

## Project Goal

We evaluate whether SmolVLA can generalize to unseen combinations of:

- Object
- Skill
- Spatial region

The main evaluation setting is value-seen but tuple-unseen: each individual object, skill, and spatial value appears in training, but the full object-skill-spatial tuple is unseen during training.

## Planned Pipeline

1. Set up LeRobot, SmolVLA, and LIBERO.
2. Generate Object x Skill x Spatial task combinations.
3. Create value-seen / tuple-unseen train-test splits.
4. Fine-tune SmolVLA under different training coverage levels.
5. Run closed-loop evaluation in LIBERO.
6. Analyze failures into object, skill, and spatial errors.
7. Plot compositional generalization curves and failure decomposition figures.

## Selected First-Stage Task Set

The first-stage experiment uses the `libero_spatial` benchmark suite.

This task set focuses on spatial generalization:
- Object: `black_bowl`
- Skill: `pick_and_place`
- Target object: `plate`
- Variable factor: source spatial relation

The selected task list is saved in:

```text
data/final_task_set.csv
```
## V1 Evaluation Results

The V1 SmolVLA policy was evaluated on all 10 LIBERO-Spatial tasks.

- Suite: LIBERO-Spatial
- Number of tasks: 10
- Success count: 0 / 10
- Success rate: 0.0000
- Average reward: 0.0000
- Average rollout length: 300 steps

This confirms that the V1 evaluation pipeline is functional, but the initial V1 policy does not yet solve the LIBERO-Spatial tasks.

### V1 Failure Analysis

The V1 evaluation pipeline was successfully built and tested across 30 LIBERO tasks, including LIBERO-Spatial, LIBERO-Object, and LIBERO-Goal.

However, the policy achieved 0/30 success in rollout evaluation. To further diagnose this failure, dataset-level action prediction debugging was performed on a training demonstration. The policy showed a high action prediction error, with an MAE of 0.7965.

This indicates that the V1 policy has not learned reliable action prediction yet. The most likely reason is insufficient training, since the initial V1 model was trained only as a short baseline to validate the full training, checkpointing, and evaluation pipeline.

The next step is to train a longer V1.1 model and check whether dataset-level action prediction error decreases before moving to larger compositional generalization experiments.

### V1.1 Longer Training

A longer V1.1 model was trained with 2000 steps instead of the original 200-step V1 baseline.

Dataset-level action prediction improved substantially:

- V1 MAE: 0.7965
- V1.1 MAE: 0.4250

However, V1.1 still achieved 0/10 success on LIBERO-Spatial rollout evaluation. This suggests that longer training improves action prediction, but the current model is still not strong enough to complete full manipulation tasks. The gripper action dimension remains especially difficult, with high prediction error on action_6.

The next step is to further debug gripper behavior and action scaling before training a stronger V1.2 model.

### V1.2 Full Training

A V1.2 full model was trained for 10,000 steps.

Dataset-level action prediction continued to improve:

- V1 MAE: 0.7965
- V1.1 MAE: 0.4250
- V1.2 full MAE: 0.3859

However, V1.2 full still achieved 0/10 success on LIBERO-Spatial rollout evaluation.

This suggests that longer training alone improves action prediction but is not sufficient to solve full manipulation tasks. The gripper action dimension remains the main bottleneck, with action_6 still showing high prediction error.

## V1 Failure Analysis and Task0 Overfit Experiments

After building the initial SmolVLA + LIBERO evaluation pipeline, several V1-stage experiments were conducted to diagnose why the policy failed in closed-loop rollout.

### Summary of V1 Evaluation Results

The V1.2 multi-task policy was evaluated on LIBERO-Spatial:

| Experiment | Dataset / Training Setup | Evaluation | Result |

|---|---|---|---|

| V1.2 | Multi-task LIBERO-Spatial | 10 spatial tasks | 0 / 10 |

| V1.3 | Task0 single-task overfit | Task0 rollout | 0 / 1 |

| V1.4 | Task0 stronger 2-layer model | Task0 rollout | 0 / 1 |

| V1.5 | Task0 gripper-weighted loss | Task0 rollout | 0 / 1 |

| V1.5 + postprocess | Clip + smoothing + binary gripper | Task0 rollout | 0 / 1 |

| V1.6 | Motion + gripper weighted loss | Dataset MSE only | Worse than V1.5 |

### Key Debugging Findings

Several possible failure causes were tested and ruled out:

- Expert action replay achieved reward = 1.0, showing that the LIBERO environment, initial state, and action format are valid.

- Dataset and environment images were compared. The best match was the original image orientation, so there was no obvious vertical flip or RGB/BGR mismatch.

- The rollout state adapter was corrected from using `robot0_eef_quat[:3]` to an axis-angle style orientation representation closer to the dataset `ee_ori`.

- Gripper binary post-processing did not improve rollout success.

- Action clipping and smoothing also did not improve rollout success.

### Dataset Action Prediction Results

The task0 overfit experiments improved dataset-level action prediction, but not enough for successful closed-loop control.

| Model | Overall MAE | action_0 MAE | action_2 MAE | action_6 MAE |

|---|---:|---:|---:|---:|

| V1.3 task0 overfit | 0.3426 | 0.3671 | 0.5406 | 0.9205 |

| V1.4 stronger task0 | 0.3054 | 0.3556 | 0.4473 | 0.7528 |

| V1.5 gripper-weighted | 0.2762 | 0.3123 | 0.4398 | 0.4908 |

| V1.6 motion + gripper weighted | 0.2960 | 0.2954 | 0.4970 | 0.5536 |

V1.5 improved the gripper/action_6 error significantly, but closed-loop rollout still failed. V1.6 did not improve over V1.5.

### Best Checkpoint Scan

A checkpoint scan was performed for V1.5. The best overall dataset MAE was found at:

```text

checkpoint_step_6000.pt

overall MAE = 0.3101
、、、

## Current Status

- [x] AutoDL RTX 5090 instance created
- [x] CUDA and PyTorch verified
- [x] LeRobot installed
- [x] `lerobot-info` passed
- [x] SmolVLA loading test
- [x] LIBERO smoke test
- [x] Task CSV generation
- [x] Train-test split generation
- [x] Split leakage check
- [x] LIBERO task inspection
- [x] LIBERO task annotation draft
- [x] Refine runnable task annotations
- [x] Select final runnable task set
- [x] Final task set check
- [x] LIBERO environment reset test
- [x] Random rollout demo smoke collection
- [x] Read random demo smoke test
- [x] Official LIBERO demonstration download
- [x] Official LIBERO demonstration loading test
- [x] LIBERO PyTorch dataset loading test
- [x] SmolVLA batch input test
- [x] SmolVLA mini training smoke test
- [x] Full fine-tuning v1
- [x] SmolVLA checkpoint loading test
- [x] Evaluation v1 rollout

V1 rollout successfully executed on LIBERO spatial task 0. The policy ran for 300 steps but did not solve the task yet.

- [ ] Failure diagnosis
- [ ] LeRobot-compatible dataset conversion
