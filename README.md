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
- [ ] Demonstration collection
- [ ] LeRobot dataset loading test
- [ ] Fine-tuning
- [ ] Evaluation
- [ ] Failure diagnosis
