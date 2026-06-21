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

## Current Status

- [x] AutoDL RTX 5090 instance created
- [x] CUDA and PyTorch verified
- [x] LeRobot installed
- [x] `lerobot-info` passed
- [ ] SmolVLA loading test
- [ ] LIBERO smoke test
- [ ] Task CSV generation
- [ ] Train-test split generation
- [ ] Fine-tuning
- [ ] Evaluation
- [ ] Failure diagnosis
cat > README.md <<'EOF'
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

## Current Status

- [x] AutoDL RTX 5090 instance created
- [x] CUDA and PyTorch verified
- [x] LeRobot installed
- [x] `lerobot-info` passed
- [ ] SmolVLA loading test
- [ ] LIBERO smoke test
- [ ] Task CSV generation
- [ ] Train-test split generation
- [ ] Fine-tuning
- [ ] Evaluation
- [ ] Failure diagnosis


