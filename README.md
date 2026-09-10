---
pretty_name: SmolVLA LIBERO Compositional Generalization Benchmark
task_categories:
  - robotics
language:
  - en
tags:
  - robotics
  - libero
  - smolvla
  - vision-language-action
  - lerobot
  - compositional-generalization
  - benchmark
  - imitation-learning
license: other
---

# SmolVLA × LIBERO Compositional Generalization

This repository studies whether **SmolVLA** can execute familiar robotic
manipulation factors in combinations that were never observed during training.
It combines closed-loop LIBERO evaluation, controlled leave-one-combination-out
(LOCO) experiments, failure diagnosis, and the construction of a larger
36-task compositional benchmark.

> **Current status:** the V4 LOCO evaluation is complete. The registered-object
> 36-task benchmark has passed Gates 1–4. Under the strict Gate 5 review,
> **32/36 logical tasks are feasible**; Tasks 15, 17, 33, and 35 are retained
> as physically infeasible cases. Demonstration collection and new SmolVLA
> training have not started.

## Research question

Can a vision-language-action policy generalize to a value-seen but
tuple-unseen combination of:

- **object** — which object to manipulate;
- **skill** — place on top, place inside, or push;
- **spatial region** — left, middle, or right?

Each factor value is represented during training, while the complete held-out
tuple is not. This separates compositional transfer from ordinary
in-distribution task performance.

## Main results

### V3: validated single-task baseline

A SmolVLA policy trained for 10,000 steps on 50 demonstrations of LIBERO-Spatial
task 0 established that the training and closed-loop evaluation pipeline can
produce successful behavior.

| Evaluation | Setting | Success |
|---|---|---:|
| Demonstration initial states | 50 rollouts, `n_action_steps=50` | 36/50 (72%) |
| Official benchmark states | 20 rollouts, `n_action_steps=50` | 9/20 (45%) |
| Official benchmark states | 20 rollouts, `n_action_steps=25` | **14/20 (70%)** |

The 25-action execution horizon was selected before the V4 target-task
evaluation. Full V3 results and ablations are documented in
[`results/V3_TASK0_RESULTS.md`](results/V3_TASK0_RESULTS.md).

### V4: leave-one-combination-out evaluation

Two policies were trained for 90,000 steps. One held out task 3 and the other
held out task 6. Each policy was evaluated on both its unseen target and a
matched seen-task control using the same 50 official benchmark initial states.

| Model | Task | Role | Success |
|---|---:|---|---:|
| `task3_holdout` | 3 | held out | **0/50 (0%)** |
| `task3_holdout` | 6 | seen control | 44/50 (88%) |
| `task6_holdout` | 3 | seen control | 42/50 (84%) |
| `task6_holdout` | 6 | held out | **0/50 (0%)** |

The held-out LOCO macro score was **0%**, compared with **86%** mean success on
the matched seen controls: an **86 percentage-point generalization gap**. The
seen controls show that the result is not explained by a generally broken
policy or evaluator.

This conclusion is limited to two audited target-role folds with one training
seed per fold. See the frozen
[`evaluation protocol`](results/V4_LOCO_EVAL_PROTOCOL.md),
[`formal results`](results/V4_LOCO_RESULTS.md), and
[`post-hoc diagnostic protocol`](results/V4_LOCO_DIAGNOSTIC_PROTOCOL.md).

## Current work: LIBERO registered-object 36-task proxy

The next stage expands the study to a balanced Cartesian benchmark:

```text
4 objects × 3 skills × 3 spatial regions = 36 logical tasks
36 tasks × 4 balanced source layouts = 144 BDDL environments
```

| Factor | Values |
|---|---|
| Object | `akita_black_bowl`, `white_yellow_mug`, `alphabet_soup`, `cream_cheese` |
| Skill | `put_on_top`, `put_inside`, `push_to` |
| Region | `left`, `middle`, `right` |

This is a **registered-object proxy**, not an exact implementation of the
earlier textual object set. The V3/V4 policies are pilot baselines and are not
trained models for this new benchmark.

### Validation gates

| Gate | Requirement | Status |
|---:|---|---:|
| 1 | Generate 36 task rows and 144 BDDL files | Passed |
| 2 | Representative environment, determinism, and camera smoke tests | Passed (12/12) |
| 3 | Complete reset audit | Passed (720/720) |
| 4 | Positive and isolated-negative goal semantics | Passed (96/96) |
| 5 | Exact, safely replayable physical-feasibility trajectories | **32/36 feasible; 4 physically infeasible** |

The strict Gate 5 decision records 32 feasible tasks and four physically
infeasible tasks: 15 (`white_yellow_mug push_to left`), 17
(`white_yellow_mug push_to right`), 33 (`cream_cheese push_to left`), and 35
(`cream_cheese push_to right`). Failed trajectories, contact traces, and
diagnostic attempts for those cases are retained and are not relabeled as
successes. Replacement tasks, if explored, are reported separately and do not
change the original LIBERO-36 count.

Demonstration collection and training remain blocked until the feasible-task
set, evidence manifests, and data-collection protocol are frozen. Candidate
work that has not passed exact replay and provenance checks is deliberately
excluded from the official result.

The full benchmark definition, camera contract, success predicates, gate
criteria, and current evidence are recorded in
[`results/LIBERO_36_PROXY_TASK_PROTOCOL.md`](results/LIBERO_36_PROXY_TASK_PROTOCOL.md).
Current development lives on the
[`libero36-source-xy-redesign`](https://github.com/Al111vin/smolvla-libero-compositional-generalization/tree/libero36-source-xy-redesign)
branch.

## Repository structure

```text
.
├── data/
│   ├── libero_36/                 # task specs, layout specs, and 144 BDDL files
│   ├── manifests/                 # dataset conversion provenance
│   └── splits/                    # compositional train/test splits
├── results/
│   ├── V3_TASK0_RESULTS.md
│   ├── V4_LOCO_*.md
│   ├── LIBERO_36_PROXY_TASK_PROTOCOL.md
│   ├── LIBERO_36_GATE5_OFFICIAL_V6_PROTOCOL.md
│   └── GATE5_TASK15_CONTACT_DIAGNOSIS.md
│   └── libero36_source_xy_redesign_diagnostic/
├── scripts/                       # generation, validation, training, and evaluation
├── patches/                       # compatibility patches
└── requirements.txt
```

## Reproducibility entry points

The repository keeps protocols and compact evidence under version control.
Large checkpoints, datasets, and raw diagnostic artifacts are intentionally
excluded.

```bash
# Generate the 36-task specification and BDDL files in a separate directory.
python scripts/generate_libero_36_bddl.py \
  --output-dir /tmp/libero_36_generated

# Run the frozen full reset and goal-semantics audits.
python scripts/validate_libero_36_envs.py --mode full
python scripts/validate_libero_36_goals.py --mode full
```

Before running an experiment, read its frozen protocol and use the recorded
commit, seeds, task-state matrix, camera contract, and action horizon. After
restoring the externally stored raw V4 rollouts, result summaries can be
regenerated with:

```bash
python scripts/summarize_v4_loco.py
```

## Environment

The main experiments were run with:

- Ubuntu 22.04
- Python 3.12.11
- PyTorch 2.9.1 with CUDA 12.8
- NVIDIA GeForce RTX 5090
- LeRobot 0.5.1
- LIBERO / robosuite / MuJoCo

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

LIBERO assets and pretrained SmolVLA checkpoints are not stored in this
repository and must be obtained separately from their upstream projects.

## License and upstream assets

No standalone license grant has been selected for this benchmark release yet.
The repository therefore uses the Hugging Face `other` license label until a
license is chosen. LIBERO assets, upstream demonstrations, and pretrained
SmolVLA weights are not redistributed here; users must follow the terms of the
respective upstream projects.

## Project roadmap

- [x] Build and validate the SmolVLA–LIBERO training/evaluation pipeline
- [x] Establish a successful single-task V3 baseline
- [x] Run the frozen two-fold V4 LOCO evaluation
- [x] Freeze the 36-task proxy benchmark and pass Gates 1–4
- [x] Complete the strict Gate 5 feasibility decision (32 feasible, 4 infeasible)
- [ ] Freeze the evidence manifests and collection protocol for the 32 feasible tasks
- [ ] Collect balanced demonstrations under the frozen camera contract
- [ ] Create value-seen / tuple-unseen training splits
- [ ] Train multi-seed SmolVLA policies on the 36-task proxy
- [ ] Report compositional generalization curves and failure decomposition

## Scientific interpretation

The current evidence supports a narrow conclusion: under two audited
LIBERO-Spatial LOCO folds, the policy learned seen combinations but did not
transfer to the held-out compositions. It does not establish that SmolVLA
cannot generalize compositionally in every task family or training regime.

Likewise, the 36-task proxy is still a benchmark-construction effort. Until
Gate 5 is complete and demonstrations are collected under the frozen protocol,
it must not be presented as a trained-model evaluation result.
