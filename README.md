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
> **32/36 logical tasks are feasible**; Tasks 15, 17, 33, and 35 remain
> physically infeasible and are excluded from the feasible training set.
> Data collection for all **32/32 feasible tasks is complete**, with **160
> successful episodes** (5 per task). Final QC, the data freeze, provenance
> addendum, and training-loader semantic preflight are complete. The first
> formal Fold 01 training is complete at 90,000 steps, but its strict held-out
> evaluation for the alphabet-soup/put-inside/middle combination (original
> LIBERO task 22) is **0/150**. The separate Task 023 formal evaluation is
> **0/300**. Task1 edge-pinch v2 fine-tuning and a pose-window weighted
> follow-up are also complete, but remain unsuccessful under the strict
> protocol. **Fold 02 remains locked.** Raw HDF5 and LeRobot data remain in
> the private Hugging Face dataset; GitHub stores only lightweight evidence and
> reproducibility metadata.

### Joint32 formal benchmark status (2026-09-21)

The frozen 32-task feasible set has now been evaluated with the strict
closed-loop protocol: 4 checkpoints (`030000`, `060000`, `090000`, `last`) ×
50 deterministic initial states per task, for **6,400 rollouts** total. All
32 task summaries and all 128 checkpoint groups are complete; no
deterministic-algorithm errors were recorded. The aggregate result is
**0/6,400 successes** and mean reward 0.0 at every checkpoint. This is a
negative result for this trained policy under the registered strict protocol,
not evidence that the evaluator is broken. The aggregate evidence is stored
in `results/evaluations/libero36_joint32_strict_aggregate_v1/summary.json` on
the GPU and should be archived as a lightweight JSON artifact.

This aggregate does not change the separately defined Task1 gate and does not
unlock Fold 02. Tasks 15, 17, 33, and 35 remain excluded as physically
infeasible Gate 5 cases.

## Demonstration collection status

The final feasible data collection contains all 32 included tasks, with five
successful episodes per task (160 episodes total). Tasks 15, 17, 33, and 35 are
retained as physically infeasible Gate 5 cases and are not included in training.

| Task | Object × skill × spatial | Episodes | Frames | Status |
|---:|---|---:|---:|---|
| 0 | akita_black_bowl × put_on_top × left | 5 | 3,290 | Complete |
| 1 | akita_black_bowl × put_on_top × middle | 5 | 2,800 | Complete |
| 2 | akita_black_bowl × put_on_top × right | 5 | 2,990 | Complete |
| 3 | akita_black_bowl × put_inside × left | 5 | 2,980 | Complete |
| 16 | white_yellow_mug × push_to × middle | 5 | 3,625 | Complete |

Each pilot reuses the formal Gate 5 replay, samples five distinct deterministic
initial states, records `agentview` and `eye_in_hand` images at 128×128, and
passes the independent HDF5 and LeRobot QC checks. The lightweight collection
records are tracked in:

- `data/training/libero36_feasible_v1/inventory.json`
- `data/training/libero36_feasible_v1/manifest.json`
- `data/demos/libero36_expansion_plan_v1.json`
- `data/demos/libero36_task*_5_success_pilot_v1/manifest.json`
- `data/demos/libero36_task*_5_success_pilot_v1/lerobot_qc.json`

Raw HDF5 and LeRobot directories are intentionally not committed to GitHub.
They are stored in the Hugging Face dataset repository
[`Alllvinnn/smolvla-libero-compositional-generalization`](https://huggingface.co/datasets/Alllvinnn/smolvla-libero-compositional-generalization)
at revision `c237443e`.

To retrieve the externally stored dataset (with the Hugging Face CLI):

```bash
hf download Alllvinnn/smolvla-libero-compositional-generalization \
  --type dataset --revision c237443e --local-dir datasets/hf_snapshot
```

The GitHub repository contains only protocols, manifests, inventories, QC
reports, and acquisition metadata; large binary artifacts remain on Hugging
Face to keep Git history lightweight.

Final reproducibility records:

- [`final data QC`](results/libero36_final_data_qc_v1.json)
- [`data freeze`](results/libero36_data_freeze_v1.json)
- [`data/code provenance addendum`](results/libero36_data_provenance_addendum_v1.json)
- [`training-loader semantic preflight`](results/libero36_training_loader_semantic_preflight_v1.json)
- [`artifact_kind known limitation`](results/libero36_artifact_kind_known_limitation.md)

The frozen feasible manifest remains the canonical data source. The first
Fold 01 training/evaluation pass is complete but does not meet the strict
closed-loop gate (0/150 on its held-out task), so it does not unlock Fold 02.
Task 023 is separately recorded as 0/300. The private Hugging Face dataset is
the canonical location for raw HDF5 and LeRobot data.

### Task1 edge-pinch v2 status

The Task1 edge-pinch v2 branch is retained as a completed diagnostic track,
not as evidence for unlocking Fold 02. The 96-episode dataset passed conversion
and content QC. The baseline fine-tune and the pose-window weighted follow-up
each produced six checkpoints. Under the strict deterministic evaluation
protocol, the baseline achieved **0/30** successes on batch-1 seeds and the
pose-window weighted follow-up achieved **0/15** successes across checkpoints
1500, 2500, and 3000. No deterministic-algorithm errors were observed.

The experiments and failure diagnostics are archived in the lightweight
evidence files at the repository root and in the corresponding Hugging Face
experiment folder. No checkpoint binaries, raw HDF5, or rollout payloads are
stored in GitHub.

### Joint32 seen-task sanity check (2026-09-21)

Before changing the training recipe, three tasks that were included in the
joint32 training set (Tasks 16, 24, and 30) were evaluated under the same
strict protocol on 10 initial states at each of four checkpoints. This was
120 additional rollouts, all completed without evaluator errors, with
**0/120 successes**. The machine-readable summary and analysis are archived
in `results/evaluations/libero36_seen_task_sanity_v1/` and on Hugging Face.

Because the policy also fails on these seen tasks, the current evidence does
not isolate compositional generalization as the sole cause of the earlier
0/6,400 result. Training/checkpoint loading and the observation/action schema
must be audited first. Fold 02 remains locked.

### Joint32 pipeline audit (2026-09-21)

A read-only audit checked the four formal checkpoints, model configuration,
pre/post-processing metadata, training completion, and the existing seen-task
sanity evidence. All checkpoints contain complete model and training-state
files; the model schema is two 128×128 visual inputs, 15 state values, and 7
action values with `STATE/ACTION=MEAN_STD`. The 90,000-step run ended normally
with final logged loss about 0.029. The existing strict seen-task control remains
0/120, so the joint32 failure cannot yet be attributed only to compositional
generalization. The machine-readable audit is
[`joint32 pipeline audit`](docs/audits/joint32_pipeline_audit_final_v1.json).

This evidence does not justify another training run or Fold 02. The next
required step is a known-good single-task control through the identical
training-to-evaluation path, followed by any minimal repair supported by that
control.

### Known-good control audit (2026-09-21)

The historical V3 single-task checkpoint was rerun through the current strict
evaluator with the formal 280-step budget: 5 checkpoints × 5 fixed states =
25 rollouts, **0/25 successes**. This does not prove that the historical
checkpoint is incapable; it shows that the current strict path does not yet
reproduce the historical success protocol. The next action is to reconstruct
the exact historical camera/preprocessing/evaluator path before any additional
joint32 training. See [`known-good control audit`](docs/audits/known_good_control_audit_v1.json).

### Protocol reconstruction plan (2026-09-21)

A staged protocol-reconstruction effort has started. The selected approach is
to reverse-audit historical run manifests, evaluator hashes, protocol files,
checkpoint aliases, and current source differences before any new training.
No Fold 02 run or joint32 retraining is authorized by this phase.
See [`phase A protocol reconstruction`](docs/audits/phaseA_protocol_reconstruction.json).

### Protocol reconstruction phase B (2026-09-21)

An isolated patched evaluator now executes the historical protocol without
mutating the read-only LIBERO task language object. The first fixed-state
smoke completed the full 280-step rollout but returned `success=false`; the
historical success is therefore still not reproduced. The original evaluator
was not modified. See [`phase B protocol reconstruction`](docs/audits/phaseB_protocol_reconstruction.json).

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

The feasible-task set, evidence manifests, and data-collection protocol are
now frozen. Demonstration collection, the formal 32-task joint training, and
the strict 6,400-rollout evaluation are complete; the aggregate result is
reported above.
Candidate work that has not passed exact replay and provenance checks remains
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


## Task1 pose-window weighted training

- Experiment: `task1_edge_pinch_v2_pose_window_weighted_v2`
- Dataset: 77 training episodes from the frozen Task1 edge-pinch v2 LeRobot dataset.
- Method: isolated 3x loss weighting on the audited close-pose window; baseline data and checkpoints were not modified.
- Training: 3000 steps, batch size 8, seed 42; checkpoints 000500 through 003000 completed.
- Closed-loop strict evaluation: checkpoints 1500, 2500, and 3000 across batch1 seeds 555101–555105 produced 0/5 success at every checkpoint (15/15 rollouts completed; no deterministic-algorithm errors).
- Evidence JSON: `results_task1_pose_window_weighted_v2_training.json` and `results_task1_pose_window_weighted_strict_batch1_v1_eval.json`.
- Fold 02 remains locked; this experiment does not change the official Task1 gate.


## Task 022 strict joint32 evaluation

- Task: alphabet soup × put_inside × middle (original LIBERO task 22).
- Joint32 formal training checkpoints 030000/060000/090000/last were evaluated under the strict deterministic protocol on 50 frozen initial states each (200 rollouts total).
- Result: 0/50 success at every checkpoint; mean reward 0.0; no deterministic-algorithm errors or invalid-action records.
- Summary SHA-256: `231b60b963e5e46ce94bd8ea6d3c997cb3699b77f425b0a6db5ccf78e992897e`.
- Evidence JSON: `results_task022_joint32_strict_eval_20260920.json`.
- This result does not unlock Fold 02; Task1/Fold 02 gates remain unchanged.

Phase B found that the historical V3 runtime is not fully recoverable from the current GPU: the old helper used GLX, post-reset seeding, all-zero stabilization actions, and a historical instruction string. Five historical-helper control states were rerun under the available EGL environment and produced 0/5 successes. Joint32 rerun remains gated on protocol parity.

Phase C is gated: the historical evaluator source matching the recorded run hashes is absent from Git history and the current GPU. Joint32 reruns are paused until that source or an equivalent archived runtime is recovered; Fold 02 remains locked.
