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

### Teacher-recommended single-task control (2026-10-02)

The historical native LIBERO Spatial task-0 V3 checkpoint succeeded in **5/5**
repeats at one fixed initialization and evaluation condition, with reward 1 in
each run. This establishes outcome repeatability at that condition; trajectories
were not identical. Official native task data was reconstructed from the LIBERO
Hugging Face source and verified against its published file hash: 50 episodes
and 5,068 frames. An independent conversion audit found exact float32 agreement
for every action and state, exact pixel agreement for both cameras, and passing
samples through the current LeRobot loader and preprocessing.

The bounded **10,000-update single-task control completed successfully** through
the current official training entry with the existing episode-compatibility
patch. Checkpoints at 2,500, 5,000, 7,500, and 10,000 updates were verified in
an independent output directory. The run used the current joint32 initialization
and optimizer setup. LeRobot automatically rescaled the 90,000-update scheduler
to this shorter run (warmup 3,000→333; decay horizon 90,000→10,000), so this is
a current-pipeline capability test, not an exposure-matched causal comparison to
the 90,000-update joint32 run. The first matched-evaluation launch exposed a
missing EGL environment setting before any rollout was recorded; that failed
attempt is preserved separately and excluded. The EGL-configured retry finished
with **0/5** successes on the repeated fixed initialization and **8/20** on the
paired 20-initialization set; the recovered historical checkpoint scored 5/5
and 13/20 on those respective comparisons. The preregistered stable-single-task
gate is not met, so no task-count scaling will start. This diagnostic does not
change the registered LIBERO-36 result or unlock Fold 02. Per-initialization
outcomes and checkpoint provenance are in
`results/teacher_native_task0_current_pipeline_10k_eval_summary_20261002.json`.
Detailed configuration and evidence are in
`results/teacher_native_task0_stageBC_decision_20261002.json`.

The next bounded diagnostic holds that current data/training recipe fixed and
changes only initialization, using the recovered historical task0 starting
base rather than the joint32 checkpoint. No-gradient construction through the
same LeRobot policy factory successfully loaded the historical base under the
current model configuration (450,046,176 parameters; no training/output was
performed by the check). This follow-up is not a historical recipe
reproduction: the old dataset revision and several optimizer/preprocessing
settings are unavailable or differ. Task-count scaling remains blocked until
single-task success gates pass. See
[`training recipe audit`](results/teacher_native_task0_training_recipe_difference_audit_20261002.json)
and [`unattended plan`](docs/teacher_control_unattended_plan_20261002.md).
The independent 10,000-step run and its bounded evaluation orchestrator have
now been launched on the current GPU; the first live check observed training
underway. Results will be reported only after the exit status, checkpoints, and
all 25 planned rollouts are verified.
The evaluation now has a separate CPU-only completion watcher and CSV
validator; replaying the validator against the prior 25-rollout CSV set
reproduced its recorded 0/5 fixed and 8/20 paired results, including the
Wilson interval. This validator does not change the evaluation protocol.
The historical-base-initialized 10,000-update follow-up has completed with all
four planned checkpoints. At checkpoint 10,000, the five repeats at fixed
benchmark init 3 produced **0/5** successes; the matched 20-init set produced
**9/20** (Wilson 95% CI **[0.2582, 0.6579]**). This is one more success than the
prior current-pipeline initialization (8/20), but the paired exact McNemar
test is **p=1.0**, so this is not evidence of a reliable improvement. The
pre-registered single-task gate (fixed 5/5 and paired at least 11/20) is not
met. This is a bounded current-recipe initialization comparison, not a
reproduction of the unavailable historical training recipe; task-count scaling
remains blocked and Fold 02 remains locked.

The initial sequential evaluator stopped after the five fixed repeats and
paired init 0 because of a transient Hugging Face Hub processor-resolution
disconnect. Those original outputs and the error log were preserved. A local
cache-only processor check passed; an isolated recovery run reused the five
completed fixed repeats and paired init 0, then reran missing paired inits
1--19 under the same evaluator, seeds, and protocol. All **25/25** records
passed validation. The validated summary is
[`historical-base initialization evaluation`](results/teacher_native_task0_base_init_current_recipe_10k_eval_summary_20261002.json);
the retry-safe recovery runner is
[`run_teacher_native_task0_base_init_eval_recovery_20261002.sh`](scripts/run_teacher_native_task0_base_init_eval_recovery_20261002.sh).

A read-only action-trace audit of the 20 paired initializations found similar
command smoothness in the shared first 60 steps (mean gripper sign-switch rate
3.95 vs. 3.54 per 100 steps; position-action delta 0.197 vs. 0.193; rotation
action delta 0.033 vs. 0.034, success vs. failure). Across each full rollout,
the failure group had more gripper sign switches (12.37 vs. 6.10 per 100 steps)
and larger mean action deltas (0.244 vs. 0.208 on dimensions 0--2; 0.047 vs.
0.038 on dimensions 3--5). Because successful episodes often terminate early
while failures run to the 300-step cap, the full-rollout contrast is
length-confounded. This is exploratory correlation, not evidence that action
oscillation causes failure. The first-60 comparison instead suggests there is
no clear early action-quality separation in this small sample. See the
[`action-trace audit`](results/teacher_native_task0_base_init_action_trace_audit_20261002.json)
and its reproducible
[`audit script`](scripts/audit_teacher_native_task0_base_init_action_traces_20261002.py).
No additional training or task scaling has been started.

The fixed-init checkpoint progression diagnostic is complete: checkpoints
2,500, 5,000, and 7,500 each scored **0/5**, matching the existing 10,000-step
reference at **0/5**. All 15 rollouts passed protocol validation; every rollout
ran to the 300-step cap with reward 0. This single-initialization diagnostic
shows no fixed-init success window across the sampled training trajectory, but
does not establish why training failed, replace the paired-20 result, or
authorize task-count scaling. See the
[`checkpoint progression summary`](results/teacher_native_task0_checkpoint_fixed_repeats_20261002.json).
The isolated runner and validator are
[`checkpoint repeat runner`](scripts/run_teacher_native_task0_checkpoint_fixed_repeats_20261002.sh)
and
[`checkpoint repeat summarizer`](scripts/summarize_teacher_native_task0_checkpoint_fixed_repeats_20261002.py).

The recipe audit verified that the current base-initialized run keeps the same
50-episode native data and loader, while its peak learning rate is 10x lower
than the recovered historical successful recipe (1e-5 vs. 1e-4); the exact
historical dataset revision and several other recipe details are unavailable.
After comparing a 90k extension (deferred for cost), a multi-factor historical
recipe recreation (deferred as confounded), and a bounded single-factor test,
the next selected diagnostic is a 10k single-task run scaling the complete
cosine learning-rate schedule by 10x. It keeps the initialization, data,
training code, seed, update count, and evaluation protocol fixed in a new output
directory. This is not a full historical reproduction. The design is recorded
in [`LR-scale ablation design`](results/teacher_native_task0_lrscale_ablation_design_20261002.json).
The design was subsequently executed; see the dated outcome below.

### 2026-10-02 — Single-task LR-scale outcome and gate

The LR-scale run completed 10,000 updates with exit code 0. Its loss was finite
through training and fell from 0.406 at 2k updates to about 0.10 near 9k. On the
registered strict task-0 evaluator, the 10k checkpoint scored **5/5** at the
repeated fixed init and **12/20** on paired initializations (Wilson 95% CI
**[0.3866, 0.7812]**). The fixed-init checkpoint progression run for the prior
lower-LR recipe scored 0/5 at steps 2.5k, 5k, 7.5k, and 10k; its final paired
score was 9/20. The LR-scale result clears the predeclared single-task gate
(fixed 5/5 and paired at least 11/20), but the paired exact McNemar comparison
against the current-pipeline reference is p=0.2188, so this is not a claim of a
statistically established improvement or a full reproduction of the
historical recipe. See the
[`LR-scale strict evaluation summary`](results/teacher_native_task0_lrscale_ablation_20261002.json).

The project may now advance to designing the next **4-task, exposure-matched
diagnostic stage**. Task IDs/data availability and per-task update exposure
must be verified before training; no 4-task run has started. This gate does not
alter the LIBERO-36 benchmark outcome and does not unlock Fold 02.

#### Stage E design decision (2026-10-02)

The same-frozen-data comparison was selected to avoid confusing multi-task
interference with lower per-task training exposure. Stage E1 is now complete:
the task-0-only model trained for 10,000 updates on five episodes (3,290 frames)
from the frozen 32-task dataset, then received strict evaluation on 50 frozen
custom-task initial states. It scored **0/50** (95% Wilson CI **[0, 0.0713]**),
so the predeclared 28/50 gate failed and Stage E2 four-task training was not
started. This does not alter the formal benchmark result or Fold 02.

The data audit compared the five frozen task-0 episodes with the recovered
source HDF5: all per-frame actions and 15-D states matched exactly; agentview
and wrist images matched exactly at three sampled frames per episode (not an
exhaustive image comparison). This weakens the data-conversion-damage
explanation but does not isolate the training or inference cause. The native
`libero_spatial` task-0 control is a different task: its recovered V3 checkpoint
repeatedly succeeds on native benchmark init 3 (5/5), while it scored 0/50 when
cross-applied to the custom task-0 held-out states. That cross-task result is
not a valid positive control and must not be used to blame the strict evaluator.

The diagnostic-only smoke and five-start paired run completed. With 10 added
open-gripper wait actions, both checkpoints scored 0/5; because the collector
stores HDF5 states after passive settle, that was not an exact action-start
replay. A zero-wait paired rerun then also scored 0/5 for both checkpoints
(all rewards 0; all reached 280 steps). The historical checkpoint was trained
on native `libero_spatial` task 0, so its custom-task failures are cross-task
probes, not a valid positive control. The Stage E1 checkpoint is task-matched.
All five source HDF5 episodes are marked successful, terminal-held, exact-action
replays, and passed QC.

A sparse teacher-forced audit of 665 observations across the five training
episodes found overall one-step action MAE **0.0107**; per-channel error was
about 5–15% of each channel's mean absolute expert action. This is in-sample
diagnostic evidence only: it does not establish training causality or explain
why closed-loop rollout fails. The next check is horizon-wise multi-step action
chunk fit on the same expert sequences, now completed: across 615 overlapping
50-step windows the full-chunk MAE was **0.00978**. The first 25 steps (the
execution prefix under `n_action_steps=25`) had MAE **0.01041**, versus
**0.00916** for the remaining 25; all five demonstrations also had a lower
last-10-step MAE than first-10-step MAE. Thus this in-sample audit does **not**
show error growing through the predicted chunk. It still does not explain the
0/50 held-out or 0/5 zero-wait closed-loop outcomes: windows overlap and come
from the same five training episodes, so the result is neither held-out nor
closed-loop evidence. No additional training or task scaling will start from
this alone. Evidence is in the
[`10-wait demo-start diagnostic`](results/teacher_task0_demo_start_diagnostic_20261003_v1_summary.json),
[`zero-wait diagnostic`](results/teacher_task0_demo_start_diagnostic_20261003_v1_nowait_summary.json),
[`teacher-forced fit summary`](results/teacher_task0_teacher_forced_action_fit_v1_20261003.json),
[`fit audit script`](scripts/audit_task0_teacher_forced_action_fit_v1.py),
[`full-chunk fit summary`](results/teacher_task0_chunk_fit_horizon_v1_20261003.json),
[`full-chunk audit script`](scripts/audit_task0_action_chunk_horizon_fit_v1.py),
[`Stage E design`](results/teacher_directed_task_scaling_stageE_design_20261002.json),
and [`Stage E1 control-chain result`](results/teacher_task0_control_chain_stageE1_result_20261003.json).
The next bounded check is exhaustive read-only image-pixel parity for every
frame in the five source HDF5 episodes versus the corresponding frozen
LeRobot task-0 episodes. It is complete: all 3,290 frames in each of the two
cameras matched pixel-for-pixel, and episode correspondence was established by
exact full-sequence action/state matches rather than file order. Together with
the earlier all-frame action/state parity, this closes raw source-to-dataset
conversion parity for the five demonstrations. It does **not** verify the
closed-loop failure. A follow-up read-only train-versus-evaluation
preprocessing parity check on matched frames is complete. Across 25
observations sampled from
all five task-0 demonstrations, raw loader/evaluator images and stored 15-D
states matched exactly, and the saved training preprocessor produced identical
model inputs on both paths: zero mismatches and maximum absolute difference
0.0, including both cameras, state, language tokens/mask, and task text. The
training and strict-evaluation code use the same saved preprocessor factory.
This closes recorded-frame preprocessing parity but not live simulator state
construction. A follow-up exact action replay used each episode's recorded
seed and the collector's 20 zero-action settle steps. All 3,290 observations
and reward sequences matched the source exactly. On the same live quaternions,
the collector's robosuite axis-angle conversion and the evaluator's
quaternion-normalizing helper differed by at most 1.2e-7; this gives no
evidence that conversion mismatch explains failure on the demonstrated
trajectories. Policy-generated states remain untested. The StageE1
teacher-forced checkpoint audit is now complete on 665 observations from the
same five training demonstrations at 2,500/5,000/7,500/10,000 updates, using
the same observation-keyed action-sampling noise. Overall in-sample action MAE
decreased from 0.02648 to 0.00869; a full 10,000-step repeat reproduced the
same result byte-for-byte. This is not held-out or closed-loop evidence and
does not change the StageE1 result (0/50 strict successes; 0/5 zero-wait
demo-start successes), so StageE2 and Fold02 remain blocked. The audit also
found that earlier unseeded one-step estimates varied because SmolVLA samples
initial action noise when no noise tensor is supplied; strict deterministic
operator settings alone do not fix that random input. See the
[`checkpoint learning-curve summary`](results/teacher_task0_checkpoint_learning_curve_v1_20261003.json),
the four [`2,500-step`](results/stageE1_seeded_full_002500.json),
[`5,000-step`](results/stageE1_seeded_full_005000.json),
[`7,500-step`](results/stageE1_seeded_full_007500.json), and
[`10,000-step`](results/stageE1_seeded_full_010000.json) outputs, and the
[`seed-aware audit script`](scripts/audit_task0_teacher_forced_action_fit_v1.py).
The next step is a read-only provenance audit of the historical purportedly
successful checkpoint and its original task/evaluator/seed conditions before
considering any training change.
Evidence is in the
[`full-frame image parity summary`](results/teacher_task0_training_image_parity_v1_20261003.json)
and [`audit script`](scripts/audit_task0_training_image_parity_v1.py), plus the
[`preprocessing parity summary`](results/teacher_task0_model_input_preprocessing_parity_v1_20261003.json)
and [`audit script`](scripts/audit_task0_model_input_preprocessing_parity_v1.py).
The exact-replay conversion result and its script are in
[`quaternion conversion summary`](results/teacher_task0_live_quaternion_conversion_v2_full_20261003.json)
and [`audit script`](scripts/audit_task0_live_quaternion_conversion_v2.py).

### Task-balanced sampling proposal (2026-09-23)

The joint32 dataset has five episodes per task, but frame counts range from
1,280 to 4,900 per task (about 3.8×). An offline audit verified that a
task-uniform sampler can cover all 32 task indices without modifying the
dataset. This remains a design-only proposal: no loader integration or
training has started. The preferred next experiment is an independently
outputted task-balanced sampler with a loader smoke test and the same strict
V3 EGL evaluation gate. Fold 02 remains locked.

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

### Phase H terminal-window weighted repair (2026-09-23)

The terminal-window weighted minimal-repair model completed 90,000 training
steps and was evaluated with the recovered V3 evaluator under EGL. The strict
diagnostic run covered checkpoints `030000`, `060000`, and `090000`, five
benchmark initializations per checkpoint (15 rollouts total), with
`n_action_steps=25` and a 300-step budget. All three checkpoints scored **0/5
successes**, reward 0.0, and 300 steps on every rollout; no deterministic
algorithm errors occurred. This diagnostic result does not alter the official
protocol or unlock Fold 02. Checkpoint files and raw rollout data remain on the
GPU; GitHub contains only the lightweight summary
`results/phaseH_terminal_window_weighted_v1_strict_eval_v2_summary_20260923.json`.

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

### Phase G: action-target audit before any further repair training (2026-09-22)

The calibrated joint32 model and the gripper-weighted minimal repair both
remain at 0 success on their strict task-0 audits. A trainer-side replay audit
covered all 160 frozen episodes (five per task across 32 tasks) and found no
loader, state/action shape, task-text, image-range, or normalization mismatch.
Therefore the next recommended step is a read-only action-target audit, not
another blind training sweep. It will measure per-dimension action scale and
sparsity, late-stage/release-window coverage, and 50-step chunk alignment
before proposing one bounded objective/data change. Fold 02 remains locked and
no training is started by this plan.
See [`Phase G objective/data redesign options`](results/phaseG_objective_data_redesign_options_v1_20260922.json).

### Phase G action-target audit result (2026-09-22)

The read-only audit covered 101,660 frames from all 160 frozen episodes and
all 32 tasks. Every action dimension was finite and populated; no collapsed
channel, loader mismatch, or isolated task-group anomaly was identified.
Because the audit did not reveal one bounded target-construction defect, no
second repair training run is authorized from this evidence alone. The next
training change, if approved later, must be a separately designed and
versioned objective ablation. Fold 02 remains locked.
See [`Phase G action-target audit decision`](results/phaseG_action_target_audit_decision_20260922.json).

### Phase H: bounded objective ablation design (2026-09-23)

The audit does not justify another broad sweep. Three single-variable options
were compared: terminal-window loss weighting, temporal smoothness, and
release-event chunk reweighting. The recommended candidate is terminal-window
weighting because it targets the observed late-stage failure while preserving
the validated state/action semantics and evaluator. This is a design only:
training has not started and requires explicit approval. See
[`Phase H objective ablation options`](results/phaseH_minimal_objective_ablation_options_v1_20260923.json).

### Transparent benchmark v1 (approved 2026-09-22)

#### Task-balanced training follow-up (2026-09-25)

The task-balanced joint32 training completed cleanly for 90,000 steps and
produced independent 030000, 060000, and 090000 checkpoints. Under the same
EGL-backed benchmark-v1 evaluator (`wait_steps=10`, `n_action_steps=25`,
`max_steps=300`, task 0, five fixed benchmark initializations), the new model
achieved **0/15 successes**: 0/5 at each checkpoint, with reward 0.0 and the
full 300-step budget on every rollout. This is diagnostic evidence only; it
does not change the Fold 02 gate. See
[`task-balanced benchmark summary`](results/phaseL_joint32_task_balanced_benchmark_v1_5init_summary_20260925.json).

This is a negative intervention result: task balancing did not improve the
calibrated task-0 benchmark, so the sampler change will not be expanded into
another training sweep. The next gate is a new, explicitly designed temporal
training-objective or protocol experiment; no additional repair training is
started from this result alone, and Fold 02 remains locked.

### Phase M temporal-objective options (2026-09-29)

After the negative task-balanced result, three next-step options were compared:
expand evaluation only, weight release-transition frames, or weight the final
10--15% terminal window. The recommended candidate is terminal-window
loss-weighting because it is the smallest isolated test of the remaining
temporal-supervision hypothesis and does not repeat failed sampler,
gripper-weighting, or action-horizon changes. This is a design artifact only;
no new training has started. See
[`Phase M options`](results/phaseM_temporal_objective_options_20260929.json).

The project now has an explicitly frozen, transparent reconstructed evaluator
for the LIBERO Spatial task-0 sanity audit. It is a **new benchmark** and does
not rewrite historical V3 numbers. The protocol, seed rule, state/action
schema, and evaluator hash are recorded in
[`results/BENCHMARK_V1_PROTOCOL.md`](results/BENCHMARK_V1_PROTOCOL.md) and
[`results/benchmark_v1_manifest.json`](results/benchmark_v1_manifest.json).
The historical V3 evaluator has not been recovered; therefore a mismatch with
the legacy 14/20 result remains a documented limitation rather than evidence
that a model or training run is defective.

The first benchmark-v1 smoke control completed on the GPU using the existing
010000 control checkpoint: one benchmark initialization, 300 control steps,
`n_action_steps=25`, seed 12345, success `false`, reward `0.0`. This is a
pipeline smoke result only; it is preserved under
[`results/benchmark_v1_smoke_control_20260922/`](results/benchmark_v1_smoke_control_20260922/)
and does not alter any legacy result or Fold 02 gate.

The first joint32 benchmark-v1 minimal audit is also complete: checkpoints
030000, 060000, and 090000 were each evaluated on benchmark initialization 0
with the frozen settings above. All three ran the full 300 control steps with
reward `0.0` and success `false` (0/3). This is a calibration-stage result
with one initialization, not a sufficient basis for retraining or Fold 02.
The machine-readable aggregate is
[`results/benchmark_v1_joint32_minimal_20260922/summary.json`](results/benchmark_v1_joint32_minimal_20260922/summary.json).

The expanded benchmark-v1 audit is complete as well: 3 checkpoints × 5
benchmark initializations = 15 rollouts. Results were 0/5 at each checkpoint
(030000, 060000, 090000), all with reward `0.0` and the full 300-step budget.
This confirms the observed failure on a small initialization set, but the
missing historical positive control still prevents attributing it to the
model or choosing a retraining intervention. See
[`results/benchmark_v1_joint32_full_20260922/summary.json`](results/benchmark_v1_joint32_full_20260922/summary.json).

As a calibration check, the historical V3 010000 checkpoint that is associated
with the legacy 14/20 report was also run on five benchmark-v1 initializations;
it produced 0/5. This is a confirmed protocol mismatch, not a contradiction
of the legacy result, because the historical evaluator/runtime remains
unrecovered. The evidence is preserved in
[`results/benchmark_v1_historical_v3_candidate_20260922/summary.json`](results/benchmark_v1_historical_v3_candidate_20260922/summary.json).
Repeating the same five initializations with `n_action_steps=50` also produced
0/5, so the mismatch is not explained by the action horizon alone. Those
results are preserved under
[`results/benchmark_v1_historical_v3_n50_candidate_20260922/summary.json`](results/benchmark_v1_historical_v3_n50_candidate_20260922/summary.json).

An archived positive-result summary is present in the repository (legacy V3:
14/20 benchmark and 50/50 HDF5-initialized), but its original checkpoint path,
task-0 HDF5, and exact evaluator/runtime are missing. It is therefore historical
evidence rather than a reproducible benchmark-v1 control. The artifact audit is
recorded in
[`results/historical_v3_positive_summary_artifact_audit_20260922.json`](results/historical_v3_positive_summary_artifact_audit_20260922.json).

### Phase D decision gate (2026-09-22)

The current recommendation is **no retraining and no inference-side change
yet**. The joint32 matrix is 0/15, but benchmark v1 has not reproduced a
known-good positive control, so changing the model would confound an unresolved
protocol question with a model intervention. The four-option comparison and
the next gate are recorded in
[`results/phaseD_benchmark_v1_decision_gate_20260922.json`](results/phaseD_benchmark_v1_decision_gate_20260922.json).
Fold 02 remains locked.

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


## Phase C reconstructed V3 audit (2026-09-21)

A reconstructed EGL evaluator (`scripts/eval_v3_task0.py`) was run on joint32 checkpoints 030000, 060000, and 090000 for LIBERO task 0, benchmark init indices 0–4, with `wait_steps=10`, `n_action_steps=25`, and `max_steps=280. All 15 rollouts completed with success 0/15 and reward 0. This is a reconstructed protocol result, not a byte-identical reproduction of the historical evaluator; Fold 02 remains locked. Evidence: `results/phaseC_joint32_v3_reconstructed_eval_v2_summary.json`.


## Phase D decision gate (2026-09-21)

The Phase C reconstructed EGL audit produced 0/15 joint32 successes on task 0 across checkpoints 030000/060000/090000. This is insufficient to justify an inference fix or retraining because no positive-control checkpoint has been validated under the same evaluator. The current recommendation is no retraining until a positive control or exact historical evaluator is recovered; Fold 02 remains locked. The public base model control also produced 0/5 under the same evaluator, confirming execution but not supplying a positive policy control. Evidence: `results/phaseD_decision_gate_v1.json`.

## Phase B seed-corrected V3 checkpoint control (2026-09-21)

The historical V3 checkpoint loco_fold01_formal_v1/checkpoints/010000 was rerun on all 20 benchmark initial states with the transparent reconstructed evaluator. The command passed a fixed seed base of 12345; the evaluator then applied seed = 12345 + init_index, matching the repository V3 documentation. The run completed 20/20 with 0/20 successes, zero reward, and 300 steps on every rollout. This is not a positive control and does not reproduce the repository historical 14/20 result; it is evidence that the historical evaluator/runtime remains materially different from the reconstructed EGL evaluator. The earlier trial that passed 12345 + init_index on the command line was invalid for this seed convention and is retained only as a protocol-debug record. Evidence: results/phaseB_v3_checkpoint010000_reconstructed_benchmark_v2_seedbase_v1_summary.json. Fold 02 remains locked.

### Recovered V3 positive control and shared joint32 comparison (2026-09-22)

The old GPU was recovered and supplied the original V3 010000 checkpoint and task-0 HDF5. Their SHA256 hashes were verified after transfer to the current GPU. Using the same repository evaluator (`scripts/eval_v3_task0.py`), task 0, benchmark init 0, seed 12345, 10 wait steps, 25 action steps, and 300-step budget, the recovered historical checkpoint succeeded (`reward=1.0`, 82 steps). Under that identical evaluator and initialization, joint32 checkpoints 030000, 060000, and 090000 each failed (`reward=0`, 300 steps). This is a positive-control sanity comparison, not yet the full historical 20/50-initialization reproduction; Fold 02 remains locked. Evidence: `results/phaseB_C_recovered_v3_positive_control_and_joint32_init0_20260922.json`.
### Full calibrated joint32 audit (2026-09-22)

Using the same recovered V3 evaluator, EGL runtime, task-0 benchmark initializations 0--19, seeds 12345--12364, `wait_steps=10`, `n_action_steps=25`, and `max_steps=300`, the historical V3 control achieved 12/20 successes. Joint32 checkpoints `030000`, `060000`, and `090000` each achieved 0/20; all 20 rollouts per checkpoint reached the step limit with zero reward. This is an evaluation audit, not a causal proof of one root cause and does not unlock Fold 02. Evidence: `results/phaseC_joint32_full60_and_phaseD_options_20260922.json`.

The recommended Phase D follow-up is a minimal, isolated training-contract ablation (state/action ordering and normalization, camera preprocessing, and task-language/task-id mapping checked one variable at a time). No training has started from this plan; see `results/phaseD_minimal_training_contract_ablation_plan_20260922.json`.

The concrete draft currently recommends changing only training-time `n_action_steps` from 25 to 50, matching the recovered V3 checkpoint configuration, with a fresh output directory and the calibrated evaluator held fixed. It is a draft only and requires explicit approval before any training; see `results/phaseD_n_action_steps_ablation_config_draft_v1.json`.

### Joint32 n_action_steps=50 ablation sanity check (2026-09-22)

The approved minimum ablation trained a new checkpoint family with
`n_action_steps=50` and was evaluated with the strict evaluator on three seen
tasks (16, 24, and 30), four checkpoints (`030000`, `060000`, `090000`, and
`last`), and 10 fixed initial states per task/checkpoint. All 120 rollouts
completed without deterministic-algorithm errors, but **0/120 succeeded**
(0/40 for each task and checkpoint). This is diagnostic evidence only; it does
not change the official gate and does not authorize Fold 02.

Primary evidence:
[`n_action_steps=50 sanity summary`](results/phaseC_joint32_n_action_steps50_seen_sanity_v2_summary_20260922.json).

### Recovered V3 positive-control rerun (2026-09-22)

The recovered historical V3 checkpoint was rerun with the reconstructed
evaluator using the GPU's EGL backend. Across 20 benchmark initial states at
`wait_steps=10`, `n_action_steps=25`, and `max_steps=300`, it achieved **13/20
successes** (mean reward `0.65`). This establishes a runnable positive control
for the recovered asset and EGL runtime; it is separate from the earlier
same-named checkpoint candidate whose archived artifacts were incomplete.

Evidence:
[`recovered V3 positive-control summary`](results/phaseB_historical_v3_recovered_asset_full20_egl_summary_20260922.json).

### Calibrated joint32 task-0 sanity check (2026-09-22)

Using the same `eval_v3_task0.py`, EGL runtime, benchmark initialization rule,
`wait_steps=10`, `n_action_steps=25`, and `max_steps=300` as the recovered V3
positive control, joint32 checkpoints `030000`, `060000`, and `090000` were
each tested on five benchmark states. The result was **0/15 successes** (0/5
for every checkpoint), with no evaluator or rendering error. This confirms the
joint32 failure on this task is not caused by the earlier missing-EGL setup;
it remains a model/checkpoint result under the calibrated protocol.

Evidence:
[`calibrated joint32 sanity summary`](results/phaseC_joint32_calibrated_eval_v1_summary_20260922.json).

### Phase E minimal gripper-weighted repair (2026-09-22)

The approved Option B repair trained an isolated checkpoint family with a
3x loss weight on the gripper action dimension; all other data, optimizer,
seed, pretrained initialization, and training schedule settings were held
fixed. Training completed cleanly at 90,000 steps and produced independent
`030000`, `060000`, and `090000` checkpoints. Under the calibrated EGL
`eval_v3_task0.py` protocol (`wait_steps=10`, `n_action_steps=25`,
`max_steps=300`, five benchmark initializations per checkpoint), the repair
model achieved **0/15 successes**. There were no evaluator/runtime errors;
all rollouts reached the 300-step limit with zero reward. This minimal repair
does not improve the calibrated task-0 result, so no further weighting sweep
is recommended before revisiting the broader training contract. Fold 02
remains locked.

Evidence:
[`Phase E weighted strict summary`](results/phaseE_joint32_gripper_weighted_strict_v1_summary_20260922.json).

### Next-step gate after Phase E (2026-09-22)

The gripper-weighted repair is now a negative ablation (`0/15`) and should
not be expanded into a weight sweep. The recommended next step is a
**read-only trainer-side task-0 action/state replay audit**. It will compare
the loader-normalized observations and actions against the evaluator contract
on representative frozen samples before any additional training is approved.
An inference-only adapter change and full retraining remain deferred because
neither has a demonstrated concrete defect to target. Fold 02 remains locked.

Evidence:
[`Phase F next-step options`](results/phaseF_next_step_options_after_gripper_weighted_v1_20260922.json).

### Phase F trainer-side replay audit (2026-09-22)

The read-only trainer-side audit covered all 160 frozen training episodes
(five episodes for each of the 32 feasible tasks) using the 90k weighted
checkpoint's preprocessor. Every representative first frame loaded without
error: state shape was 15, action shape was 7, task text matched
`tasks.parquet`, values were finite, and both images were CHW 128x128 floats
in `[0,1]`. No loader schema or normalization mismatch was found. Therefore
another single-variable repair is not justified by the current evidence;
objective/data redesign requires a new explicit experiment design before any
further training. Fold 02 remains locked.

Evidence:
[`Phase F replay audit summary`](results/phaseF_trainer_side_task0_replay_audit_v1_summary_20260922.json),
[`Phase F replay audit decision`](results/phaseF_trainer_side_task0_replay_audit_decision_20260922.json).

### Phase M terminal-window weighting benchmark (2026-09-29)

The completed terminal-window weighted training run was evaluated on the
approved benchmark-v1 task-0 protocol at checkpoints 30k, 60k, and 90k,
using five fixed initializations per checkpoint (15 rollouts total). All
15 rollouts reached the 300-step limit with zero reward and no evaluator
errors (`0/15` success at every checkpoint). This diagnostic intervention
therefore provides no evidence that terminal-window weighting improves the
joint32 closed-loop objective; Fold 02 remains locked and no further sweep
of this weighting variant is recommended.

Evidence:
[`Phase M terminal-window benchmark summary`](results/phaseM_terminal_window_weighted_benchmark_v1_5init_summary_20260929.json).

### Phase M2 objective/terminal contract audit (2026-09-29)

A read-only code audit compared the calibrated evaluator's success rule and
300-step terminal budget with the SmolVLA training objective. The evaluator
uses environment success or positive reward, while training optimizes the
flow-matching action loss over padded demonstration action chunks; there is
no native terminal-success label or temporal-window configuration. This is a
known objective limitation, not a demonstrated evaluator or preprocessing
defect. Since the terminal-window intervention was negative, no new training
was started from this audit and Fold 02 remains locked.

Evidence:
[`Phase M2 objective/terminal contract audit`](results/phaseM_objective_terminal_contract_audit_20260929.json).

### Phase M multitask benchmark (2026-09-29)

To check whether the task-0 failure was isolated, the same calibrated V3
evaluator was run on tasks 0--4, init 0, at checkpoints 30k/60k/90k (15
independent rollouts). The result was `0/15` success: every task and every
checkpoint reached the 300-step budget with zero reward and no evaluator
error. This broadens the negative evidence beyond task 0, but remains a
diagnostic benchmark rather than a Fold 02 gate change.

Evidence:
[`Phase M multitask benchmark summary`](results/phaseM_terminal_window_weighted_multitask_benchmark_v1_summary_20260929.json).

### Phase M recovered historical positive control (2026-09-29)

The true recovered historical V3 asset (distinct from the same-named
training-prep checkpoint) was rerun under the calibrated protocol on task 0,
benchmark init 0, with `wait_steps=10` and `n_action_steps=25`. It succeeded
with reward `1.0` in 86 steps and no evaluator error. This restores the
positive control and confirms that the calibrated evaluator can produce a
known-good success; the joint32 multi-task `0/15` result therefore remains a
model-side negative result rather than an execution-only failure.

Evidence:
[`Recovered historical positive control`](results/phaseM_recovered_historical_control_task0_init0_20260929.json).

### Phase M checkpoint provenance audit (2026-09-29)

The recovered historical V3 positive-control checkpoint and the same-named
`loco_fold01_formal_v1/010000` training-prep checkpoint were compared by file
manifest and SHA256. Their model weights, configs, and normalization tensors
are different artifacts. The recovered path is therefore the only valid
positive-control reference; the training-prep path must not be substituted
for it in protocol comparisons.

Evidence:
[`Phase M checkpoint provenance audit`](results/phaseM_checkpoint_provenance_audit_20260929.json).

### Phase N next-intervention decision (2026-09-29)

With the recovered positive control restored and joint32 failing across five
tasks, three paths were compared. An unsupported inference patch is not
recommended because no concrete evaluator/state defect remains. The preferred
path is one bounded training-objective experiment, but only after an offline
release/terminal proxy and a no-GPU training smoke check are validated. If
that proxy cannot distinguish known-good terminal behavior, stop new training
and report the negative evidence instead. Fold 02 remains locked.

Evidence:
[`Phase N intervention decision`](results/phaseN_joint32_next_intervention_decision_20260929.json).

### Phase N offline terminal proxy preflight (2026-09-29)

The frozen joint32 dataset was analyzed without GPU training: 160 episodes
and 101,660 frames were scanned, using the final 20% of each episode as a
terminal window. A terminal proxy is computable, but it is not yet a
validated success discriminator: 155/160 episodes contain terminal gripper
sign changes and the mean terminal gripper standard deviation is `0.5591`.
This confirms that a release-related signal exists in the data, but does not
justify weighting it without success-labelled or positive-control validation.
The training gate therefore remains closed and Fold 02 remains locked.

Evidence:
[`Phase N offline terminal proxy`](results/phaseN_offline_terminal_proxy_20260929.json).

### Phase N action-proxy comparison (2026-09-29)

Existing action CSVs were compared without new rollouts: one recovered
positive-control success versus 15 joint32 failures. In the final 20% window,
the positive control had gripper mean `0.999`, std `0.0045`, and zero sign
changes; joint32 failures averaged gripper mean `0.442`, std `0.2397`, with
7/15 episodes showing sign changes. This is directionally useful, but the
positive group has only one rollout and therefore is not sufficient to
authorize loss weighting or a new training run. A larger positive-control
sample is required before treating this proxy as causal evidence.

Evidence:
[`Phase N action-proxy comparison`](results/phaseN_action_proxy_comparison_20260929.json).

### Phase N expanded positive-control proxy check (2026-09-29)

The positive-control sample was expanded from one to five recovered V3
rollouts and compared with the 15 joint32 failures. The difference is less
clean than the original n=1 snapshot: positive controls averaged gripper
mean `0.6741` and std `0.3553`, while joint32 failures averaged `0.4417` and
std `0.2397`; sign changes occurred in 2/5 versus 7/15. Position-motion
statistics also differ because the groups have different task trajectories.
This expansion therefore does **not** validate a causal terminal proxy or
authorize loss weighting. The proposed training gate remains closed.

Evidence:
[`Phase N expanded positive-control proxy check`](results/phaseN_action_proxy_5init_comparison_20260929.json).

### Phase N ten-positive-control proxy check (2026-09-29)

The recovered V3 positive-control sample was expanded to ten task-0
initializations and compared with the existing 15 joint32 failures. The
terminal gripper means were `0.5951` versus `0.4417`, and sign changes
occurred in 4/10 versus 7/15. The direction remains weak and overlapping;
the large position-motion difference is confounded by different trajectories.
This is still not a validated causal proxy, so the training gate remains
closed and no loss weighting is authorized.

Evidence:
[`Phase N ten-positive-control proxy check`](results/phaseN_action_proxy_10positive_comparison_20260929.json).

### Phase N training gate closure (2026-09-29)

After the calibrated positive-control sample was expanded to ten rollouts,
the terminal proxy still did not provide a stable, trajectory-matched
success discriminator. The current evidence therefore does not authorize a
new objective weighting experiment or an inference-only patch. The project
keeps the negative joint32 result as the current finding; a future repair
requires either a concrete implementation defect or a validated offline
proxy on matched positive/negative trajectories. Fold 02 remains locked.

Evidence:
[`Phase N training gate closure`](results/phaseN_training_gate_closure_20260929.json).

### Joint32 provenance/contract audit (2026-09-22)

The read-only audit confirms that `libero36_feasible_32_frozen_v1` uses a compact
`task_index` 0--31 mapping ordered by the original feasible LIBERO task IDs,
including task 0, and that its language strings match the evaluator task
instructions. The dataset contracts are `observation.state=[15]` and
`action=[7]`, matching the calibrated checkpoint metadata. No direct task-index
or state/action schema mismatch was found to explain the calibrated joint32
result (0/15 on task 0). The next recommended gate is image preprocessing and
normalization plus a small replay-contract audit; Fold 02 remains locked.

Evidence: `results/phaseD_joint32_provenance_contract_audit_20260922.json`.
### Joint32 image preprocessing contract audit (2026-09-22)

The read-only comparison found no direct image-contract mismatch: both dataset
images and the recovered evaluator use 128×128 RGB frames, the evaluator
converts HWC uint8 frames to CHW float tensors in [0,1], and the saved
checkpoint preprocessor declares visual normalization as `IDENTITY` (with state
and action using `MEAN_STD`). The remaining uncertainty is only the exact
trainer-side application of `use_imagenet_stats`; the next gate is a small
loader-to-evaluator frame replay audit. Fold 02 remains locked.

Evidence: `results/phaseD_joint32_image_contract_audit_20260922.json`.
### Joint32 image replay contract check (2026-09-22)

Five real JPEG frames from the frozen LeRobot shard were decoded successfully
for both camera streams. They are RGB `uint8`, 128×128, and satisfy the
evaluator's exact HWC→CHW float32/[0,1] conversion contract. No hidden image
encoding or dimensional mismatch was found. This is a representation check,
not a pixel-identical scene replay; the next decision gate is checkpoint/training
provenance rather than another image-format experiment.

Evidence: `results/phaseD_joint32_image_replay_contract_audit_20260922.json`.
### Joint32 task-0 action replay audit (2026-09-22)

Using the exact saved checkpoint preprocessor/postprocessor, eight real task-0
training frames were evaluated open-loop at checkpoint 090000. The mean action
MAE was `0.00813` (maximum `0.01358`). This weakens a global normalization or
action-scale mismatch as the explanation for the calibrated 0/15 closed-loop
result, while not claiming long-horizon success. The next intervention should
therefore be isolated on the training/data temporal objective, with a new
checkpoint and the same strict evaluator; Fold 02 remains locked.

Evidence: `results/phaseD_joint32_task0_action_replay_audit_20260922.json`.

### Phase N matched existing action-capture diagnostic (2026-09-29)

Existing V3 action CSVs were compared for five recovered historical positive
controls and five joint32 terminal-window-weighted failures at overlapping
benchmark initializations. The positive group had mean terminal gripper value
`0.7168` and five total sign changes; the joint32 group had mean `0.9097` and
zero sign changes. Because the checkpoint and model families differ, this is
diagnostic evidence only and is not a causal intervention result. No new
rollouts or training were started, and Fold 02 remains locked.

Evidence:
[`Phase N matched existing action-capture diagnostic`](results/phaseN_matched_existing_action_capture_v1_summary_20260929.json).

### Phase N matched state/action capture (2026-09-29)

Using the calibrated V3 evaluator with an opt-in diagnostic state capture,
five overlapping benchmark initializations were run for the recovered
historical checkpoint and the joint32 terminal-window-weighted checkpoint.
The recovered control produced `3/5` successes; joint32 produced `0/5`.
Every CSV contains 15-D state and 7-D action rows. This is a diagnostic
comparison only: the checkpoints are different model families, so it does not
authorize a causal repair or new training objective. The initial positive
control mismatch on two inits is retained as part of the measured variation;
the historical known-good result remains the separate calibrated control.
Fold 02 remains locked.

Evidence:
[`Phase N matched state/action capture`](results/phaseN_matched_state_action_capture_v1_summary_20260929.json).

The follow-up replay audit found partial, not exact, positive-control
replayability: init3/init4 agree with the prior action-proxy records, while
init0/init1 swap success and failure. This confirms strong initialization
sensitivity and means the diagnostic is descriptive rather than a deterministic
causal comparison. Joint32 remains `0/5` in the matched run; Fold 02 remains
locked.

Evidence:
[`Phase N positive-control replay consistency`](results/phaseN_positive_control_replay_consistency_20260929.json).

### Phase N final joint32 decision (2026-09-29)

The joint32 repair-training gate is now formally closed. The project accepts
the calibrated negative result as the current stage conclusion and will not
expand GPU experiments without a new falsifiable repair hypothesis. No
inference patch, checkpoint overwrite, result deletion, or Fold 02 unlock is
authorized. Reopening requires either an independently reproduced concrete
implementation defect or a trajectory-matched offline proxy with a
prespecified threshold.

Evidence:
[`Phase N final joint32 negative conclusion`](results/phaseN_joint32_final_negative_conclusion_20260929.json).
