# Teacher-control goal and unattended plan (2026-10-02)

## Objective and interpretation

Distinguish current training/data-pipeline failure, insufficient per-task training,
and multitask interference. Do not promise positive results or treat loss/offline
proxy metrics as closed-loop success. Native LIBERO spatial task0 is a diagnostic
control, not a replacement for the protected original 36-task benchmark.

## Verified starting point

- Historical native task0 checkpoint: recovered historical V3 010000.
- Fixed task/init3/CLI seed12345/effective seed12348, execution horizon25:
  five repetitions succeeded (reward1); steps93/89/174/90/89.
  Trajectories differ; this establishes outcome repeatability only.
- Official HDF5 from yifengzhu-hf/LIBERO-datasets revision
  97773100c1474cd0d686ebd173cc0e4fd5442466:
  SHA256 ff6f26121653c77280eb40a38773a74141c11a8509f3466058cb56dd2cc60ead,
  508779600 bytes, 50 episodes, 5068 frames. All schema/finite/image checks passed.
- Source recovered under /root/smolvla-training-prep/recovery/native_task0_20261002.
  This is official-source reconstruction, not verified byte identity to the old
  converted dataset. Old GPU was deleted; never retry it.
- Custom joint32 task0 (left-region bowl) is NOT native spatial task0.
- No train/eval/unattended process was found in the current targeted process check.

## Stage goals and decision gates

| Stage | Small goal | Completion evidence / next action |
|---|---|---|
| A | Recover matched positive-control task | Completed HDF5 audit; preserve original and audit |
| B | Rebuild current-format dataset | Separate new root; 50 episodes/5068 frames; every action/state compared to source float32; image orientation/values and task text checked; loader/preprocessor smoke test |
| C | Design and run current-pipeline single-task control | Save exact config and comparison to historical/current joint32 configs; verify pretrained model and initialization; original official training entry plus episode compatibility wrapper, no new weighted loss or sampler; one bounded 10000-step pilot, separate output, checkpoints2500/5000/10000 |
| D | Verify single-task closed-loop capability | Same native evaluator and fixed positive-control condition; 5 repeats for predeclared final10000 checkpoint, then existing 20 benchmark initializations with matched historical reference. Reward/success and failures reported separately from determinism compatibility. Select/checkpoint comparisons explicitly diagnostic, not untouched test evidence |
| E | Fair task-count scaling design | Only after D: same task family/protocol, nested4/8/16/32 sets; fixed actual per-task sampling and comparable per-task learning-rate exposure; saved manifests and task budgets before launch; do NOT reuse constant total steps or pool custom/native task identities |
| F | Research conclusion and publication | Compare per-task closed-loop results, uncertainty, training exposure and failure modes; persist evidence and publish authorized summaries/scripts/README; Fold02 only after existing gate audit and authorized transition |

Stage C is a capability pilot, not an exposure-matched causal comparison against
joint32 90000 steps. Failure cannot by itself prove a pipeline bug. Initialization,
optimization schedule, preprocessing, dataset provenance and actual sample exposure
must be accounted for before attribution.

Observed runtime detail for this pilot: LeRobot scales the configured cosine
schedule when total steps are shorter than decay steps. The 10000-step run
therefore uses 333 warmup steps and 10000 decay steps (from configured 3000 and
90000). Treat the run as a capability test under the standard entry point's
runtime behavior; do not report the schedule as unchanged.

Stage D provisional progression gate: fixed condition 5/5 outcomes and full-init
performance not materially below matched historical reference (predeclare a 10
percentage-point practical margin before looking at new outcomes). Small samples
do not prove statistical equivalence. Before expensive scaling, independently
repeat a training seed and require the same practical gate. Freeze assessment
criteria before evaluation; do not lower them after failure.

## Compare alternatives before each mutation

Write an independent decision JSON: evidence, >=2 viable options, expected benefit,
cost, confounders, falsifying outcome, chosen recommendation and stop conditions.
Current comparison: (1) rebuild matched official native dataset [recommended];
(2) use custom left-region task with matching evaluator [different question,
needs matched positive reference]; (3) custom/native cross-task test [inadmissible
as a pipeline control]. Next compare current-pipeline capability pilot against
historical recipe reproduction: choose the former for the teacher's question,
reserve historical recipe reproduction as a bounded follow-up if failure is
unresolved. Never alter multiple variables and claim one-variable causality.

## Unattended execution contract

- Continue in this chat via one scheduled heartbeat every30 minutes; read this
  plan and live artifacts each run. Do not duplicate existing jobs/monitors.
- One GPU workload at a time. Check PID/start time/command, active jobs, GPU
  memory and free disk; use a lock and independent durable logs/output roots.
- Automatically execute recommended in-scope restoration, conversion, audits,
  single-task pilot and diagnostic evaluation after their preflight passes.
- Do not auto-expand costly scaling before the dataset/task-family/exposure plan
  exists and single-task gates pass. No indefinite training or parameter sweep.
- On failure save exit code/log/config and diagnosis. Compare fixes; at most one
  evidence-backed repair retry for a stage. Repeated same blocker requests user
  direction; do not silently spin or claim unattended progress.
- No deleting files, replacing old checkpoints/results, GPU restart/shutdown,
  changing Git branch, force push, protected benchmark modification, credential
  disclosure, or new paid service. Preserve user edits and untracked files.
- Preserve state/action index mapping: action_log[i] causes per_step_log[i+1].
  open=-1 / close=+1. Record full checkpoint+model group+task+init+seed+protocol.
- Determinism flags/no errors do not prove trajectory identity. Fixed-condition
  outcome repeatability and independent-init performance are separate gates.
- Sync authorized lightweight scripts/configs/JSON summaries and README to the
  verified current GitHub branch and existing private HF repo; verify remote
  revisions after upload. Never upload keys/tokens, large raw datasets/checkpoints
  or videos without separate explicit scope confirmation. Auth failure queues
  sync; no plaintext-password workaround. HF workflow uses hf-cli skill.
- Remain quiet if nothing actionable changes; notify stage completion, meaningful
  result, failure or required input. Stop scheduling after goal completion or
  explicit user cancellation. Do not archive this research chat automatically.

## Current next step

Rebuild and fully verify native task0 using the existing converter without
--overwrite, then produce stage-C options/config/preflight before GPU training.
Goal created, heartbeat activation is tracked separately from remote job status.
