V4 LIBERO Spatial LOCO Post-hoc Diagnostic Protocol

This document freezes a diagnostic-only analysis of the already
completed V4 LOCO formal evaluation. It does not define a new success
rate experiment and must not replace the 200-rollout formal result.

## Frozen primary result

The primary result remains the official benchmark evaluation produced
under `v4_loco_official_v1`:

- `task3_holdout`: task 3 held out, task 6 seen control
- `task6_holdout`: task 6 held out, task 3 seen control
- 50 benchmark initial states per model-task cell
- 280 maximum policy actions
- 10 stabilization actions
- `n_action_steps=25`
- environment seed `1000 + init_index`
- policy seed `2000 + init_index`
- success criterion `env.check_success()`

The tracked formal rollout CSV and manifest, plus their saved NPZ
action traces, are the immutable inputs to this diagnostic.

## Frozen case selection

The selection file is:

`data/diagnostics/v4_loco_case_selection.csv`

It contains 20 task-state cases:

- tasks 3 and 6
- five `seen_success_heldout_failure` cases per task
- five `both_failure` cases per task

Each selected task-state case is replayed for both models, giving 40
diagnostic replays. The evaluator verifies every stratum label against
the frozen formal outcomes before starting.

Full `diagnostic` mode is locked to both models, the tracked selection,
the tracked formal rollout CSV and manifest, and
`results/v4_loco_diagnostics/`. A one-model or alternate-path run is
allowed only in `smoke` mode and is never a completed diagnostic set.

The selection is post-hoc and diagnostic-only. It is not a random
sample for estimating a population success rate.

## Exact action replay

The diagnostic evaluator does not load a checkpoint, run the policy,
or use the GPU. It replays the exact float32 `applied_actions` stored
in each formal NPZ trace.

Initialization exactly follows the formal protocol:

1. seed Python and NumPy with `1000 + init_index`;
2. seed the environment when supported;
3. reset the environment;
4. set the official LIBERO benchmark initial state as float64;
5. apply ten `[0, 0, 0, 0, 0, 0, -1]` stabilization actions;
6. reseed Python and NumPy with `2000 + init_index`;
7. execute the saved formal actions without postprocessing or
   clipping them again.

For an N-action formal trace:

- the diagnostic step table has N rows;
- row k contains action k and the state after action k;
- the video has N+1 frames;
- frame 0 is the post-stabilization state;
- frame k is the state after action k.

The stabilization period is not included in the diagnostic video.

## Mandatory replay parity

A replay is valid only if all of the following match the frozen formal
trace:

- number of executed actions;
- every reward, with absolute tolerance `1e-7`;
- every `done` flag;
- every `env.check_success()` value;
- final success and formal step count.

Any mismatch raises an error. The evaluator must not append a summary
row or mark that replay complete.

The evaluator also verifies:

- the formal rollout CSV hash recorded by the formal manifest;
- the formal protocol document hash;
- formal evaluator commit and hash;
- checkpoint metadata for each model;
- both 100-file formal action-trace bundle hashes;
- the selected-case strata against formal outcomes.

## Recorded observations and proxies

The per-action compressed CSV records:

- raw, processed, and applied formal actions;
- reward, `done`, and official-success parity;
- robot joints, EEF position, and gripper position/velocity;
- target-bowl, cookie-box, and plate positions;
- EEF-to-bowl distance;
- bowl-to-cookie and bowl-to-plate distances;
- target-bowl lift from the post-stabilization height;
- left and right fingerpad contacts with the target bowl;
- robosuite bilateral grasp-contact proxy;
- target-bowl contact with the cookie box and plate;
- the manual bowl-on-plate predicate.

The grasp-contact proxy is true only when both Panda fingerpad groups
contact target-bowl collision geoms at the sampled control-step
endpoint. It is not proof of a stable force-closure grasp.

For tasks 3 and 6, the evaluator requires
`plate_state.check_ontop(bowl_state)` to agree with
`env.check_success()` at every replayed action.

## Predefined diagnostic milestones

The summary records whether and when each replay first reaches:

1. approach: EEF-to-target-bowl distance at most 5 cm;
2. bilateral grasp-contact proxy;
3. lift: target bowl at least 3 cm above its post-stabilization height;
4. near plate: target-bowl to plate XY distance at most 8 cm;
5. target-bowl contact with the plate;
6. target bowl on the plate;
7. official task completion.

Approach and near-plate checks include the post-stabilization initial
state. A first-step value of 0 means the threshold already held before
the first replayed action.

These milestones are descriptive proxies. They do not establish model
intent, attention, language understanding, or a unique causal failure
mode.

## Video artifact

Frames are buffered during simulation and encoded only after replay
parity succeeds. Each MP4 is:

- 20 frames per second;
- 128 pixels high by 256 pixels wide;
- agentview on the left;
- wrist view on the right;
- RGB with no vertical flip or overlay;
- exactly N+1 frames.

Video and compressed step tables are written through temporary files
and atomically replaced. The summary CSV is rewritten atomically and
serves as the completion marker. Resume validation checks hashes,
schema, row counts, video frame count, resolution, frame rate, replay
parity, grasp-contact consistency, and goal-predicate consistency.

## Interpretation boundary

The formal success rates and paired tests remain the sole quantitative
generalization result. These selected replays may be used to describe
observed trajectories and milestone patterns only.

Because `n_action_steps=25`, actions inside a cached chunk are not all
fresh responses to the immediately preceding video frame. Diagnostic
language must therefore avoid frame-by-frame causal claims.

Raw diagnostic outputs are stored under
`results/v4_loco_diagnostics/` and are ignored by Git. Only a compact,
separately generated diagnostic summary should be considered for
tracking after all 40 exact replays pass validation.
