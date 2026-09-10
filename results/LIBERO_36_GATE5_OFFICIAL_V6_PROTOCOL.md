# LIBERO-36 Gate 5 official replay protocol v6

This protocol is the executable acceptance interpretation of Gate 5 in
`LIBERO_36_PROXY_TASK_PROTOCOL.md`.  It supersedes neither the benchmark
layout nor the Gate 1--4 records.  In particular, it does not modify, replace,
or count the pre-existing v5 pilot summaries.

## Scope

For each of the 36 logical tasks, preserve the frozen layout, BDDL, reset
seed, native environment horizon, and Gate 3 / Gate 4 prerequisites.  Record
one robot trajectory and a fresh-reset replay.  The original and replay are
both retained as immutable NPZ evidence.

## Acceptance checks

For a push task, a trajectory passes only when all of the following hold.

1. The recorded state, action, reset-settle, contact, support, and goal traces
   are byte-exact on a fresh reset with the same seed.
2. The trajectory contains an actual `push` phase with target contact, but
   never triggers the grasp proxy.  Incidental contact during approach or
   terminal settling is not a push.
3. The target stays table-supported throughout and is lifted no more than
   0.030 m above its recorded stabilized reset height.
4. No unexpected object contact or robot--distractor contact occurs.
5. The exact goal relation, target-region XY test, and table support each hold
   for 20 consecutive terminal zero-action steps.  In the final five steps,
   target translation per step is at most 0.001 m.
6. The unmodified environment reaches no artificial controller time extension:
   its native horizon remains in force.

The 0.030 m lift ceiling is the already-recorded project definition of
“excessive lifting”; this v6 protocol freezes it rather than retuning it.

## Deliberately diagnostic fields

`selected_fingerpad_only_target_contact`, `opposite_pad_never_contacts`, and
the v5 `MAX_ACTIVE_ACTIONS = 930` controller horizon are retained in evidence
for diagnosis, but are not Gate 5 acceptance checks.  Gate 5 requires a push
without grasping or excessive lifting; it does not prescribe a single
fingerpad or a 930-action cap.  Allowing a non-grasping auxiliary robot contact
does not allow target lifting, grasping, distractor contact, object collision,
or a changed simulator horizon.

## Evidence separation

All v6 results live under `results/libero36_gate5_official_v6/`.  Existing v5
files are read-only historical evidence.  A task is promoted only from a v6
summary emitted by `scripts/verify_gate5_official_v6_push_attempt.py` and its
matching fresh-reset replay artifact.
