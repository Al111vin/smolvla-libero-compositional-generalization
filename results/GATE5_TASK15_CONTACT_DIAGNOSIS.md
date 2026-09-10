# Task 15 contact diagnosis, 2026-09-05

Status: incomplete. No new Gate 5 pass is claimed. Failure of the tested
controllers is not evidence of physical infeasibility.

The read-only state inspector `scripts/inspect_gate5_contact_state.py`
restored states from `/tmp/libero36_gate5_task15_safe_hand_v8/trajectory.npz`
and inspected simulator contacts after `forward()`, without stepping physics.
This is a geometric inspection, not a new deterministic replay certification.

At the first push action (272), six hand contacts touch the mug rim near
world z=1.003 m. Their target-to-hand normal z components are 0.9978--0.9999.
At step 292 the hand contact normals still have z components above 0.9988.
At step 921 both remaining hand contacts have z components above 0.9992.
Thus the purported hand push primarily presses on the rim from above.
One initial finger-body contact has a mostly lateral normal but does not
persist into the later inspected states.

This evidence does not establish insufficient friction, an unreachable
workspace, or an impossible task. Increasing lateral command magnitude
without establishing a lateral contact normal cannot distinguish these
hypotheses. The previous controllers did not check contact direction.

Next controller prerequisite: establish contact with an exterior mug wall,
verify its horizontal normal aligns with the intended push, and inspect
the full gripper clearance before beginning a push segment. Reject rim-only
contact as an unsuitable controller entry (not a changed Gate 5 threshold).
Keep the frozen scene and the user's strict acceptance requirements intact.
Do not promote the permissive v6 interpretation as equivalent to existing
strict evidence without reconciling the acceptance definitions.

Remote inspection: `/tmp/gate5_contact_inspection_v8/contacts.json` and
`step_272.png`, `step_292.png`, `step_921.png`.
Local copies: `/tmp/gate5_contact_inspection_v8/`.

## Exterior-wall entry tests

The new diagnostic helper `scripts/gate5_contact_direction.py` orients each
contact normal from robot toward target, independent of simulator geometry
ordering. A controller-entry candidate requires abs(normal.z) <= 0.35 and
normal.xy dot push_direction >= 0.8. These are entry-selection heuristics,
not changes to the formal Gate 5 acceptance thresholds. Tests confirmed
geometry-order invariance and rejection of rim and opposite-facing contacts.
The probe checks all 20 hold steps and rejects any historical grasp,
nonselected target contact, robot distractor contact, or object collision.

Task 15, left pad, yaw -90 degrees, pad z=0.9601 m, pad x=0.1301 m:

- `/tmp/gate5_task15_external_wall_055.json`: 223 actions; high transit
  residual 0.02376 m. No contact and zero target drift.
- `/tmp/gate5_task15_external_wall_055_v2.json`: 323 actions; a 0.030 m
  transit waypoint tolerance allows the next vertical leg. Its residual
  remains 0.02341 m; final pad is (0.10988, 0.06860, 1.04833).
  No contact and zero target drift. Lateral entry was not reached.

These results locate an approach-controller convergence issue before
contact. They do not establish that the desired low contact pose is
unreachable. Next inspect joint limits and pose/orientation residuals or
solve a collision-checked approach path to that pose; do not infer friction
or task infeasibility from these pre-contact failures.

## Joint and low-approach diagnostics

`/tmp/gate5_task15_external_wall_joint_diagnostic.json` reproduces the
pre-contact stop: no robot contacts, minimum joint-limit margin 0.4839 rad,
orientation error 0.04810 rad. The evidence excludes a hard joint-limit
stop at that state, not general reachability or controller conditioning.

`/tmp/gate5_task15_external_wall_low.json` allows the collision-free high
waypoint's 0.0234 m residual and then executes the low approach. It records
360 actions, exactly one selected-pad-contact step, 24 nonselected-contact
steps, 0/20 held lateral-pad-contact steps, maximum target drift 0.000436 m,
and maximum target lift 0.001820 m. The controller entry is rejected.
Minimum joint-limit margin at the endpoint is 0.4391 rad; orientation error
is 0.01049 rad. The final contacts are hand-to-mug g7/g8.

This demonstrates a transient low pad contact is physically reached by
the approach, but does not demonstrate an acceptable sustained side push.
Next work should examine the gripper geometry at the transient contact and
plan clearance of the full hand; the strict selected-contact history and
hold requirements must remain intact.

## Transient contact direction

The trace `/tmp/gate5_wall_trace/left_yaw-90.0_height0.06.npz` now retains
all diagnostic states; inspection files are under `/tmp/gate5_wall_contact_frames`.
At action 339, the left pad position is (0.109568, 0.067565, 0.971559),
short of the intended exterior x=0.130111. Its contact with mug g7 has
robot-to-target normal (0.845468, -0.476772, -0.240566). This points mainly
toward +x, opposite the intended -x push. The hand simultaneously contacts
the rim. On action 340 (first zero-action hold step), pad contact disappears.

Consequently this was not a valid side-push entry even at its transient
contact frame. The probe now rejects a wrong-facing selected-pad normal
immediately instead of treating any selected-pad contact as entry success.
The diagnostic trace has no replay certification and is not formal evidence.
