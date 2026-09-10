# Task 034 Gate-5 push trajectory

This directory stores the isolated layout-1 trajectory for Task 34
(`cream_cheese push_to middle`).  The controller uses the left finger with a
10 mm pad-height offset, a +32 degree wrist yaw, 110 four-step push
increments, and a 5 mm retreat.

`trajectory.npz` contains the original action/state/contact trace and the
fresh-reset replay embedded by the controller.  `replay.npz` contains the
extracted replay arrays.  `replay_summary.json` is produced by
`scripts/verify_gate5_push_attempt.py`; it requires exact action and state
traces, selected-only contact, no grasp/distractor/unexpected contact, table
support, and a 20-step terminal relation/XY hold.

The controller records to `/tmp` on every reproduction.  Evidence is promoted
to this directory only after the independent verifier passes, so it cannot
replace an earlier accepted artifact during experimentation.
