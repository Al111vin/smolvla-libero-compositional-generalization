# Task 006 Gate-5 trajectory

This directory contains the isolated successful layout-1 trajectory for
task 6 (`akita_black_bowl push_to left`) under
`libero_36_proxy_tabletop_draft_v5`.

Controller configuration:

- reset index: 0
- yaw: -71 degrees
- selected finger: right
- controller radius: 0.085 m
- approach standoff: 0.100 m
- pad height offset: 0.04550 m
- contact-search XYZ action scale: 0.60
- active gripper action: -1.0 (open)

The contact-search translation scale applies only during
`contact_search`. The subsequent push, retreat, and terminal hold use
the frozen formal calibrator behavior.

The isolated attempt passed all per-trajectory checks: selected-pad
contact was acquired, the opposite pad never contacted the target,
there was no nonselected robot-target contact, no robot-distractor or
unexpected object contact, table support held throughout all active
steps, and the terminal relation and XY predicates remained stable.

The formal calibrator replayed the saved actions deterministically.
The maximum absolute trajectory-state replay difference was exactly
zero. `trajectory.npz` retains both the original and embedded replay
arrays; `replay.npz` contains the extracted replay arrays.
