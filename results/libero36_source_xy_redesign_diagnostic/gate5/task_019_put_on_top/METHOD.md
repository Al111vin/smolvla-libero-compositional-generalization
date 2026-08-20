# Task 019 Gate-5 trajectory

This directory contains the exact, safely replayable layout-1
trajectory for task 19 (`alphabet_soup put_on_top middle`)
under `libero_36_proxy_tabletop_draft_v5`.

Controller configuration:

- layout: 1
- object: `alphabet_soup_1`
- receiver: `plate_1`
- destination region: `middle`
- grasp X offset: 0.000 m
- grasp Z offset: -0.015 m
- close steps: 45
- vertical lift: 0.110 m
- carry height above plate: 0.150 m

The recorded attempt passed all trajectory checks. The object was
grasped before transport, the goal relation was acquired and held
through the terminal actions, and the receiver remained in its
required region. There was no robot-distractor, robot-plate, or
unexpected object contact.

Formal open-loop replay passed. The initial-state and full-trajectory
maximum absolute state differences were exactly zero. All recorded
state, contact, grasp, relation, done, target-position, and
end-effector traces matched exactly. `trajectory.npz` and
`replay.npz` are byte-identical.
