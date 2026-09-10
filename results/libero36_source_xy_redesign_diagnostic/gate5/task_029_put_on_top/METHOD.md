# Task 29 Gate 5 method

- Object: `cream_cheese_1`
- Skill: `put_on_top`
- Destination region: `right`
- Layout: 1
- End-effector yaw: -45 degrees
- Grasp-height offset: 0.000 m
- Vertical lift: 0.110 m
- Carry height above plate: 0.150 m
- Carry action limit: 240 steps with early convergence
- Recorded actions: 675
- Robot–distractor contact steps: 0
- Robot–plate contact steps: 0
- Unexpected-object contact steps: 0
- Exact replay state maximum absolute difference: 0.0
- Recording passed: true
- Exact replay passed: true

The -45-degree orientation preserves the cream-cheese grasp while
rotating the Panda hand away from `akita_black_bowl_1`. Task 29 uses
the extended carry convergence allowance; Tasks 27 and 28 converge
before the original limit.
