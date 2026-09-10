# Gate 5 task 005: put_inside

Protocol: `libero_36_proxy_tabletop_draft_v5`

- Logical task: 5
- Layout: 1
- Spatial region: `right`
- Target: `akita_black_bowl_1`
- Receiver: `basket_1`
- Actions: 605
- Wrist yaw: -90 degrees
- Rim X offset: +0.044 m
- Close steps: 45
- Seat-grasp lift: +0.020 m
- Basket release height: basket Z +0.140 m
- Exact replay: passed
- Maximum replay state difference: 0.0
- Unexpected object contact steps: 0
- Robot-distractor contact steps: 0
- Robot-basket contact steps: 0
- Terminal `In` relation hold: passed
- Receiver target-region preservation: passed

The bowl is released above the basket because the gripper cannot
safely enter the narrow basket opening while maintaining the rim
pinch. The resulting free drop settles into the validated contain
region and remains stable through the terminal hold.
