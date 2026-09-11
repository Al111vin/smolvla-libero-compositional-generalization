# LIBERO-36 pilot data acquisition

The pilot artifacts are collected by exact replay of the formal Gate 5
trajectory, after a deterministic passive settle, with five distinct reset
seeds per task. Each frame stores the 128×128 `agentview` and
`eye_in_hand` images, robot state, and the replay action.

Only lightweight manifests and QC reports are tracked in GitHub. HDF5 and
LeRobot directories are stored in the Hugging Face dataset repository:

```bash
hf download Alllvinnn/smolvla-libero-compositional-generalization \
  --type dataset --revision c237443e --local-dir datasets/hf_snapshot
```

The authoritative collection order and per-task artifact paths are in
`data/demos/libero36_expansion_plan_v1.json`. The unified training manifest
is `data/training/libero36_feasible_v1/manifest.json`. Tasks 15, 17, 33, and
35 remain excluded as physically infeasible under the strict Gate 5 protocol.
