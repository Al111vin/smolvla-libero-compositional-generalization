# LIBERO Spatial LOCO LeRobot Datasets

Two independent training datasets are materialized for the primary
target-role LOCO experiments.

| Held-out task | Training task IDs | Episodes | Frames | Stats SHA256 |
|---:|---|---:|---:|---|
| 6 | 0, 2, 3, 4, 7, 8, 9 | 350 | 43,435 | `818faa38d36fb4a19660c825e4a24564307d1e710a8f02f862e5f4cb71b34793` |
| 3 | 0, 2, 4, 6, 7, 8, 9 | 350 | 44,695 | `db75a60d8740a9a99952429696affd7950a1d50d7a9eb832f71b496773fd94be` |

Each training task contributes 50 demonstration episodes. Tasks 1 and
5 are excluded from both primary experiments according to the audited
LOCO protocol.

The folds are stored as separate physical LeRobot datasets because
LeRobot 0.5.1 episode filtering selects training rows but does not
recompute `dataset.meta.stats`. Using a shared mother dataset would
therefore leak held-out state and action normalization statistics.

The large LeRobot dataset directories are ignored by Git. The tracked
episode manifests preserve the mapping between global LeRobot episode
indices, LIBERO task IDs, source demonstrations, language instructions,
and source HDF5 files.

Rebuild the datasets with:

- `scripts/convert_libero_split_to_lerobot.py`
- `data/splits/libero_spatial_loco_task6_train.csv`
- `data/splits/libero_spatial_loco_task3_train.csv`
