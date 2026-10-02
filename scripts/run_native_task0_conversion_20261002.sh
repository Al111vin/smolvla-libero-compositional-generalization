#!/usr/bin/env bash
set -euo pipefail
BASE=/root/smolvla-training-prep
REC="$BASE/recovery/native_task0_20261002"
OUT="$BASE/datasets/lerobot/libero_spatial_task0_reconstructed_20261002"
PY=/usr/local/miniconda3/envs/py312/bin/python
SOURCE="$REC/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
exec 9>"$REC/conversion.lock"
flock -n 9 || { echo 'CONVERSION_ALREADY_RUNNING'; exit 75; }
test ! -e "$OUT" || { echo 'OUTPUT_EXISTS_REFUSE_OVERWRITE'; exit 73; }
"$PY" "$REC/audit_native_task0_recovery_20261002.py" "$SOURCE"
"$PY" "$BASE/scripts/convert_libero_hdf5_to_lerobot_v2.py" \
  --input "$SOURCE" --output "$OUT" \
  --repo-id local/libero_spatial_task0_reconstructed_20261002 \
  --fps 20 --expected-episodes 50 --expected-frames 5068
echo 'CONVERSION_FINISHED_PENDING_INDEPENDENT_QC'
