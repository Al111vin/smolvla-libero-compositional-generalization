#!/usr/bin/env bash
set -euo pipefail
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
PY=/usr/local/miniconda3/envs/py312/bin/python
EVAL=/root/smolvla-eval-prep/scripts/eval_v3_task0_state_capture_v1.py
CK=/root/smolvla-eval-prep/recovered_historical_v3_20260922/outputs/smolvla_v3_task0_10k_run1/checkpoints/010000/pretrained_model
OUT=/root/smolvla-eval-prep/results/teacher_control_repeats_20261002_v1
test ! -e "$OUT" || { echo 'Output already exists; refusing restart'; exit 2; }
mkdir -p "$OUT"
sha256sum "$EVAL" "$CK/model.safetensors" > "$OUT/input_sha256.txt"
for repeat in 1 2 3 4 5; do
  "$PY" "$EVAL" --checkpoint "$CK" --task-id 0 --init-source benchmark --init-index 3 --wait-steps 10 --n-action-steps 25 --max-steps 300 --seed 12345 --device cuda --capture-state --results-dir "$OUT/repeat_$repeat" > "$OUT/repeat_$repeat.log" 2>&1
done
echo COMPLETE
