#!/usr/bin/env bash
set -euo pipefail
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
PY=/usr/local/miniconda3/envs/py312/bin/python
OUT=/root/smolvla-eval-prep/results/phaseN_matched_state_action_capture_v1
mkdir -p "$OUT/positive" "$OUT/negative"
P=/root/smolvla-eval-prep/recovered_historical_v3_20260922/outputs/smolvla_v3_task0_10k_run1/checkpoints/010000/pretrained_model
N=/root/smolvla-training-prep/results/training/libero36_joint32_terminal_window_weighted_v1/checkpoints/090000/pretrained_model
for i in 0 1 2 3 4; do
  "$PY" /root/smolvla-eval-prep/scripts/eval_v3_task0_state_capture_v1.py --checkpoint "$P" --init-source benchmark --init-index "$i" --n-action-steps 25 --seed 12345 --results-dir "$OUT/positive/init_$i" --capture-state
  "$PY" /root/smolvla-eval-prep/scripts/eval_v3_task0_state_capture_v1.py --checkpoint "$N" --init-source benchmark --init-index "$i" --n-action-steps 25 --seed 12345 --results-dir "$OUT/negative/init_$i" --capture-state
done
