#!/bin/bash
set -u
BASE=/root/smolvla-training-prep/results/training/loco_fold01_formal_v1/checkpoints/010000/pretrained_model
OUT=/root/smolvla-eval-prep/results/benchmark_v1_historical_control_multitask_20260929
EVAL=/root/smolvla-libero-compositional-generalization/scripts/eval_v3_task0.py
PY=/usr/local/miniconda3/envs/py312/bin/python
mkdir -p "$OUT"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=/root/smolvla-libero-compositional-generalization/scripts
for task in 0 1 2 3 4; do
  d="$OUT/task_${task}_init_0"
  mkdir -p "$d"
  "$PY" "$EVAL" --checkpoint "$BASE" --task-id "$task" --init-source benchmark --init-index 0 --wait-steps 10 --n-action-steps 25 --max-steps 300 --seed $((12345+task)) --device cuda --results-dir "$d" > "$d/run.log" 2>&1 || echo "FAILED task=$task" >> "$OUT/errors.log"
done
find "$OUT" -name '*_summary.csv' | sort > "$OUT/manifest.txt"
echo "HISTORICAL_CONTROL_MULTITASK_COMPLETE records=$(wc -l < "$OUT/manifest.txt")"
