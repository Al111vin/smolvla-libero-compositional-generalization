#!/bin/bash
set -u
OUT=/root/smolvla-eval-prep/results/benchmark_v1_recovered_historical_control_task0_5init_20260929
BASE=/root/smolvla-eval-prep/recovered_historical_v3_20260922/outputs/smolvla_v3_task0_10k_run1/checkpoints/010000/pretrained_model
EVAL=/root/smolvla-libero-compositional-generalization/scripts/eval_v3_task0.py
PY=/usr/local/miniconda3/envs/py312/bin/python
mkdir -p "$OUT"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=/root/smolvla-libero-compositional-generalization/scripts
for i in 0 1 2 3 4; do
  d="$OUT/init_${i}"
  mkdir -p "$d"
  "$PY" "$EVAL" --checkpoint "$BASE" --task-id 0 --init-source benchmark --init-index "$i" --wait-steps 10 --n-action-steps 25 --max-steps 300 --seed $((12345+i)) --device cuda --results-dir "$d" > "$d/run.log" 2>&1 || echo "FAILED init=$i" >> "$OUT/errors.log"
done
find "$OUT" -name '*_summary.csv' | sort > "$OUT/manifest.txt"
echo "POSITIVE_CONTROL_COMPLETE records=$(wc -l < "$OUT/manifest.txt")"
