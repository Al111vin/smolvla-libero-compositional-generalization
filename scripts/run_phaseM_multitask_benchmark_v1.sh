#!/bin/bash
set -u
BASE=/root/smolvla-training-prep/results/training/libero36_joint32_terminal_window_weighted_v1/checkpoints
OUT=/root/smolvla-eval-prep/results/benchmark_v1_terminal_window_weighted_multitask_20260929
EVAL=/root/smolvla-libero-compositional-generalization/scripts/eval_v3_task0.py
PY=/usr/local/miniconda3/envs/py312/bin/python
mkdir -p "$OUT"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=/root/smolvla-libero-compositional-generalization/scripts
for ck in 030000 060000 090000; do
  for task in 0 1 2 3 4; do
    d="$OUT/checkpoint_${ck}/task_${task}_init_0"
    mkdir -p "$d"
    "$PY" "$EVAL" --checkpoint "$BASE/$ck/pretrained_model" --task-id "$task" --init-source benchmark --init-index 0 --wait-steps 10 --n-action-steps 25 --max-steps 300 --seed $((12345+task)) --device cuda --results-dir "$d" > "$d/run.log" 2>&1 || echo "FAILED ck=$ck task=$task" >> "$OUT/errors.log"
  done
done
find "$OUT" -name '*_summary.csv' | sort > "$OUT/manifest.txt"
echo "MULTITASK_BENCHMARK_COMPLETE records=$(wc -l < "$OUT/manifest.txt")"
