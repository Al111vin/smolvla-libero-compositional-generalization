#!/usr/bin/env bash
set -euo pipefail

EVAL=/root/smolvla-eval-prep/scripts/eval_v3_task0.py
CKPT_ROOT=/root/smolvla-training-prep/results/training/libero36_joint32_formal_v1/checkpoints
OUT=/root/smolvla-eval-prep/results/audits/phaseC_joint32_full60_20260922
mkdir -p "$OUT"
for ckpt in 030000 060000 090000; do
  for i in $(seq 0 19); do
    seed=$((12345+i))
    d="$OUT/checkpoint_$ckpt/init_$i"
    mkdir -p "$d"
    MUJOCO_GL=egl /usr/local/miniconda3/envs/py312/bin/python "$EVAL" \
      --finetuned-checkpoint "$CKPT_ROOT/$ckpt/pretrained_model" \
      --suite benchmark --task-id 0 --init-index "$i" --seed "$seed" \
      --wait-steps 10 --n-action-steps 25 --max-steps 300 --output-dir "$d" \
      --output-prefix "joint32_${ckpt}_init${i}_n25_seed${seed}"
  done
done
echo "PHASE_C_JOINT32_FULL60_COMPLETE"
