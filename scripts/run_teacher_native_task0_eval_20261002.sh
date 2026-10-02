#!/usr/bin/env bash
set -euo pipefail
BASE=/root/smolvla-training-prep
EVAL=/root/smolvla-eval-prep
REC="$BASE/recovery/native_task0_20261002"
CKPT="$BASE/results/training/teacher_native_task0_current_pipeline_10k_20261002_v1/checkpoints/010000/pretrained_model"
OUT="$EVAL/results/teacher_native_task0_current_pipeline_10k_eval_20261002_v1"
PY=/usr/local/miniconda3/envs/py312/bin/python
SCRIPT="$EVAL/scripts/eval_v3_task0_state_capture_v1.py"
TRAIN_LOG="$REC/training.log"
TRAIN_OK_MARKER='TEACHER_SINGLE_TASK_TRAIN_EXIT_CODE=0'
TRAIN_EXIT_MARKER='TEACHER_SINGLE_TASK_TRAIN_EXIT_CODE='

# Remain unattended across the currently approved single-task training run.
# Stop on an explicit non-zero train exit or after a bounded one-hour wait.
for attempt in $(seq 1 120); do
  if grep -q "$TRAIN_OK_MARKER" "$TRAIN_LOG"; then
    break
  fi
  if grep -q "$TRAIN_EXIT_MARKER" "$TRAIN_LOG"; then
    echo 'TRAINING_EXITED_NONZERO'; exit 74
  fi
  sleep 30
done
grep -q "$TRAIN_OK_MARKER" "$TRAIN_LOG" || { echo 'TRAINING_WAIT_TIMEOUT'; exit 74; }

exec 9>"$BASE/teacher_control_gpu.lock"
locked=0
for attempt in $(seq 1 24); do
  if flock -n 9; then locked=1; break; fi
  sleep 5
done
test "$locked" -eq 1 || { echo 'GPU_JOB_LOCKED_AFTER_TRAINING'; exit 75; }
for step in 002500 005000 007500 010000; do
  test -f "$BASE/results/training/teacher_native_task0_current_pipeline_10k_20261002_v1/checkpoints/$step/pretrained_model/model.safetensors"
done
test -f "$SCRIPT" || { echo 'EVALUATOR_MISSING'; exit 74; }
test ! -e "$OUT" || { echo 'OUTPUT_EXISTS_REFUSE_OVERWRITE'; exit 73; }
mkdir -p "$OUT"
cd "$EVAL"
for rep in 1 2 3 4 5; do
  rep_label=$(printf '%02d' "$rep")
  "$PY" "$SCRIPT" --checkpoint "$CKPT" --task-id 0 \
    --init-source benchmark --init-index 3 --max-steps 300 \
    --wait-steps 10 --n-action-steps 25 --seed 12345 \
    --results-dir "$OUT/fixed_init3_repeat_$rep_label"
done
for init in $(seq 0 19); do
  cli_seed=$((12345 + init))
  "$PY" "$SCRIPT" --checkpoint "$CKPT" --task-id 0 \
    --init-source benchmark --init-index "$init" --max-steps 300 \
    --wait-steps 10 --n-action-steps 25 --seed "$cli_seed" \
    --results-dir "$OUT/paired_20_init"
done
echo 'TEACHER_NATIVE_TASK0_EVALUATION_COMPLETE_PENDING_SUMMARY'
