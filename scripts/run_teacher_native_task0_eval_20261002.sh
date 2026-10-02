#!/usr/bin/env bash
set -euo pipefail
BASE=/root/smolvla-training-prep
EVAL=/root/smolvla-eval-prep
REC="$BASE/recovery/native_task0_20261002"
CKPT="$BASE/results/training/teacher_native_task0_current_pipeline_10k_20261002_v1/checkpoints/010000/pretrained_model"
OUT="$EVAL/results/teacher_native_task0_current_pipeline_10k_eval_20261002_v1"
PY=/usr/local/miniconda3/envs/py312/bin/python
SCRIPT="$EVAL/scripts/eval_v3_task0_state_capture_v1.py"
exec 9>"$BASE/teacher_control_gpu.lock"
flock -n 9 || { echo 'GPU_JOB_LOCKED'; exit 75; }
test -f "$REC/training.log" || { echo 'TRAIN_LOG_MISSING'; exit 74; }
grep -q 'TEACHER_SINGLE_TASK_TRAIN_EXIT_CODE=0' "$REC/training.log" || {
  echo 'TRAINING_NOT_CONFIRMED_SUCCESSFUL'; exit 74;
}
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
