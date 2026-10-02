#!/usr/bin/env bash
set -euo pipefail

# Diagnostic checkpoint progression at the already-used fixed task-0 init.
# This does not change training, paired-init results, or the single-task gate.
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1

BASE=/root/smolvla-training-prep
EVAL=/root/smolvla-eval-prep
REC="$BASE/recovery/native_task0_20261002"
NAME=teacher_native_task0_base_init_current_recipe_10k_20261002_v1
CKPT_ROOT="$BASE/results/training/$NAME/checkpoints"
OUT="$EVAL/results/$NAME/checkpoint_fixed_repeats_v1"
SUMMARY_OUT="$EVAL/results/$NAME/checkpoint_fixed_repeats_summary_v1.json"
TRAIN_LOG="$REC/$NAME.training.log"
FINAL_SUMMARY="$EVAL/results/$NAME/summary_recovery_v2.json"
SUMMARY_SCRIPT="$REC/summarize_teacher_native_task0_checkpoint_fixed_repeats_20261002.py"
PY=/usr/local/miniconda3/envs/py312/bin/python
SCRIPT="$EVAL/scripts/eval_v3_task0_state_capture_v1.py"

test -f "$TRAIN_LOG" && grep -q 'TEACHER_BASE_INIT_TRAIN_EXIT_CODE=0' "$TRAIN_LOG"
test -f "$FINAL_SUMMARY" && test -f "$SUMMARY_SCRIPT" && test -f "$SCRIPT"
test ! -e "$OUT" || { echo "REFUSE_OVERWRITE=$OUT" >&2; exit 73; }
test ! -e "$SUMMARY_OUT" || { echo "REFUSE_OVERWRITE=$SUMMARY_OUT" >&2; exit 73; }
for step in 002500 005000 007500; do
  test -f "$CKPT_ROOT/$step/pretrained_model/model.safetensors"
done

exec 9>"$BASE/teacher_control_gpu.lock"
flock -n 9 || { echo 'GPU_JOB_LOCKED' >&2; exit 75; }
mkdir -p "$OUT"
cd "$EVAL"

for step in 002500 005000 007500; do
  for rep in 01 02 03 04 05; do
    ckpt="$CKPT_ROOT/$step/pretrained_model"
    out="$OUT/checkpoint_$step/repeat_$rep"
    log="$OUT/checkpoint_${step}_repeat_${rep}.log"
    mkdir -p "$out"
    "$PY" "$SCRIPT" --checkpoint "$ckpt" --task-id 0 \
      --init-source benchmark --init-index 3 --max-steps 300 \
      --wait-steps 10 --n-action-steps 25 --seed 12345 \
      --device cuda --capture-state --results-dir "$out" > "$log" 2>&1
    test "$(find "$out" -maxdepth 1 -name '*_summary.csv' -type f | wc -l)" -eq 1
    test "$(find "$out" -maxdepth 1 -name '*_actions.csv' -type f | wc -l)" -eq 1
  done
done

"$PY" "$SUMMARY_SCRIPT" \
  --eval-root "$OUT" \
  --current-final-summary "$FINAL_SUMMARY" \
  --training-log "$TRAIN_LOG" \
  --output "$SUMMARY_OUT"
echo "TEACHER_BASE_INIT_CHECKPOINT_AUDIT_COMPLETE=$OUT"
