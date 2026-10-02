#!/usr/bin/env bash
set -euo pipefail
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
BASE=/root/smolvla-training-prep
EVAL=/root/smolvla-eval-prep
REC="$BASE/recovery/native_task0_20261002"
NAME=teacher_native_task0_base_init_current_recipe_10k_20261002_v1
CKPT="$BASE/results/training/$NAME/checkpoints/010000/pretrained_model"
OUT="$EVAL/results/$NAME/eval_v1"
PY=/usr/local/miniconda3/envs/py312/bin/python
SCRIPT="$EVAL/scripts/eval_v3_task0_state_capture_v1.py"
TRAIN_LOG="$REC/${NAME}.training.log"
OK_MARKER='TEACHER_BASE_INIT_TRAIN_EXIT_CODE=0'
EXIT_MARKER='TEACHER_BASE_INIT_TRAIN_EXIT_CODE='
for attempt in $(seq 1 120); do
  if grep -q "$OK_MARKER" "$TRAIN_LOG"; then break; fi
  if grep -q "$EXIT_MARKER" "$TRAIN_LOG"; then echo 'TRAINING_EXITED_NONZERO'; exit 74; fi
  sleep 30
done
grep -q "$OK_MARKER" "$TRAIN_LOG" || { echo 'TRAINING_WAIT_TIMEOUT'; exit 74; }
exec 9>"$BASE/teacher_control_gpu.lock"
locked=0
for attempt in $(seq 1 24); do
  if flock -n 9; then locked=1; break; fi
  sleep 5
done
test "$locked" -eq 1 || { echo 'GPU_JOB_LOCKED_AFTER_TRAINING'; exit 75; }
for step in 002500 005000 007500 010000; do
  test -f "$BASE/results/training/$NAME/checkpoints/$step/pretrained_model/model.safetensors"
done
test -f "$SCRIPT" || { echo 'EVALUATOR_MISSING'; exit 74; }
test ! -e "$OUT" || { echo 'OUTPUT_EXISTS_REFUSE_OVERWRITE'; exit 73; }
mkdir -p "$OUT"
cd "$EVAL"
for rep in 1 2 3 4 5; do
  label=$(printf '%02d' "$rep")
  "$PY" "$SCRIPT" --checkpoint "$CKPT" --task-id 0 \
    --init-source benchmark --init-index 3 --max-steps 300 \
    --wait-steps 10 --n-action-steps 25 --seed 12345 --device cuda --capture-state \
    --results-dir "$OUT/fixed_init3_repeat_$label" > "$OUT/fixed_init3_repeat_$label.log" 2>&1
done
for init in $(seq 0 19); do
  seed=$((12345 + init))
  "$PY" "$SCRIPT" --checkpoint "$CKPT" --task-id 0 \
    --init-source benchmark --init-index "$init" --max-steps 300 \
    --wait-steps 10 --n-action-steps 25 --seed "$seed" --device cuda --capture-state \
    --results-dir "$OUT/paired_20_init" > "$OUT/paired_20_init_${init}.log" 2>&1
done
echo 'TEACHER_BASE_INIT_EVALUATION_COMPLETE_PENDING_SUMMARY' > "$REC/${NAME}.evaluation_complete.marker"
