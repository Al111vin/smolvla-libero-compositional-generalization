#!/usr/bin/env bash
set -euo pipefail
BASE=/root/smolvla-training-prep
REC="$BASE/recovery/native_task0_20261002"
NAME=teacher_native_task0_base_init_current_recipe_10k_20261002_v1
CFG="$REC/${NAME}.json"
OUT="$BASE/results/training/$NAME"
PY=/usr/local/miniconda3/envs/py312/bin/python
exec 9>"$BASE/teacher_control_gpu.lock"
flock -n 9 || { echo 'GPU_JOB_LOCKED'; exit 75; }
test ! -e "$OUT" || { echo 'OUTPUT_EXISTS_REFUSE_OVERWRITE'; exit 73; }
if pgrep -f 'python.*(lerobot_train|task_balanced_train|blind_eval|eval_v3)' >/dev/null; then
  echo 'CONFLICTING_WORKLOAD_REFUSE_START'; exit 75
fi
if test -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"; then
  echo 'GPU_COMPUTE_PROCESS_PRESENT'; exit 75
fi
cd "$BASE"
export PYTHONPATH="$BASE${PYTHONPATH:+:$PYTHONPATH}"
trap 'code=$?; echo TEACHER_BASE_INIT_TRAIN_EXIT_CODE=$code' EXIT
"$PY" "$REC/preflight_teacher_native_task0_20261002.py" "$CFG"
"$PY" -m scripts.lerobot_train_loco_compat --config_path="$CFG"
for step in 002500 005000 007500 010000; do
  test -f "$OUT/checkpoints/$step/pretrained_model/model.safetensors"
done
echo 'TEACHER_BASE_INIT_TRAIN_COMPLETE_PENDING_CLOSED_LOOP_EVAL'
