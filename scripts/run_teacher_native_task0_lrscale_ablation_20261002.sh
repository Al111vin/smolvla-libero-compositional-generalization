#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1
BASE=/root/smolvla-training-prep
EVAL=/root/smolvla-eval-prep
REC="$BASE/recovery/native_task0_20261002"
SOURCE_NAME=teacher_native_task0_base_init_current_recipe_10k_20261002_v1
NAME=teacher_native_task0_base_init_lr10x_current_recipe_10k_20261002_v1
CFG="$REC/$NAME.json"
PRELIGHT="$REC/$NAME.preflight.json"
TRAIN_LOG="$REC/$NAME.training.log"
TRAIN_MARKER='TEACHER_TASK0_LR10X_TRAIN_EXIT_CODE='
CKPT_ROOT="$BASE/results/training/$NAME/checkpoints"
TRAIN_OUT="$BASE/results/training/$NAME"
EVAL_OUT="$EVAL/results/$NAME/eval_v1"
SUMMARY_OUT="$EVAL/results/$NAME/summary.json"
PY=/usr/local/miniconda3/envs/py312/bin/python
EVAL_SCRIPT="$EVAL/scripts/eval_v3_task0_state_capture_v1.py"
PREFLIGHT="$REC/preflight_teacher_native_task0_20261002.py"
SUMMARIZER="$REC/summarize_teacher_native_task0_base_init_eval_20261002.py"
REFERENCE="$REC/teacher_native_task0_current_pipeline_10k_eval_summary_20261002.json"

for path in "$TRAIN_OUT" "$EVAL_OUT" "$SUMMARY_OUT" "$PRELIGHT" "$TRAIN_LOG"; do
  test ! -e "$path" || { echo "REFUSE_OVERWRITE=$path" >&2; exit 73; }
done
test -f "$CFG" && test -f "$PREFLIGHT" && test -f "$SUMMARIZER" && test -f "$REFERENCE" && test -f "$EVAL_SCRIPT"

exec 9>"$BASE/teacher_control_gpu.lock"
flock -n 9 || { echo 'GPU_JOB_LOCKED'; exit 75; }
if pgrep -f 'python.*(lerobot_train|task_balanced_train|blind_eval|eval_v3)' >/dev/null; then
  echo 'CONFLICTING_WORKLOAD_REFUSE_START'; exit 75
fi
if test -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"; then
  echo 'GPU_COMPUTE_PROCESS_PRESENT'; exit 75
fi

cd "$BASE"
export PYTHONPATH="$BASE${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c 'import json,sys; c=json.load(open(sys.argv[1])); assert c["job_name"] == "teacher_native_task0_base_init_lr10x_current_recipe_10k_20261002_v1"; assert c["output_dir"] == "/root/smolvla-training-prep/results/training/teacher_native_task0_base_init_lr10x_current_recipe_10k_20261002_v1"; assert c["dataset"]["repo_id"] == "local/libero_spatial_task0_reconstructed_20261002"; assert c["policy"]["pretrained_path"] == "/root/smolvla-training-prep/models/smolvla_base_libero"; assert c["steps"] == 10000 and c["batch_size"] == 8 and c["seed"] == 1000; assert c["policy"]["chunk_size"] == 50 and c["policy"]["n_action_steps"] == 25; assert c["policy"]["optimizer_lr"] == 1e-4 and c["optimizer"]["lr"] == 1e-4 and c["scheduler"]["peak_lr"] == 1e-4 and c["scheduler"]["decay_lr"] == 2.5e-6' "$CFG"
"$PY" "$PREFLIGHT" "$CFG" 1e-4 > "$PRELIGHT"
grep -q '"passed": true' "$PRELIGHT"

set +e
"$PY" -m scripts.lerobot_train_loco_compat --config_path="$CFG" > "$TRAIN_LOG" 2>&1
train_code=$?
set -e
echo "${TRAIN_MARKER}${train_code}" >> "$TRAIN_LOG"
test "$train_code" -eq 0 || { echo "TRAIN_FAILED_EXIT=$train_code" >&2; exit 74; }
for step in 002500 005000 007500 010000; do
  test -f "$CKPT_ROOT/$step/pretrained_model/model.safetensors" || { echo "MISSING_CHECKPOINT=$step" >&2; exit 74; }
done

mkdir -p "$EVAL_OUT"
cd "$EVAL"
for rep in 1 2 3 4 5; do
  label=$(printf '%02d' "$rep")
  "$PY" "$EVAL_SCRIPT" --checkpoint "$CKPT_ROOT/010000/pretrained_model" --task-id 0 \
    --init-source benchmark --init-index 3 --max-steps 300 \
    --wait-steps 10 --n-action-steps 25 --seed 12345 --device cuda --capture-state \
    --results-dir "$EVAL_OUT/fixed_init3_repeat_$label" \
    > "$EVAL_OUT/fixed_init3_repeat_$label.log" 2>&1
done
for init in $(seq 0 19); do
  seed=$((12345 + init))
  "$PY" "$EVAL_SCRIPT" --checkpoint "$CKPT_ROOT/010000/pretrained_model" --task-id 0 \
    --init-source benchmark --init-index "$init" --max-steps 300 \
    --wait-steps 10 --n-action-steps 25 --seed "$seed" --device cuda --capture-state \
    --results-dir "$EVAL_OUT/paired_20_init" \
    > "$EVAL_OUT/paired_20_init_${init}.log" 2>&1
done

"$PY" "$SUMMARIZER" --eval-root "$EVAL_OUT" --historical-reference "$REFERENCE" \
  --output "$SUMMARY_OUT" --training-log "$TRAIN_LOG" --run-id "$NAME" \
  --training-success-marker "${TRAIN_MARKER}0"
echo "TEACHER_TASK0_LR10X_EXPERIMENT_COMPLETE=$SUMMARY_OUT"
