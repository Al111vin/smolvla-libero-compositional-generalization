#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1

BASE=/root/smolvla-training-prep
EVAL=/root/smolvla-eval-prep
ROOT="$BASE/recovery/task_scaling_native_spatial_20261003"
REC="$ROOT/task0_single_control_v1"
NAME=teacher_native_spatial_task0_single_current_recipe_40k_batch2_v1
CFG="$REC/$NAME.json"
PREFLIGHT="$REC/$NAME.preflight.json"
TRAIN_OUT="$BASE/results/training/$NAME"
CKPT_ROOT="$TRAIN_OUT/checkpoints"
TRAIN_LOG="$REC/$NAME.training.log"
DRIVER_LOG="$REC/$NAME.eval_resume_driver.log"
EVAL_SCRIPT="$EVAL/scripts/eval_v3_task0_state_capture_v1.py"
SUMMARIZER="$REC/summarize_teacher_native_spatial_single_task_control_v1.py"
EVAL_OUT="$EVAL/results/$NAME"
SUMMARY="$EVAL_OUT/summary.json"
JOINT4_EVAL="$EVAL/results/teacher_native_spatial_tasks0_3_balanced_current_recipe_40k_v2_protocol_corrected_v1/paired_20_init/task_0"
EXPECTED_EVAL_SHA256=89fd36a89dc45a56382219d4c3f6e5d12a3b1689abb3440ec29be346815af7cb
PY=/usr/local/miniconda3/envs/py312/bin/python

finish() {
  rc=$?
  trap - EXIT
  printf 'EVAL_RESUME_EXIT_CODE=%s\n' "$rc"
  exit "$rc"
}
trap finish EXIT

test -s "$CFG" && test -s "$PREFLIGHT" && test -s "$TRAIN_LOG"
test -s "$EVAL_SCRIPT" && test -s "$SUMMARIZER"
test -d "$EVAL_OUT"
test ! -e "$SUMMARY" || { echo "REFUSE_OVERWRITE=$SUMMARY" >&2; exit 73; }
grep -q 'End of training' "$TRAIN_LOG"
grep -q '^TEACHER_NATIVE_SPATIAL_TASK0_SINGLE_40K_TRAIN_EXIT_CODE=0$' "$TRAIN_LOG"

for step in 010000 020000 030000 040000; do
  test -s "$CKPT_ROOT/$step/pretrained_model/model.safetensors" || {
    echo "MISSING_CHECKPOINT=$step" >&2
    exit 74
  }
done

actual_eval_sha256=$(sha256sum "$EVAL_SCRIPT" | awk '{print $1}')
test "$actual_eval_sha256" = "$EXPECTED_EVAL_SHA256" || {
  echo "EVALUATOR_HASH_MISMATCH=$actual_eval_sha256" >&2
  exit 76
}
"$PY" - "$CFG" "$PREFLIGHT" <<'PY'
import hashlib, json, pathlib, sys
cfg_path, preflight_path = map(pathlib.Path, sys.argv[1:])
cfg = json.loads(cfg_path.read_text())
preflight = json.loads(preflight_path.read_text())
assert preflight['status'] == 'passed'
assert preflight['training_started'] is False
assert cfg['job_name'] == 'teacher_native_spatial_task0_single_current_recipe_40k_batch2_v1'
assert cfg['steps'] == 40000 and cfg['batch_size'] == 2 and cfg['seed'] == 1000
assert cfg['dataset']['repo_id'] == 'local/libero_spatial_tasks0_3_native_v1'
assert cfg['dataset']['episodes'] == list(range(50))
assert cfg['policy']['chunk_size'] == 50 and cfg['policy']['n_action_steps'] == 25
assert preflight['evaluator_sha256'] == '89fd36a89dc45a56382219d4c3f6e5d12a3b1689abb3440ec29be346815af7cb'
PY

# This is an evaluation-only recovery. Never train or replace any existing result.
if find "$EVAL_OUT" -mindepth 1 -print -quit | grep -q .; then
  echo "REFUSE_NONEMPTY_EVAL_ROOT=$EVAL_OUT" >&2
  exit 73
fi
exec 9>"$BASE/teacher_control_gpu.lock"
flock -n 9 || { echo 'GPU_JOB_LOCKED' >&2; exit 75; }
if pgrep -af '[l]erobot_train|[t]ask_balanced_train_wrapper|[e]val_v3_task0_state_capture' >/dev/null; then
  echo 'CONFLICTING_WORKLOAD_REFUSE_START' >&2
  exit 75
fi
if test -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"; then
  echo 'GPU_COMPUTE_PROCESS_PRESENT' >&2
  exit 75
fi

mkdir -p "$EVAL_OUT/checkpoint_probe" \
  "$EVAL_OUT/paired_20_init/task0" \
  "$EVAL_OUT/fixed_task0_init3"
cd "$EVAL"
export PYTHONPATH="$BASE${PYTHONPATH:+:$PYTHONPATH}"

for step in 010000 020000 030000; do
  out="$EVAL_OUT/checkpoint_probe/step_$step/task0_init3"
  log="$EVAL_OUT/checkpoint_probe/step_$step.log"
  mkdir -p "$EVAL_OUT/checkpoint_probe/step_$step"
  test ! -e "$out" && test ! -e "$log"
  echo "START_CHECKPOINT_PROBE=$step"
  "$PY" "$EVAL_SCRIPT" --checkpoint "$CKPT_ROOT/$step/pretrained_model" --task-id 0 \
    --init-source benchmark --init-index 3 --max-steps 300 --wait-steps 10 \
    --n-action-steps 25 --seed 12348 --device cuda --capture-state \
    --results-dir "$out" > "$log" 2>&1
  echo "COMPLETE_CHECKPOINT_PROBE=$step"
done

for init in $(seq 0 19); do
  cli_seed=$((12345 + init))
  out="$EVAL_OUT/paired_20_init/task0/init_$init"
  log="$EVAL_OUT/paired_20_init/task0/init_$init.log"
  test ! -e "$out" && test ! -e "$log"
  echo "START_PAIRED_INIT=$init"
  "$PY" "$EVAL_SCRIPT" --checkpoint "$CKPT_ROOT/040000/pretrained_model" --task-id 0 \
    --init-source benchmark --init-index "$init" --max-steps 300 --wait-steps 10 \
    --n-action-steps 25 --seed "$cli_seed" --device cuda --capture-state \
    --results-dir "$out" > "$log" 2>&1
  echo "COMPLETE_PAIRED_INIT=$init"
done

for rep in $(seq 1 4); do
  label=$(printf '%02d' "$rep")
  out="$EVAL_OUT/fixed_task0_init3/repeat_$label"
  log="$EVAL_OUT/fixed_task0_init3/repeat_$label.log"
  test ! -e "$out" && test ! -e "$log"
  echo "START_INIT3_REPEAT=$label"
  "$PY" "$EVAL_SCRIPT" --checkpoint "$CKPT_ROOT/040000/pretrained_model" --task-id 0 \
    --init-source benchmark --init-index 3 --max-steps 300 --wait-steps 10 \
    --n-action-steps 25 --seed 12348 --device cuda --capture-state \
    --results-dir "$out" > "$log" 2>&1
  echo "COMPLETE_INIT3_REPEAT=$label"
done

"$PY" "$SUMMARIZER" --task-id 0 --eval-root "$EVAL_OUT" \
  --joint4-eval-root "$JOINT4_EVAL" --training-log "$TRAIN_LOG" \
  --checkpoint-root "$CKPT_ROOT" --output "$SUMMARY" --run-id "$NAME"
echo "TEACHER_NATIVE_SPATIAL_TASK0_SINGLE_CONTROL_COMPLETE=$SUMMARY"
