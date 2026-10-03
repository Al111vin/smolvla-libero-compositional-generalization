#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1
BASE=/root/smolvla-training-prep
EVAL=/root/smolvla-eval-prep
REC="$BASE/recovery/task_scaling_native_spatial_20261003"
NAME=${RUN_NAME:-teacher_native_spatial_tasks0_3_balanced_current_recipe_40k_v1}
CFG="$REC/$NAME.json"
PREFLIGHT="$REC/$NAME.preflight.json"
TRAIN_OUT="$BASE/results/training/$NAME"
CKPT_ROOT="$TRAIN_OUT/checkpoints"
TRAIN_LOG="$REC/$NAME.training.log"
EVAL_SCRIPT="$EVAL/scripts/eval_v3_task0_state_capture_v1.py"
EVAL_OUT="$EVAL/results/$NAME"
SUMMARY="$EVAL_OUT/summary.json"
PY=/usr/local/miniconda3/envs/py312/bin/python
TRAIN_MARKER='TEACHER_NATIVE_SPATIAL_4TASK_40K_TRAIN_EXIT_CODE='

for path in "$TRAIN_OUT" "$TRAIN_LOG" "$EVAL_OUT" "$SUMMARY"; do
  test ! -e "$path" || { echo "REFUSE_OVERWRITE=$path" >&2; exit 73; }
done
test -s "$CFG" && test -s "$PREFLIGHT" && test -s "$REC/conversion_manifest.json"
grep -q '"status": "passed"' "$PREFLIGHT"

exec 9>"$BASE/teacher_control_gpu.lock"
flock -n 9 || { echo 'GPU_JOB_LOCKED'; exit 75; }
if pgrep -af '[l]erobot_train|[t]ask_balanced_train_wrapper|[e]val_v3_task0_state_capture' >/dev/null; then
  echo 'CONFLICTING_WORKLOAD_REFUSE_START'; exit 75
fi
if test -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"; then
  echo 'GPU_COMPUTE_PROCESS_PRESENT'; exit 75
fi

cd "$BASE"
export PYTHONPATH="$BASE${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - "$CFG" "$REC/conversion_manifest.json" "$EVAL_SCRIPT" <<'PY'
import hashlib, json, pathlib, sys
cfg_path, manifest_path, evaluator_path = map(pathlib.Path, sys.argv[1:])
cfg = json.loads(cfg_path.read_text())
manifest = json.loads(manifest_path.read_text())
assert cfg['steps'] == 40000 and cfg['batch_size'] == 8
assert cfg['dataset']['repo_id'] == 'local/libero_spatial_tasks0_3_native_v1'
assert cfg['dataset']['episodes'] == list(range(200))
assert cfg['job_name'] == pathlib.Path(cfg['output_dir']).name
assert cfg['output_dir'] == f'/root/smolvla-training-prep/results/training/{cfg["job_name"]}'
assert manifest['episode_count'] == 200 and manifest['frame_count'] == 22709
assert manifest['task_index_by_libero_id'] == {'0': 0, '1': 1, '2': 2, '3': 3}
assert hashlib.sha256(evaluator_path.read_bytes()).hexdigest() == '89fd36a89dc45a56382219d4c3f6e5d12a3b1689abb3440ec29be346815af7cb'
PY

set +e
"$PY" -m scripts.task_balanced_train_wrapper_v2 --config_path="$CFG" > "$TRAIN_LOG" 2>&1
train_code=$?
set -e
echo "${TRAIN_MARKER}${train_code}" >> "$TRAIN_LOG"
test "$train_code" -eq 0 || { echo "TRAIN_FAILED_EXIT=$train_code" >&2; exit 74; }
for step in 010000 020000 030000 040000; do
  test -s "$CKPT_ROOT/$step/pretrained_model/model.safetensors" || { echo "MISSING_CHECKPOINT=$step" >&2; exit 74; }
done

mkdir -p "$EVAL_OUT"
# Same fixed task-0/init-3 condition used to qualify the single-task candidate.
for step in 010000 020000 030000 040000; do
  out="$EVAL_OUT/checkpoint_probe/step_$step/task0_init3"
  log="$EVAL_OUT/checkpoint_probe/step_$step.log"
  test ! -e "$out" && test ! -e "$log"
  "$PY" "$EVAL_SCRIPT" --checkpoint "$CKPT_ROOT/$step/pretrained_model" --task-id 0 \
    --init-source benchmark --init-index 3 --max-steps 300 --wait-steps 10 \
    --n-action-steps 25 --seed 12351 --device cuda --capture-state \
    --results-dir "$out" > "$log" 2>&1
done

# Registered paired evaluation: same 20 init indices and effective seeds for all tasks.
for task in 0 1 2 3; do
  for init in $(seq 0 19); do
    seed=$((12345 + 2 * init))
    out="$EVAL_OUT/paired_20_init/task_$task/init_$init"
    log="$EVAL_OUT/paired_20_init/task_$task/init_$init.log"
    test ! -e "$out" && test ! -e "$log"
    "$PY" "$EVAL_SCRIPT" --checkpoint "$CKPT_ROOT/040000/pretrained_model" --task-id "$task" \
      --init-source benchmark --init-index "$init" --max-steps 300 --wait-steps 10 \
      --n-action-steps 25 --seed "$seed" --device cuda --capture-state \
      --results-dir "$out" > "$log" 2>&1
  done
done

# Repeat the anchor positive condition five times to observe binary stability.
for rep in $(seq 1 5); do
  label=$(printf '%02d' "$rep")
  out="$EVAL_OUT/fixed_task0_init3/repeat_$label"
  log="$EVAL_OUT/fixed_task0_init3/repeat_$label.log"
  test ! -e "$out" && test ! -e "$log"
  "$PY" "$EVAL_SCRIPT" --checkpoint "$CKPT_ROOT/040000/pretrained_model" --task-id 0 \
    --init-source benchmark --init-index 3 --max-steps 300 --wait-steps 10 \
    --n-action-steps 25 --seed 12351 --device cuda --capture-state \
    --results-dir "$out" > "$log" 2>&1
done

"$PY" "$BASE/scripts/summarize_teacher_native_spatial_4task_eval_v1.py" \
  --eval-root "$EVAL_OUT" \
  --baseline-eval-root "$EVAL/results/teacher_native_task0_base_init_lr10x_current_recipe_10k_20261002_v1/eval_v1/paired_20_init" \
  --run-id "$NAME" \
  --checkpoint-root "$CKPT_ROOT" \
  --output "$SUMMARY"
echo "TEACHER_NATIVE_SPATIAL_4TASK_40K_COMPLETE=$SUMMARY"
