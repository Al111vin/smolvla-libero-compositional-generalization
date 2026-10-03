#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1
BASE=/root/smolvla-training-prep
EVAL=/root/smolvla-eval-prep
ROOT="$BASE/recovery/task_scaling_native_spatial_20261003"
REC="$ROOT/task1_single_control_v1"
NAME=teacher_native_spatial_task1_single_current_recipe_40k_batch2_v1
CFG="$REC/$NAME.json"
PREFLIGHT="$REC/$NAME.preflight.json"
TRAIN_OUT="$BASE/results/training/$NAME"
CKPT_ROOT="$TRAIN_OUT/checkpoints"
TRAIN_LOG="$REC/$NAME.training.log"
EVAL_SCRIPT="$EVAL/scripts/eval_v3_task0_state_capture_v1.py"
EVAL_OUT="$EVAL/results/$NAME"
SUMMARY="$EVAL_OUT/summary.json"
JOINT4_EVAL="$EVAL/results/teacher_native_spatial_tasks0_3_balanced_current_recipe_40k_v2_protocol_corrected_v1/paired_20_init/task_1"
PY=/usr/local/miniconda3/envs/py312/bin/python
TRAIN_MARKER='TEACHER_NATIVE_SPATIAL_TASK1_SINGLE_40K_TRAIN_EXIT_CODE='

for path in "$TRAIN_OUT" "$TRAIN_LOG" "$EVAL_OUT" "$SUMMARY"; do
  test ! -e "$path" || { echo "REFUSE_OVERWRITE=$path" >&2; exit 73; }
done
test -s "$CFG" && test -s "$PREFLIGHT" && test -s "$ROOT/conversion_manifest.json"
grep -q '"status": "passed"' "$PREFLIGHT"

exec 9>"$BASE/teacher_control_gpu.lock"
flock -n 9 || { echo 'GPU_JOB_LOCKED' >&2; exit 75; }
if pgrep -af '[l]erobot_train|[t]ask_balanced_train_wrapper|[e]val_v3_task0_state_capture' >/dev/null; then
  echo 'CONFLICTING_WORKLOAD_REFUSE_START' >&2; exit 75
fi
if test -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"; then
  echo 'GPU_COMPUTE_PROCESS_PRESENT' >&2; exit 75
fi

cd "$BASE"
export PYTHONPATH="$BASE${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - "$CFG" "$PREFLIGHT" "$EVAL_SCRIPT" <<'PY'
import hashlib, json, pathlib, sys
cfg_path, preflight_path, evaluator_path = map(pathlib.Path, sys.argv[1:])
cfg = json.loads(cfg_path.read_text())
preflight = json.loads(preflight_path.read_text())
assert preflight['status'] == 'passed' and preflight['training_started'] is False
assert cfg['job_name'] == 'teacher_native_spatial_task1_single_current_recipe_40k_batch2_v1'
assert cfg['output_dir'] == '/root/smolvla-training-prep/results/training/' + cfg['job_name']
assert cfg['dataset']['repo_id'] == 'local/libero_spatial_tasks0_3_native_v1'
assert cfg['dataset']['episodes'] == list(range(50, 100))
assert cfg['steps'] == 40000 and cfg['batch_size'] == 2 and cfg['seed'] == 1000
assert cfg['policy']['pretrained_path'] == '/root/smolvla-training-prep/models/smolvla_base_libero'
assert cfg['policy']['chunk_size'] == 50 and cfg['policy']['n_action_steps'] == 25
assert cfg['policy']['optimizer_lr'] == 1e-4 and cfg['optimizer']['lr'] == 1e-4
assert cfg['scheduler']['num_warmup_steps'] == 3000 and cfg['scheduler']['num_decay_steps'] == 90000
assert hashlib.sha256(evaluator_path.read_bytes()).hexdigest() == preflight['evaluator_sha256']
assert preflight['new_config_sha256'] == hashlib.sha256(cfg_path.read_bytes()).hexdigest()
PY

set +e
"$PY" -m scripts.lerobot_train_loco_compat --config_path="$CFG" > "$TRAIN_LOG" 2>&1
train_code=$?
set -e
echo "${TRAIN_MARKER}${train_code}" >> "$TRAIN_LOG"
test "$train_code" -eq 0 || { echo "TRAIN_FAILED_EXIT=$train_code" >&2; exit 74; }
for step in 010000 020000 030000 040000; do
  test -s "$CKPT_ROOT/$step/pretrained_model/model.safetensors" || { echo "MISSING_CHECKPOINT=$step" >&2; exit 74; }
done

mkdir -p "$EVAL_OUT"
cd "$EVAL"
# Evaluator records effective_seed=CLI_seed+init_index; this matches the registered 12345+2*i values.
for init in $(seq 0 19); do
  cli_seed=$((12345 + init))
  out="$EVAL_OUT/paired_20_init/task1/init_$init"
  log="$EVAL_OUT/paired_20_init/task1/init_$init.log"
  test ! -e "$out" && test ! -e "$log"
  "$PY" "$EVAL_SCRIPT" --checkpoint "$CKPT_ROOT/040000/pretrained_model" --task-id 1 \
    --init-source benchmark --init-index "$init" --max-steps 300 --wait-steps 10 \
    --n-action-steps 25 --seed "$cli_seed" --device cuda --capture-state \
    --results-dir "$out" > "$log" 2>&1
done

# Four extra repeats + the paired init3 row yield n=5 at the same effective seed 12351.
for rep in $(seq 1 4); do
  label=$(printf '%02d' "$rep")
  out="$EVAL_OUT/fixed_task1_init3/repeat_$label"
  log="$EVAL_OUT/fixed_task1_init3/repeat_$label.log"
  test ! -e "$out" && test ! -e "$log"
  "$PY" "$EVAL_SCRIPT" --checkpoint "$CKPT_ROOT/040000/pretrained_model" --task-id 1 \
    --init-source benchmark --init-index 3 --max-steps 300 --wait-steps 10 \
    --n-action-steps 25 --seed 12348 --device cuda --capture-state \
    --results-dir "$out" > "$log" 2>&1
done

"$PY" "$REC/summarize_teacher_native_spatial_task1_single_control_v1.py" \
  --eval-root "$EVAL_OUT" --joint4-eval-root "$JOINT4_EVAL" \
  --training-log "$TRAIN_LOG" --checkpoint-root "$CKPT_ROOT" \
  --output "$SUMMARY" --run-id "$NAME"
echo "TEACHER_NATIVE_SPATIAL_TASK1_SINGLE_CONTROL_COMPLETE=$SUMMARY"
