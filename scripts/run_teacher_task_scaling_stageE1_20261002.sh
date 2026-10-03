#!/usr/bin/env bash
set -Eeuo pipefail

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

BASE=/root/smolvla-training-prep
EVAL=/root/smolvla-eval-prep
PROJECT=/root/smolvla-libero-compositional-generalization
REC="$BASE/recovery/task_scaling_stageE_20261002"
DEPLOY="$REC/launch_task0_stageE1_v2"
DATA="$BASE/datasets/lerobot/libero36_feasible_32_frozen_v1"
REPO_ID=local/libero36_feasible_32_frozen_v1
NAME=teacher_native_custom_task0_frozen32_5demo_lr10x_10k_20261002_v2
CFG="$DEPLOY/$NAME.json"
TRAIN_OUT="$BASE/results/training/$NAME"
TRAIN_LOG="$REC/$NAME.training.log"
PREFLIGHT="$REC/$NAME.preflight_v2.json"
PRELIGHT_LOG="$REC/$NAME.preflight_v2.log"
EVAL_ROOT="$EVAL/results/evaluations/teacher_stageE1_task0_frozen32_20261002_v2"
SUMMARY="$REC/$NAME.result.json"
EVAL_VIEW="$REC/$NAME.eval_checkpoint_view"
PY=/usr/local/miniconda3/envs/py312/bin/python
PREFLIGHT_SCRIPT="$DEPLOY/preflight_task_subset_balanced_v1.py"
TRAIN_WRAPPER=task_subset_balanced_train_wrapper_v1
EVAL_SCRIPT="$EVAL/scripts/eval_libero_strict_v1.py"
SUMMARIZER="$DEPLOY/summarize_teacher_task_scaling_stageE1_20261002.py"
STATE_DIR="$EVAL/results/evaluations/libero36_remaining_initial_states_v1/task_000"
BDDL=/root/smolvla-training-prep/data/libero_36/bddl/task_000_layout_1_akita_black_bowl_put_on_top_left.bddl
INSTRUCTION='pick up the akita black bowl and place it on the plate in the left region'
TRAIN_MARKER=TEACHER_STAGE_E1_EXIT_CODE=
RUNNER_MARKER=TEACHER_STAGE_E1_RUNNER_EXIT_CODE=

runner_exit() {
  code=$?
  echo "${RUNNER_MARKER}${code}"
  trap - EXIT
  exit "$code"
}
trap runner_exit EXIT

for path in "$TRAIN_OUT" "$TRAIN_LOG" "$PREFLIGHT" "$PRELIGHT_LOG" "$EVAL_ROOT" "$SUMMARY" "$EVAL_VIEW"; do
  test ! -e "$path" || { echo "REFUSE_OVERWRITE=$path" >&2; exit 73; }
done
test -f "$CFG" && test -f "$PREFLIGHT_SCRIPT" && test -f "$SUMMARIZER" && test -f "$EVAL_SCRIPT"
test -f "$DEPLOY/task_subset_balanced_sampler_v1.py" && test -f "$DEPLOY/task_subset_balanced_train_wrapper_v1.py"
test -f "$DATA/meta/info.json" && test -f "$DATA/meta/tasks.parquet"
test -f "$STATE_DIR/manifest.json" && test -f "$BDDL"

exec 9>"$BASE/teacher_control_gpu.lock"
flock -n 9 || { echo 'GPU_JOB_LOCKED' >&2; exit 75; }
conflicts=$(ps -eo comm=,args= | awk '$1 ~ /^python/ && $0 ~ /(lerobot_train|task_subset_balanced_train_wrapper|eval_libero_strict_v1)/ {print}')
if test -n "$conflicts"; then
  echo 'CONFLICTING_WORKLOAD_REFUSE_START' >&2; exit 75
fi
if test -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"; then
  echo 'GPU_COMPUTE_PROCESS_PRESENT' >&2; exit 75
fi

cd "$PROJECT"
export PYTHONPATH="$DEPLOY:$BASE:$PROJECT:$PROJECT/scripts:$EVAL/scripts${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c 'import json,sys; c=json.load(open(sys.argv[1])); assert c["job_name"] == sys.argv[2]; assert c["output_dir"] == sys.argv[3]; assert c["dataset"]["repo_id"] == "local/libero36_feasible_32_frozen_v1"; assert c["dataset"]["root"] == sys.argv[4]; assert c["dataset"]["episodes"] is None; assert c["steps"] == 10000 and c["batch_size"] == 8 and c["seed"] == 1000; assert c["policy"]["pretrained_path"] == "/root/smolvla-training-prep/models/smolvla_base_libero"; assert c["policy"]["optimizer_lr"] == 1e-4 and c["optimizer"]["lr"] == 1e-4 and c["scheduler"]["peak_lr"] == 1e-4 and c["scheduler"]["decay_lr"] == 2.5e-6; assert c["policy"]["chunk_size"] == 50 and c["policy"]["n_action_steps"] == 25' "$CFG" "$NAME" "$TRAIN_OUT" "$DATA"

"$PY" "$PREFLIGHT_SCRIPT" --config "$CFG" --dataset-root "$DATA" --repo-id "$REPO_ID" \
  --task-ids 0 --updates 10000 --batch-size 8 --seed 1000 --probe-batches 20 \
  --output "$PREFLIGHT" > "$PRELIGHT_LOG" 2>&1
grep -q '"passed": true' "$PREFLIGHT"
grep -q '"dataset_episodes": 160' "$PREFLIGHT"
grep -q '"episodes_per_requested_task": {' "$PREFLIGHT"
grep -q '"first_batch_shapes"' "$PREFLIGHT"

set +e
STAGEE_DATASET_REPO_ID="$REPO_ID" STAGEE_TASK_IDS=0 STAGEE_UPDATES=10000 STAGEE_BATCH_SIZE=8 \
STAGEE_SAMPLER_SEED=1000 "$PY" -m "$TRAIN_WRAPPER" --config_path="$CFG" > "$TRAIN_LOG" 2>&1
train_code=$?
set -e
echo "${TRAIN_MARKER}${train_code}" >> "$TRAIN_LOG"
if test "$train_code" -ne 0; then echo "TRAIN_FAILED_EXIT=$train_code" >&2; exit 74; fi

for step in 002500 005000 007500 010000; do
  test -f "$TRAIN_OUT/checkpoints/$step/pretrained_model/model.safetensors" || { echo "MISSING_CHECKPOINT=$step" >&2; exit 74; }
done
if grep -Eiq '(loss|grad_norm)[^[:cntrl:]]*(nan|(^|[^a-z])inf(inity)?([^a-z]|$))' "$TRAIN_LOG"; then
  echo 'NONFINITE_TRAINING_METRIC' >&2; exit 74
fi

mkdir -p "$EVAL_VIEW/010000"
ln -s "$TRAIN_OUT/checkpoints/010000/pretrained_model" "$EVAL_VIEW/010000/pretrained_model"
mkdir -p "$EVAL_ROOT"
cd "$EVAL"
"$PY" "$EVAL_SCRIPT" --task-id 0 --layout-id 1 --checkpoint-dir "$EVAL_VIEW" \
  --states-dir "$STATE_DIR" --bddl "$BDDL" --out "$EVAL_ROOT" --instruction "$INSTRUCTION" \
  --steps 280 --wait 10 --n-action-steps 25 --max-states 50 > "$EVAL_ROOT/eval.log" 2>&1

"$PY" "$SUMMARIZER" --run-id "$NAME" --training-log "$TRAIN_LOG" --training-output "$TRAIN_OUT" \
  --evaluation-root "$EVAL_ROOT" --preflight "$PREFLIGHT" --output "$SUMMARY"
echo "TEACHER_STAGE_E1_COMPLETE=$SUMMARY"
echo 'Stage E1 only. Stage E2 is not launched by this runner; the result must first be checked against the predeclared 28/50 gate.'
