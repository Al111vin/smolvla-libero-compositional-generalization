#!/usr/bin/env bash
set -euo pipefail

# Recover the interrupted 2026-10-02 task-0 evaluation without overwriting its
# original partial output. Completed fixed repeats and paired init 0 are copied
# into a fresh root; only missing paired inits are rerun, cache-only, with each
# retry isolated from accepted CSVs.
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1

BASE=/root/smolvla-training-prep
EVAL=/root/smolvla-eval-prep
REC="$BASE/recovery/native_task0_20261002"
NAME=teacher_native_task0_base_init_current_recipe_10k_20261002_v1
OLD_OUT="$EVAL/results/$NAME/eval_v1"
OUT="$EVAL/results/$NAME/eval_recovery_v2"
CKPT="$BASE/results/training/$NAME/checkpoints/010000/pretrained_model"
TRAIN_LOG="$REC/$NAME.training.log"
SUMMARY="$REC/summarize_teacher_native_task0_base_init_eval_20261002.py"
REFERENCE="$REC/teacher_native_task0_current_pipeline_10k_eval_summary_20261002.json"
SUMMARY_OUT="$EVAL/results/$NAME/summary_recovery_v2.json"
PY=/usr/local/miniconda3/envs/py312/bin/python
SCRIPT="$EVAL/scripts/eval_v3_task0_state_capture_v1.py"

test -f "$TRAIN_LOG" && grep -q 'TEACHER_BASE_INIT_TRAIN_EXIT_CODE=0' "$TRAIN_LOG"
test -f "$CKPT/model.safetensors"
test -f "$SCRIPT" && test -f "$SUMMARY" && test -f "$REFERENCE"
test -d "$OLD_OUT"
test ! -e "$OUT" || { echo "REFUSE_OVERWRITE=$OUT" >&2; exit 73; }
test ! -e "$SUMMARY_OUT" || { echo "REFUSE_OVERWRITE=$SUMMARY_OUT" >&2; exit 73; }

# Cooperate with the project's single-GPU lock.
exec 9>"$BASE/teacher_control_gpu.lock"
flock -n 9 || { echo 'GPU_JOB_LOCKED' >&2; exit 75; }

mkdir -p "$OUT/paired_20_init" "$OUT/attempts"
for rep in 01 02 03 04 05; do
  src="$OLD_OUT/fixed_init3_repeat_$rep"
  test -d "$src" || { echo "MISSING_FIXED_REPEAT=$rep" >&2; exit 74; }
  cp -a "$src" "$OUT/"
  cp -a "$OLD_OUT/fixed_init3_repeat_$rep.log" "$OUT/"
done

# Init 0 completed before the transient Hub disconnect; retain its exact CSVs.
shopt -s nullglob
init0_summaries=("$OLD_OUT/paired_20_init"/*benchmark_init0*_summary.csv)
init0_actions=("$OLD_OUT/paired_20_init"/*benchmark_init0*_actions.csv)
[[ ${#init0_summaries[@]} -eq 1 && ${#init0_actions[@]} -eq 1 ]] || {
  echo 'EXPECTED_ONE_COMPLETED_INIT0_SUMMARY_AND_ACTIONS' >&2; exit 74;
}
for file in "${init0_summaries[0]}" "${init0_actions[0]}"; do
  target="$OUT/paired_20_init/$(basename "$file")"
  test ! -e "$target" || { echo "REFUSE_OVERWRITE=$target" >&2; exit 73; }
  cp -a "$file" "$target"
done
cp -a "$OLD_OUT/paired_20_init_0.log" "$OUT/paired_20_init/paired_20_init_0_reused_from_eval_v1.log"

for init in $(seq 1 19); do
  cli_seed=$((12345 + init))
  effective_seed=$((cli_seed + init))
  success=0
  for attempt in 1 2 3; do
    label=$(printf '%02d' "$init")
    attempt_dir="$OUT/attempts/init_${label}_attempt_${attempt}"
    log="$OUT/attempts/init_${label}_attempt_${attempt}.log"
    mkdir -p "$attempt_dir"
    set +e
    "$PY" "$SCRIPT" --checkpoint "$CKPT" --task-id 0 \
      --init-source benchmark --init-index "$init" --max-steps 300 \
      --wait-steps 10 --n-action-steps 25 --seed "$cli_seed" \
      --device cuda --capture-state --results-dir "$attempt_dir" > "$log" 2>&1
    rc=$?
    set -e
    summaries=("$attempt_dir"/*_summary.csv)
    actions=("$attempt_dir"/*_actions.csv)
    if [[ $rc -eq 0 && ${#summaries[@]} -eq 1 && ${#actions[@]} -eq 1 ]] \
       && grep -q "Seed: $effective_seed" "$log"; then
      for file in "${summaries[0]}" "${actions[0]}"; do
        target="$OUT/paired_20_init/$(basename "$file")"
        test ! -e "$target" || { echo "REFUSE_OVERWRITE=$target" >&2; exit 73; }
        cp -a "$file" "$target"
      done
      success=1
      break
    fi
    echo "ATTEMPT_FAILED init=$init attempt=$attempt exit=$rc effective_seed=$effective_seed" >> "$OUT/attempts/retry_index.log"
    sleep 15
  done
  [[ $success -eq 1 ]] || { echo "PAIRED_INIT_FAILED_AFTER_3_ATTEMPTS init=$init" >&2; exit 74; }
done

"$PY" "$SUMMARY" --eval-root "$OUT" \
  --historical-reference "$REFERENCE" \
  --output "$SUMMARY_OUT" \
  --training-log "$TRAIN_LOG"
echo "TEACHER_BASE_INIT_RECOVERY_EVALUATION_COMPLETE=$OUT"
