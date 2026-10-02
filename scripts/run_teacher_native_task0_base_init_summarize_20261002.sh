#!/usr/bin/env bash
set -euo pipefail
BASE=/root/smolvla-training-prep
EVAL=/root/smolvla-eval-prep
REC="$BASE/recovery/native_task0_20261002"
NAME=teacher_native_task0_base_init_current_recipe_10k_20261002_v1
EVAL_PID=${1:?pass the existing evaluation orchestrator PID}
MARKER="$REC/$NAME.evaluation_complete.marker"
SUMMARY="$EVAL/results/$NAME/summary.json"
PY=/usr/local/miniconda3/envs/py312/bin/python
SUMMARIZER="$REC/summarize_teacher_native_task0_base_init_eval_20261002.py"
REFERENCE="$REC/teacher_native_task0_current_pipeline_10k_eval_summary_20261002.json"
TRAIN_LOG="$REC/$NAME.training.log"

# Wait only for the already-running bounded evaluation; this process uses no GPU.
for attempt in $(seq 1 240); do
  if test -f "$MARKER"; then break; fi
  if ! kill -0 "$EVAL_PID" 2>/dev/null; then
    test -f "$MARKER" && break
    echo 'EVALUATION_ORCHESTRATOR_EXITED_WITHOUT_COMPLETION_MARKER' >&2
    exit 74
  fi
  sleep 30
done
test -f "$MARKER" || { echo 'EVALUATION_SUMMARY_WAIT_TIMEOUT' >&2; exit 74; }
test -f "$SUMMARIZER" && test -f "$REFERENCE"
mkdir -p "$(dirname "$SUMMARY")"
"$PY" "$SUMMARIZER" \
  --eval-root "$EVAL/results/$NAME/eval_v1" \
  --historical-reference "$REFERENCE" \
  --output "$SUMMARY" \
  --training-log "$TRAIN_LOG"
echo "TEACHER_BASE_INIT_SUMMARY_COMPLETE=$SUMMARY"
