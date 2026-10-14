#!/usr/bin/env bash
# Bounded supervisor; EVAL_RUNNER is environment-only so argv-based conflict
# checks do not mistake this supervisor for a competing evaluation process.
set -u

if [[ $# -ne 4 ]]; then
  echo "usage: $0 MANIFEST RECEIPT TIMEOUT_MARKER WALLTIME_SECONDS" >&2
  exit 64
fi
: "${EVAL_RUNNER:?EVAL_RUNNER must name the pinned runner}"
: "${PYTHON_BIN:?PYTHON_BIN must name the pinned interpreter}"

manifest=$1
receipt=$2
timeout_marker=$3
walltime=$4
grace=10

for path in "$receipt" "$timeout_marker" "$receipt.tmp.$$"; do
  if [[ -e "$path" || -L "$path" ]]; then
    echo "refusing existing supervisor evidence path: $path" >&2
    exit 73
  fi
done

"$PYTHON_BIN" "$EVAL_RUNNER" --manifest "$manifest" &
child=$!
timed_out=0
start=$SECONDS

while kill -0 "$child" 2>/dev/null; do
  if (( SECONDS - start >= walltime )); then
    timed_out=1
    : > "$timeout_marker"
    kill -TERM "$child" 2>/dev/null || true
    deadline=$((SECONDS + grace))
    while kill -0 "$child" 2>/dev/null && (( SECONDS < deadline )); do
      sleep 1
    done
    kill -KILL "$child" 2>/dev/null || true
    break
  fi
  sleep 1
done

wait "$child"
rc=$?
tmp="$receipt.tmp.$$"
if [[ -e "$receipt" || -L "$receipt" ]]; then
  echo "refusing to overwrite supervisor receipt: $receipt" >&2
  exit 74
fi
timed_out_json=$([[ $timed_out -eq 1 ]] && echo true || echo false)
printf '{"returncode":%d,"timed_out":%s,"walltime_limit_s":%d,"termination_grace_s":%d}\n' \
  "$rc" "$timed_out_json" "$walltime" "$grace" > "$tmp"
if ! ln "$tmp" "$receipt"; then
  echo "failed to create no-clobber supervisor receipt" >&2
  exit 75
fi
rm "$tmp"
exit "$rc"
