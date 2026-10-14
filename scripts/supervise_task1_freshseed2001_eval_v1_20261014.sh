#!/usr/bin/env bash
# Six-hour, one-shot evaluation supervisor; terminate the complete rollout tree
# on timeout and preserve a no-clobber supervisor receipt.
set -u

if [[ $# -ne 4 ]]; then
  echo "usage: $0 MANIFEST RECEIPT TIMEOUT_MARKER WALLTIME_SECONDS" >&2
  exit 64
fi
: "${EVAL_RUNNER:?EVAL_RUNNER must name the pinned task1 runner}"
: "${PYTHON_BIN:?PYTHON_BIN must name the registered interpreter}"
manifest=$1
receipt=$2
timeout_marker=$3
walltime=$4
grace=30
runner_pid_file="${receipt%.exit_code}.runner.pid"

for path in "$receipt" "$timeout_marker" "$runner_pid_file" "$receipt.tmp.$$"; do
  if [[ -e "$path" || -L "$path" ]]; then
    echo "refusing existing supervisor evidence path: $path" >&2
    exit 73
  fi
done

setsid "$PYTHON_BIN" "$EVAL_RUNNER" --manifest "$manifest" &
child=$!
( set -o noclobber; printf '%s\n' "$child" > "$runner_pid_file" ) || exit 76
timed_out=0
start=$SECONDS
while kill -0 "$child" 2>/dev/null; do
  if (( SECONDS - start >= walltime )); then
    timed_out=1
    ( set -o noclobber; : > "$timeout_marker" ) || true
    kill -TERM -- "-$child" 2>/dev/null || kill -TERM "$child" 2>/dev/null || true
    deadline=$((SECONDS + grace))
    while kill -0 "$child" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
    kill -KILL -- "-$child" 2>/dev/null || kill -KILL "$child" 2>/dev/null || true
    break
  fi
  sleep 2
done
wait "$child"
rc=$?
tmp="$receipt.tmp.$$"
( set -o noclobber; printf '{"pid":%d,"returncode":%d,"timed_out":%s,"walltime_limit_s":%d,"termination_grace_s":%d}\n' \
    "$child" "$rc" "$([[ $timed_out -eq 1 ]] && echo true || echo false)" "$walltime" "$grace" > "$tmp" ) || exit 77
if ! ln "$tmp" "$receipt"; then
  echo "refusing to overwrite supervisor receipt: $receipt" >&2
  exit 78
fi
rm "$tmp"
exit "$rc"
