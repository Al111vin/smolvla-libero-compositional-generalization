#!/usr/bin/env bash
# One-shot, locked training supervisor. No automatic retry; timeout kills the
# trainer process group and preserves an exit receipt in the recovery folder.
set -u

here=$(cd -- "$(dirname -- "$0")" && pwd)
run_id=teacher_native_spatial_task1_freshseed2001_40k_batch2_v1_20261014
config="$here/../configs/$run_id.json"
wrapper="$here/../scripts/teacher_native_spatial_task1_freshseed2001_train_wrapper_v1_20261014.py"
recovery="$(dirname -- "$here")"
repo=/root/smolvla-training-prep
python_bin=/usr/local/miniconda3/envs/py312/bin/python
lock="$repo/teacher_control_gpu.lock"
log="$recovery/$run_id.training.log"
trainer_pid_file="$recovery/$run_id.trainer.pid"
receipt="$recovery/$run_id.exit_code"
timeout_marker="$recovery/$run_id.timed_out"
supervisor_log="$recovery/$run_id.supervisor.log"
walltime=21600
grace=30

for path in "$config" "$wrapper" "$lock"; do
  [[ -f "$path" && ! -L "$path" ]] || { echo "required regular file absent: $path" >&2; exit 66; }
done
for path in "$log" "$trainer_pid_file" "$receipt" "$timeout_marker" \
            "$supervisor_log.pid"; do
  [[ ! -e "$path" && ! -L "$path" ]] || { echo "refusing existing evidence path: $path" >&2; exit 73; }
done

exec 9<>"$lock"
flock -n -x 9 || { echo "production GPU lock is busy" >&2; exit 75; }
printf '%s\n' "$$" > "$supervisor_log.pid"
setsid "$python_bin" "$wrapper" --config_path="$config" >>"$log" 2>&1 &
child=$!
( set -o noclobber; printf '%s\n' "$child" > "$trainer_pid_file" ) || exit 76
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
( set -o noclobber; printf '{"pid":%d,"returncode":%d,"timed_out":%s,"walltime_limit_s":%d}\n' \
    "$child" "$rc" "$([[ $timed_out -eq 1 ]] && echo true || echo false)" "$walltime" > "$tmp" ) || exit 77
if ! ln "$tmp" "$receipt"; then echo "receipt already exists; preserving both" >&2; exit 78; fi
rm "$tmp"
exit "$rc"
