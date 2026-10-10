"""POSIX child-group walltime bound; not a training launcher or budget approval."""
import math
import os
import signal
import subprocess


def run_bounded(command, *, seconds, grace_seconds, cwd, stdout):
    for value in (seconds, grace_seconds):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('finite positive walltime and grace required')
    if not command or isinstance(command, (str, bytes)):
        raise ValueError('explicit argument list required')
    child = subprocess.Popen(command, cwd=cwd, stdout=stdout,
        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)

    def stop_group():
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            pass
        # The direct child may have exited while descendants ignore TERM.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()

    try:
        code = child.wait(timeout=seconds)
        # No descendant may retain a rented-resource process after direct exit.
        stop_group()
        return {'pid': child.pid, 'returncode': code, 'timed_out': False}
    except subprocess.TimeoutExpired:
        stop_group()
        return {'pid': child.pid, 'returncode': child.returncode, 'timed_out': True}
    except BaseException:
        stop_group()
        raise
