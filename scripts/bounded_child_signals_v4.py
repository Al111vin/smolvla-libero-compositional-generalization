"""Signal-safe ownership handoff: handlers record, never interrupt Popen."""
import math
import os
import signal
import subprocess
import threading
import time


class SupervisorSignal(BaseException):
    def __init__(self, number):
        self.number = number
        super().__init__(f'supervisor received signal {number}')


def run_with_signal_cleanup(command, *, seconds, grace_seconds, cwd, stdout, on_spawn=None):
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('main thread required')
    for value in (seconds, grace_seconds):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('finite positive bounds required')
    if not command or isinstance(command, (str, bytes)):
        raise ValueError('explicit arguments required')
    signals = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    original = {number: signal.getsignal(number) for number in signals}
    interrupted = []
    child = None

    def handle(number, frame):
        if not interrupted:
            interrupted.append(number)

    def cleanup():
        if child is None:
            return
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()

    try:
        for number in signals:
            signal.signal(number, handle)
        start = time.monotonic()
        if interrupted:
            raise SupervisorSignal(interrupted[0])
        child = subprocess.Popen(command, cwd=cwd, stdout=stdout,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        if on_spawn is not None:
            on_spawn(child.pid)
        # Signals during Popen cannot prevent assignment of the owned handle.
        while True:
            if interrupted:
                raise SupervisorSignal(interrupted[0])
            code = child.poll()
            if code is not None:
                return {'pid': child.pid, 'returncode': code, 'timed_out': False}
            remaining = seconds - (time.monotonic()-start)
            if remaining <= 0:
                cleanup()
                return {'pid': child.pid, 'returncode': child.returncode, 'timed_out': True}
            try:
                child.wait(timeout=min(remaining, 0.05))
            except subprocess.TimeoutExpired:
                pass
    finally:
        try:
            cleanup()
        finally:
            for number, handler in original.items():
                signal.signal(number, handler)
