"""Opt-in main-thread signal cleanup; cannot protect against SIGKILL."""
import signal
import threading
from bounded_child_v1 import run_bounded


class SupervisorSignal(BaseException):
    def __init__(self, number):
        self.number = number
        super().__init__(f'supervisor received signal {number}')


def run_with_signal_cleanup(*args, **kwargs):
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('signal supervisor must run in main thread')
    signals = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    original = {number: signal.getsignal(number) for number in signals}
    interrupted = []

    def handle(number, frame):
        # Subsequent signals must not interrupt the bounded cleanup wait.
        if not interrupted:
            interrupted.append(number)
            raise SupervisorSignal(number)

    try:
        for number in signals:
            signal.signal(number, handle)
        return run_bounded(*args, **kwargs)
    finally:
        for number, handler in original.items():
            signal.signal(number, handler)
