import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import bounded_child_signals_v3 as runner


class StartupSignalTests(unittest.TestCase):
    def test_signal_after_spawn_before_handle_return(self):
        original = subprocess.Popen
        spawned = []
        def interrupted_spawn(*args, **kwargs):
            child = original(*args, **kwargs)
            spawned.append(child)
            os.kill(os.getpid(), signal.SIGTERM)
            return child
        old = signal.getsignal(signal.SIGTERM)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner.subprocess, 'Popen', side_effect=interrupted_spawn):
                with self.assertRaises(runner.SupervisorSignal) as caught:
                    runner.run_with_signal_cleanup([sys.executable,'-c','import time; time.sleep(20)'],
                        seconds=2, grace_seconds=0.1, cwd=directory, stdout=subprocess.DEVNULL)
            self.assertEqual(caught.exception.number, signal.SIGTERM)
            self.assertIsNotNone(spawned[0].poll())
            self.assertEqual(signal.getsignal(signal.SIGTERM), old)

    def test_normal_and_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            for code, timeout in [('raise SystemExit(7)', False), ('import time; time.sleep(20)', True)]:
                result = runner.run_with_signal_cleanup([sys.executable,'-c',code],
                    seconds=0.2, grace_seconds=0.1, cwd=directory, stdout=subprocess.DEVNULL)
                self.assertEqual(result['timed_out'], timeout)
                if not timeout:
                    self.assertEqual(result['returncode'], 7)


if __name__ == '__main__':
    unittest.main()
