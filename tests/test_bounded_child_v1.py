import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.bounded_child_v1 import run_bounded


class BoundedChildTests(unittest.TestCase):
    def exercise(self, code, seconds=2):
        with tempfile.TemporaryDirectory() as directory:
            with (Path(directory)/'fixture.log').open('x') as stream:
                return run_bounded([sys.executable, '-c', code], seconds=seconds,
                    grace_seconds=0.2, cwd=directory, stdout=stream)

    def test_success_and_nonzero_not_retried(self):
        self.assertEqual(self.exercise('print("done")')['returncode'], 0)
        self.assertEqual(self.exercise('raise SystemExit(7)')['returncode'], 7)

    def test_timeout_kills_term_ignoring_child(self):
        start = time.monotonic()
        result = self.exercise('import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(20)', 0.3)
        self.assertTrue(result['timed_out'])
        self.assertEqual(result['returncode'], -9)
        self.assertLess(time.monotonic()-start, 3)

    def test_invalid_bounds_never_launch(self):
        with patch('scripts.bounded_child_v1.subprocess.Popen') as launch:
            for seconds in (0, -1, True, float('inf'), float('nan')):
                with self.assertRaises(ValueError):
                    run_bounded(['never'], seconds=seconds, grace_seconds=1, cwd='/', stdout=subprocess.DEVNULL)
            launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
