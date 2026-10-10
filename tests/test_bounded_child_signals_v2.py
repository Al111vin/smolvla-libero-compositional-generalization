import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


class SignalCleanupTests(unittest.TestCase):
    def test_term_interrupt_cleans_child_before_return(self):
        scripts = Path(__file__).resolve().parents[1]/'scripts'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); ready = root/'ready'; marker = root/'late'
            child = ('import signal,time; from pathlib import Path; '
                'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                f'Path({str(ready)!r}).touch(); time.sleep(1); Path({str(marker)!r}).touch()')
            supervisor = (f'import sys; sys.path.insert(0,{str(scripts)!r}); '
                'from bounded_child_signals_v2 import run_with_signal_cleanup,SupervisorSignal; '
                'import subprocess\n'
                'try:\n'
                f'    run_with_signal_cleanup([sys.executable,"-c",{child!r}], seconds=5, grace_seconds=0.2, cwd={directory!r}, stdout=subprocess.DEVNULL)\n'
                'except SupervisorSignal as error:\n'
                '    raise SystemExit(128+error.number)\n')
            process = subprocess.Popen([sys.executable,'-c',supervisor])
            try:
                deadline = time.monotonic()+3
                while not ready.exists() and time.monotonic()<deadline:
                    if process.poll() is not None: break
                    time.sleep(0.01)
                self.assertTrue(ready.exists())
                os.kill(process.pid, signal.SIGTERM)
                self.assertEqual(process.wait(timeout=3), 143)
                time.sleep(1)
                self.assertFalse(marker.exists())
            finally:
                if process.poll() is None:
                    process.kill(); process.wait()


if __name__ == '__main__':
    unittest.main()
