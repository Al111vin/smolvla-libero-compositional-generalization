import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from scripts import run_fresh_seed_training_v1 as runner

class SupervisorTests(unittest.TestCase):
    def fixture(self, root, code):
        (root/'lock').touch()
        return dict(gpu_lock=str(root/'lock'),output=str(root/'output'),
            external_prefix=str(root/'run'),pinned_files={},minimum_free_bytes=1,
            conflict_tokens=['absent-fresh-seed-fixture'],command=[sys.executable,'-c',code],
            cwd=str(root),walltime_seconds=.2,grace_seconds=.1)

    def test_exit_and_permanent_records(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            m=self.fixture(root,'print("fixture");raise SystemExit(7)')
            with patch.object(runner.subprocess,'check_output',side_effect=['','PID ARGS\n']):
                result=runner.run(m)
            self.assertEqual(result['returncode'],7)
            records=[json.loads(line) for line in (root/'run.launcher.pid').read_text().splitlines()]
            self.assertEqual(records[-1]['trainer'],result['pid'])
            self.assertEqual(json.loads((root/'run.exit_code').read_text()),result)
            self.assertFalse((root/'output').exists())
            with self.assertRaises(FileExistsError):
                runner.run(m)

    def test_timeout_reaped_before_terminal(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            m=self.fixture(root,'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(20)')
            with patch.object(runner.subprocess,'check_output',side_effect=['','PID ARGS\n']):
                result=runner.run(m)
            self.assertTrue(result['timed_out'])
            self.assertEqual(result['returncode'],-9)
            self.assertTrue((root/'run.exit_code').is_file())

    def test_gpu_conflict_never_launches(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            m=self.fixture(root,'raise SystemExit(0)')
            with patch.object(runner.subprocess,'check_output',return_value='123'), patch.object(runner,'run_with_signal_cleanup') as child:
                with self.assertRaises(RuntimeError):
                    runner.run(m)
                child.assert_not_called()
            self.assertFalse((root/'run.launcher.pid').exists())
