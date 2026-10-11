import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import bounded_child_signals_v4 as runner
from unittest.mock import patch

class SpawnReceiptTests(unittest.TestCase):
    def test_callback_receives_owned_pid(self):
        pids=[]
        with tempfile.TemporaryDirectory() as root:
            result=runner.run_with_signal_cleanup([sys.executable,'-c','raise SystemExit(7)'],
                seconds=2,grace_seconds=.1,cwd=root,stdout=subprocess.DEVNULL,on_spawn=pids.append)
        self.assertEqual(pids,[result['pid']])
        self.assertEqual(result['returncode'],7)

    def test_callback_failure_reaps_child(self):
        children=[]
        original=subprocess.Popen
        def spawn(*a,**k):
            child=original(*a,**k)
            children.append(child)
            return child
        def fail(pid):
            raise OSError('receipt failure')
        with tempfile.TemporaryDirectory() as root, patch.object(runner.subprocess,'Popen',side_effect=spawn):
            with self.assertRaises(OSError):
                runner.run_with_signal_cleanup([sys.executable,'-c','import time;time.sleep(20)'],
                    seconds=2,grace_seconds=.1,cwd=root,stdout=subprocess.DEVNULL,on_spawn=fail)
        self.assertIsNotNone(children[0].poll())
