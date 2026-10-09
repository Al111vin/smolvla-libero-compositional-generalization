import unittest
import tempfile
from pathlib import Path
from scripts.diagnostic_process_guard_v1 import is_conflicting_argv, conflicting_processes


class GuardTests(unittest.TestCase):
    def test_actual_workloads(self):
        self.assertTrue(is_conflicting_argv(["python", "/tmp/diagnose_first_decision_v1.py"]))
        self.assertTrue(is_conflicting_argv(["python", "-m", "scripts.lerobot_train_loco_compat"]))

    def test_inline_source_is_not_script_execution(self):
        self.assertFalse(is_conflicting_argv(["python", "-c", "s='diagnose_first_decision_v1.py'"]))
        self.assertFalse(is_conflicting_argv(["bash", "-c", "grep diagnose_first_decision_v1.py"]))
        self.assertTrue(is_conflicting_argv(["python", "/tmp/run_first_decision_diagnostic_v1.py"]))

    def test_only_caller_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for pid in (101, 102):
                (root / str(pid)).mkdir()
                (root / str(pid) / "cmdline").write_bytes(
                    b"python\0/tmp/run_first_decision_diagnostic_v1.py\0--evaluator\0/tmp/eval_v3_task0_state_capture_v1.py\0")
            self.assertEqual(conflicting_processes(root, caller_pid=101), [102])


if __name__ == "__main__":
    unittest.main()
