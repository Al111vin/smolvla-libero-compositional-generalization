import unittest
from scripts.diagnostic_process_guard_v1 import is_conflicting_argv


class GuardTests(unittest.TestCase):
    def test_actual_workloads(self):
        self.assertTrue(is_conflicting_argv(["python", "/tmp/diagnose_first_decision_v1.py"]))
        self.assertTrue(is_conflicting_argv(["python", "-m", "scripts.lerobot_train_loco_compat"]))

    def test_inline_source_is_not_script_execution(self):
        self.assertFalse(is_conflicting_argv(["python", "-c", "s='diagnose_first_decision_v1.py'"]))
        self.assertFalse(is_conflicting_argv(["bash", "-c", "grep diagnose_first_decision_v1.py"]))
        self.assertFalse(is_conflicting_argv(["python", "/tmp/run_first_decision_diagnostic_v1.py"]))


if __name__ == "__main__":
    unittest.main()
