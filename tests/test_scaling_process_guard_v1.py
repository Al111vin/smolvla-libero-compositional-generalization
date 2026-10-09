import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_scaling_eval_v1 import scaling_conflicts


class GuardTests(unittest.TestCase):
    def test_new_consumers_detected_without_self_or_inline_false_positive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for pid, argv in {1: ['python', 'run_scaling_eval_v1.py'],
                              2: ['python', 'exposure_matched_train_wrapper_v1.py'],
                              3: ['python', '-c', 'eval_scaling_environment_v1.py'],
                              4: ['python', 'eval_scaling_environment_v1.py']}.items():
                path = root / str(pid)
                path.mkdir()
                (path / 'cmdline').write_bytes(('\0'.join(argv) + '\0').encode())
            self.assertEqual(scaling_conflicts(root, caller_pid=1), [2, 4])


if __name__ == '__main__':
    unittest.main()
