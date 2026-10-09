import sys
from pathlib import Path
import tempfile
import unittest
import hashlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_scaling_eval_v1 import scaling_conflicts, unique_csv, validate_checkpoint, CHECKPOINT_FILES


class GuardTests(unittest.TestCase):
    def test_each_checkpoint_file_required_nonempty_and_model_pinned(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sha = hashlib.sha256(b'fixture').hexdigest()
            for name in CHECKPOINT_FILES:
                (root / name).write_bytes(b'fixture')
            validate_checkpoint(root, sha)
            for name in CHECKPOINT_FILES:
                path = root / name
                path.unlink()
                with self.assertRaises(ValueError):
                    validate_checkpoint(root, sha)
                path.touch()
                with self.assertRaises(ValueError):
                    validate_checkpoint(root, sha)
                path.write_bytes(b'fixture')
            with self.assertRaises(ValueError):
                validate_checkpoint(root, 'wrong')

    def test_csv_requires_exactly_one_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(ValueError):
                unique_csv(root, '*_summary.csv')
            (root / 'one_summary.csv').write_text('success,steps\nTrue,1\n')
            self.assertEqual(unique_csv(root, '*_summary.csv'),
                             [{'success': 'True', 'steps': '1'}])
            (root / 'two_summary.csv').write_text('success,steps\nFalse,1\n')
            with self.assertRaises(ValueError):
                unique_csv(root, '*_summary.csv')

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
