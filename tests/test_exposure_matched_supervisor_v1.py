import hashlib
from pathlib import Path
import tempfile
import sys
import json
import unittest
from unittest.mock import patch
from scripts.run_exposure_matched_training_v1 import run, check_absent


class SupervisorTests(unittest.TestCase):
    def test_real_cpu_child_lifecycle_and_no_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lock = root / 'lock'
            lock.touch()
            manifest = dict(gpu_lock=str(lock), output=str(root / 'output'),
                            external_prefix=str(root / 'run'), pinned_files={},
                            minimum_free_bytes=1, conflict_tokens=['unique-absent-fixture'],
                            command=[sys.executable, '-c', 'print("cpu lifecycle fixture")'], cwd=temp)
            with patch('scripts.run_exposure_matched_training_v1.subprocess.check_output',
                       side_effect=['', 'PID ARGS\n']):
                self.assertEqual(run(manifest), 0)
            self.assertEqual((root / 'run.exit_code').read_text().strip(), '0')
            record = json.loads((root / 'run.launcher.pid').read_text())
            self.assertGreater(record['trainer'], 0)
            self.assertIn('cpu lifecycle fixture', (root / 'run.training.log').read_text())
            self.assertFalse((root / 'output').exists())
            with patch('scripts.run_exposure_matched_training_v1.subprocess.Popen') as launch:
                with self.assertRaises(FileExistsError):
                    run(manifest)
                launch.assert_not_called()

    def test_launch_error_preserves_scene(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lock = root / 'lock'
            lock.touch()
            manifest = dict(gpu_lock=str(lock), output=str(root / 'output'),
                            external_prefix=str(root / 'run'), pinned_files={},
                            minimum_free_bytes=1, conflict_tokens=['unique-absent-fixture'],
                            command=[str(root / 'missing-executable')], cwd=temp)
            with patch('scripts.run_exposure_matched_training_v1.subprocess.check_output',
                       side_effect=['', 'PID ARGS\n']):
                with self.assertRaises(FileNotFoundError):
                    run(manifest)
            self.assertEqual((root / 'run.exit_code').read_text().strip(), 'launch_failed')
            self.assertTrue((root / 'run.training.log').exists())
            self.assertTrue((root / 'run.launcher.pid').exists())
            self.assertFalse((root / 'output').exists())

    def test_symlink_is_not_absent(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'link'
            path.symlink_to(Path(temp) / 'missing')
            with self.assertRaises(FileExistsError):
                check_absent([path])

    def test_preflight_failures_never_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lock = root / 'lock'
            lock.touch()
            source = root / 'source'
            source.write_bytes(b'fixture')
            manifest = dict(gpu_lock=str(lock), output=str(root / 'output'),
                            external_prefix=str(root / 'run'),
                            pinned_files={str(source): hashlib.sha256(b'fixture').hexdigest()},
                            minimum_free_bytes=1, conflict_tokens=['conflicting-trainer'],
                            command=['never-executed'], cwd=temp)
            with patch('scripts.run_exposure_matched_training_v1.subprocess.Popen') as launch:
                for outputs in (['123\n'], ['', 'PID ARGS\n123 conflicting-trainer\n']):
                    with patch('scripts.run_exposure_matched_training_v1.subprocess.check_output',
                               side_effect=outputs):
                        with self.assertRaises(RuntimeError):
                            run(manifest)
                bad = dict(manifest, pinned_files={str(source): 'wrong'})
                with self.assertRaises(ValueError):
                    run(bad)
                launch.assert_not_called()
            self.assertFalse((root / 'output').exists())
            self.assertFalse((root / 'run.training.log').exists())


if __name__ == '__main__':
    unittest.main()
