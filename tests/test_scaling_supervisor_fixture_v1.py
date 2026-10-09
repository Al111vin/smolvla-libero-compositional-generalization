"""File-backed96-row interface simulation, never a policy evaluation."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_scaling_eval_v1 as runner
import audit_scaling_final_evidence_v1 as final_audit
from scaling_eval_schedule_v1 import schedule


class SupervisorFixture(unittest.TestCase):
    def exercise(self, fail_first=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lock = root / 'lock'
            lock.touch()
            checkpoint = root / 'checkpoint'
            checkpoint.mkdir()
            for name in runner.CHECKPOINT_FILES:
                (checkpoint / name).write_bytes(b'fixture')
            evaluator = root / 'evaluator.py'
            evaluator.write_bytes(b'fixture')
            sha = hashlib.sha256(b'fixture').hexdigest()
            base = Path(runner.__file__).parent
            manifest = dict(root=str(root / 'results'), checkpoint=str(checkpoint),
                            model_sha=sha, evaluator=str(evaluator),
                            code_hashes={n: hashlib.sha256((base / n).read_bytes()).hexdigest()
                                         for n in runner.REQUIRED_CODE})
            original_open = Path.open
            def open_path(path, *args, **kwargs):
                if str(path) == '/root/smolvla-training-prep/teacher_control_gpu.lock':
                    path = lock
                return original_open(path, *args, **kwargs)
            calls = []
            rows = {r['key']: r for r in schedule()}
            def simulate(command, **kwargs):
                key = command[command.index('--key') + 1]
                calls.append(key)
                if fail_first:
                    return SimpleNamespace(returncode=7)
                target = Path(command[command.index('--root') + 1])
                target.mkdir()
                row = rows[key]
                (target / 'protocol.json').write_text(json.dumps(dict(row,
                    checkpoint=str(checkpoint), model_sha=sha, evaluator_sha=sha,
                    wrapper_sha=manifest['code_hashes']['eval_scaling_environment_v1.py'])))
                (target / 'first_input.json').write_text(json.dumps({k: {'fixture': 1}
                    for k in ('model_arrays', 'simulator_state', 'images', 'frame',
                              'input', 'processor', 'loaded_model', 'rng')}))
                summary = dict(suite='libero_spatial', task_id=row['task_id'],
                    init_source='benchmark', init_index=row['init'], seed=row['effective_seed'],
                    wait_steps=10, n_action_steps=25, success='True', steps=1, total_reward=1)
                action = dict(step=0, reward=1)
                action.update({f'{p}_{j}': 0 for p in
                    ('raw_action', 'processed_action', 'applied_action') for j in range(7)})
                action.update({f'state_{j}': 0 for j in range(15)})
                for name, values in [('episode_summary.csv', summary), ('episode_actions.csv', action)]:
                    with (target / name).open('w') as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(values))
                        writer.writeheader(); writer.writerow(values)
                return SimpleNamespace(returncode=0)
            with patch.object(Path, 'open', open_path), patch.object(runner, 'EVALUATOR_SHA', sha), \
                 patch.object(runner, 'scaling_conflicts', return_value=[]), \
                 patch.object(runner.subprocess, 'check_output', return_value=''), \
                 patch.object(runner.subprocess, 'run', side_effect=simulate):
                if fail_first:
                    with self.assertRaises(RuntimeError):
                        runner.run(manifest)
                else:
                    runner.run(manifest)
            result_root = root / 'results'
            self.assertEqual((result_root / 'exit_code').read_text().strip(), '1' if fail_first else '0')
            self.assertEqual(len(calls), 1 if fail_first else 96)
            if fail_first:
                self.assertFalse((result_root / 'completion.json').exists())
                with self.assertRaises(ValueError):
                    final_audit.audit(result_root, manifest)
            else:
                completion = json.loads((result_root / 'completion.json').read_text())
                self.assertEqual(len(completion['records']), 96)
                self.assertTrue(completion['gate']['passed'])
                self.assertFalse(completion['gate']['automatic_next_stage'])
                self.assertEqual(len(list(result_root.glob('*.verified.json'))), 96)
                with patch.object(final_audit, 'EVALUATOR_SHA', sha):
                    audited = final_audit.audit(result_root, manifest)
                    self.assertEqual(audited['gate'], completion['gate'])
                    self.assertEqual(len(audited['records']), 96)
                    self.assertTrue(all(audited['init3_action_repeat_identical'].values()))
                    first = result_root / (schedule()[0]['key'] + '.verified.json')
                    forged = json.loads(first.read_text())
                    forged['success'] = False
                    first.write_text(json.dumps(forged))
                    with self.assertRaisesRegex(ValueError, 'raw CSV'):
                        final_audit.audit(result_root, manifest)

    def test_all96_file_backed_interface_rows(self):
        self.exercise()

    def test_first_failure_stops_without_retry(self):
        self.exercise(fail_first=True)


if __name__ == '__main__':
    unittest.main()
