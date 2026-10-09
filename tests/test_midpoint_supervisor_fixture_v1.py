"""Full40 file-backed supervisor simulation; no policy or GPU use."""
import builtins
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_midpoint_eval_v1 as runner
from audit_midpoint_final_evidence_v1 import audit
from midpoint_eval_schedule_v1 import schedule


class MidpointFixture(unittest.TestCase):
    def exercise(self, fault=None):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            lock = base/'lock'; lock.touch()
            checkpoint = base/'checkpoint'; checkpoint.mkdir()
            (checkpoint/'model.safetensors').write_bytes(b'fixture')
            sha = hashlib.sha256(b'fixture').hexdigest()
            evaluator=base/'evaluator.py'; evaluator.write_bytes(b'fixture')
            codes=Path(runner.__file__).parent
            manifest = dict(root=str(base/'results'), code_hashes={n:hashlib.sha256((codes/n).read_bytes()).hexdigest() for n in runner.REQUIRED_CODE}, evaluator=str(evaluator),
                models={k:dict(path=str(checkpoint),sha=sha) for k in ('single20k','joint80k')})
            path = base/'manifest.json'; path.write_text(json.dumps(manifest))
            original_open = builtins.open
            def redirected(file, *args, **kwargs):
                if str(file) == '/root/smolvla-training-prep/teacher_control_gpu.lock':
                    file = lock
                return original_open(file, *args, **kwargs)
            calls = []
            def simulate(command, **kwargs):
                row = schedule()[len(calls)]
                calls.append(row['key'])
                if fault == 'process':
                    return SimpleNamespace(returncode=7)
                target = Path(command[command.index('--root')+1]); target.mkdir()
                protocol = dict(init=row['init'],cli_seed=row['cli_seed'],
                    effective_seed=12345+2*row['init'],environment_seed=12351,
                    wait=10,max_steps=300,action_steps=25,
                    checkpoint=str(checkpoint),model_sha=sha,
                    evaluator_sha='89fd36a89dc45a56382219d4c3f6e5d12a3b1689abb3440ec29be346815af7cb',
                    wrapper_sha=manifest['code_hashes']['eval_controlled_environment_v1.py'])
                if fault == 'protocol': protocol['wait'] = 0
                (target/'protocol.json').write_text(json.dumps(protocol))
                capture = {k:{'fixture':1} for k in ('model_arrays','simulator_state',
                    'images','frame','input','processor','loaded_model','rng')}
                if fault == 'environment' and len(calls) == 2:
                    capture['simulator_state'] = {'fixture':2}
                (target/'first_input.json').write_text(json.dumps(capture))
                summary = dict(suite='libero_spatial',task_id=0,init_source='benchmark',
                    init_index=row['init'],seed=12345+2*row['init'],wait_steps=10,
                    n_action_steps=25,success='True',steps=1,total_reward=1)
                action = dict(step=0,reward=1)
                action.update({f'{p}_{j}':0 for p in ('raw_action','processed_action','applied_action') for j in range(7)})
                action.update({f'state_{j}':0 for j in range(15)})
                for name,value in [('episode_summary.csv',summary),('episode_actions.csv',action)]:
                    with (target/name).open('w') as f:
                        writer=csv.DictWriter(f,fieldnames=list(value));writer.writeheader();writer.writerow(value)
                return SimpleNamespace(returncode=0)
            with patch.object(sys,'argv',['runner','--manifest',str(path)]), \
                 patch.object(runner,'EVALUATOR_SHA',sha), \
                 patch.object(runner.time,'monotonic',side_effect=[0,5401] if fault=='wall' else None,return_value=0), \
                 patch('builtins.open',side_effect=redirected), \
                 patch.object(runner,'conflicting_processes',return_value=[]), \
                 patch.object(runner.subprocess,'check_output',return_value=''), \
                 patch.object(runner.subprocess,'run',side_effect=simulate), \
                 patch.object(runner.os,'statvfs',return_value=SimpleNamespace(f_bavail=10**10,f_frsize=1)):
                if fault:
                    with self.assertRaises(AssertionError): runner.main()
                else: runner.main()
            root = base/'results'
            self.assertEqual((root/'exit_code').read_text().strip(), '1' if fault else '0')
            self.assertEqual(len(calls), (0 if fault=='wall' else 2 if fault=='environment' else 1) if fault else 40)
            self.assertEqual((root/'completion.json').exists(),not bool(fault))
            if not fault:
                self.assertEqual(len(json.loads((root/'completion.json').read_text())),40)
                self.assertEqual(audit(root,manifest)['successes'],{'single20k':20,'joint80k':20})
                verified=root/(schedule()[0]['key']+'.verified.json')
                forged=json.loads(verified.read_text());forged['success']=False
                verified.write_text(json.dumps(forged))
                with self.assertRaisesRegex(ValueError,'raw CSV'): audit(root,manifest)
            else:
                with self.assertRaisesRegex(ValueError,'incomplete or failed'): audit(root,manifest)

    def test_full40(self): self.exercise()
    def test_subprocess_failure(self): self.exercise('process')
    def test_protocol_failure(self): self.exercise('protocol')
    def test_environment_mismatch(self): self.exercise('environment')
    def test_wall_budget_stops_before_inference(self): self.exercise('wall')


if __name__ == '__main__': unittest.main()
