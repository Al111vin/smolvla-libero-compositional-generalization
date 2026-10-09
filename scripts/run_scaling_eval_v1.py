"""One final checkpoint, immutable96 rollouts, no adaptive retries."""
import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from scaling_eval_schedule_v1 import schedule, validate_protocol, validate_rollout, gate
from diagnostic_process_guard_v1 import conflicting_processes
from capture_reset_render_provenance_v1 import EVALUATOR_SHA


def scaling_conflicts(proc_root=Path('/proc'), caller_pid=None):
    import os
    caller_pid = os.getpid() if caller_pid is None else caller_pid
    found = set(conflicting_processes(proc_root, caller_pid))
    targets = {'run_scaling_eval_v1.py', 'eval_scaling_environment_v1.py',
               'exposure_matched_train_wrapper_v1.py', 'run_exposure_matched_training_v1.py'}
    for path in proc_root.iterdir():
        if not path.name.isdigit() or int(path.name) == caller_pid:
            continue
        try:
            argv = path.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
        except FileNotFoundError:
            continue
        if '-c' not in argv[:3] and any(Path(token).name in targets for token in argv):
            found.add(int(path.name))
    return sorted(found)


def unique_csv(target, pattern):
    files = list(target.glob(pattern))
    if len(files) != 1:
        raise ValueError('Missing or ambiguous evidence: ' + pattern)
    with files[0].open() as stream:
        return list(csv.DictReader(stream))


CHECKPOINT_FILES = ('model.safetensors', 'config.json', 'train_config.json',
                    'policy_preprocessor.json', 'policy_postprocessor.json',
                    'policy_preprocessor_step_5_normalizer_processor.safetensors',
                    'policy_postprocessor_step_0_unnormalizer_processor.safetensors')


def validate_checkpoint(checkpoint, model_sha):
    for name in CHECKPOINT_FILES:
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError('Incomplete checkpoint: ' + name)
    if hashlib.sha256((checkpoint / 'model.safetensors').read_bytes()).hexdigest() != model_sha:
        raise ValueError('Checkpoint hash mismatch')


def run(manifest):
    root = Path(manifest['root'])
    base = Path(__file__).parent
    with Path('/root/smolvla-training-prep/teacher_control_gpu.lock').open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if root.exists() or root.is_symlink():
            raise FileExistsError(root)
        for name, sha in manifest['code_hashes'].items():
            if hashlib.sha256((base / name).read_bytes()).hexdigest() != sha:
                raise ValueError('Code hash mismatch: ' + name)
        if hashlib.sha256(Path(manifest['evaluator']).read_bytes()).hexdigest() != EVALUATOR_SHA:
            raise ValueError('Frozen evaluator hash mismatch')
        checkpoint = Path(manifest['checkpoint'])
        validate_checkpoint(checkpoint, manifest['model_sha'])
        if scaling_conflicts():
            raise RuntimeError('Process conflict')
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPU occupied')
        import shutil
        if shutil.disk_usage(root.parent).free < 4_000_000_000:
            raise RuntimeError('Insufficient evaluation disk space')
        root.mkdir()
        code = 1
        records, signatures = [], {}
        try:
            for row in schedule():
                target = root / row['key']
                command = [sys.executable, str(base / 'eval_scaling_environment_v1.py'),
                    '--evaluator', manifest['evaluator'], '--checkpoint', str(checkpoint),
                    '--model-sha', manifest['model_sha'], '--root', str(target), '--key', row['key']]
                with (root / (row['key'] + '.log')).open('x') as stream:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                            timeout=600, cwd='/root/smolvla-eval-prep')
                if result.returncode != 0:
                    raise RuntimeError(f'Rollout failed: {row["key"]}, exit {result.returncode}')
                protocol = json.loads((target / 'protocol.json').read_text())
                validate_protocol(row, protocol)
                if protocol['checkpoint'] != str(checkpoint) or protocol['model_sha'] != manifest['model_sha']:
                    raise ValueError('Model provenance mismatch')
                capture = json.loads((target / 'first_input.json').read_text())
                if not capture['images'] or not capture['model_arrays']:
                    raise ValueError('Missing environment evidence')
                signature = {k: capture[k] for k in ('model_arrays', 'simulator_state',
                    'images', 'frame', 'input', 'processor', 'loaded_model', 'rng')}
                key = (row['task_id'], row['init'])
                if key in signatures and signatures[key] != signature:
                    raise ValueError('Repeated initialization provenance mismatch')
                signatures[key] = signature
                summaries = unique_csv(target, '*_summary.csv')
                if len(summaries) != 1:
                    raise ValueError('Expected exactly one summary row')
                success = validate_rollout(row, summaries[0], unique_csv(target, '*_actions.csv'))
                record = dict(row, success=success, steps=int(summaries[0]['steps']))
                records.append(record)
                with (root / (row['key'] + '.verified.json')).open('x') as stream:
                    json.dump(record, stream, indent=2)
                print('VERIFIED', row['key'], flush=True)
            result = gate({r['key']: r['success'] for r in records})
            with (root / 'completion.json').open('x') as stream:
                json.dump({'records': records, 'gate': result}, stream, indent=2)
            code = 0
        finally:
            with (root / 'exit_code').open('x') as stream:
                stream.write(str(code) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.manifest).read_text()))
