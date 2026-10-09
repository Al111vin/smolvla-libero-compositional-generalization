"""One-shot supervisor. Caller must detach it; never retries training."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def check_absent(paths):
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)


def run(manifest):
    """Manifest is registered separately; no defaults choose new paths."""
    lock = Path(manifest['gpu_lock'])
    # Lock must already exist; never silently invent a different lock.
    with lock.open('r+') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = Path(manifest['output'])
        prefix = Path(manifest['external_prefix'])
        log, pid, exit_file = [Path(str(prefix) + suffix) for suffix in
                              ('.training.log', '.launcher.pid', '.exit_code')]
        check_absent([output, log, pid, exit_file])
        for name, expected in manifest['pinned_files'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError('Pinned file hash mismatch: ' + name)
        gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                       '--format=csv,noheader'], text=True)
        if gpu.strip():
            raise RuntimeError('GPU compute process conflict')
        # Process guard supplements GPU occupancy (startup can precede allocation).
        processes = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
        import os
        for line in processes.splitlines()[1:]:
            parts = line.strip().split(None, 1)
            if len(parts) != 2 or int(parts[0]) == os.getpid():
                continue
            if any(token in parts[1] for token in manifest['conflict_tokens']):
                raise RuntimeError('Training/evaluation process conflict')
        import shutil
        if shutil.disk_usage(output.parent).free < manifest['minimum_free_bytes']:
            raise RuntimeError('Insufficient disk space')
        with log.open('x') as stream:
            # Reserve PID record before launching: a path race must not leave
            # an unrecorded running trainer.
            pid_stream = pid.open('x')
            try:
                child = subprocess.Popen(manifest['command'], cwd=manifest['cwd'],
                                         stdout=stream, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL)
            except BaseException:
                pid_stream.close()
                with exit_file.open('x') as f:
                    f.write('launch_failed\n')
                raise
            try:
                with pid_stream as f:
                    json.dump({'supervisor': os.getpid(), 'trainer': child.pid}, f)
                    f.flush()
                    os.fsync(f.fileno())
            except BaseException:
                child.terminate()
                code = child.wait()
                with exit_file.open('x') as f:
                    f.write(str(code) + '\n')
                raise
            code = child.wait()
            with exit_file.open('x') as f:
                f.write(str(code) + '\n')
            return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    sys.exit(run(json.loads(Path(args.manifest).read_text())))
