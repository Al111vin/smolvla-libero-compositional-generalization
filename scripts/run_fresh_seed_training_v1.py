"""One-shot bounded supervisor; caller must detach; no automatic retries."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bounded_child_signals_v3 import run_with_signal_cleanup, SupervisorSignal

def safe_path(path):
    p = Path(path)
    if not p.is_absolute() or any(x.is_symlink() for x in (p, *p.parents)):
        raise ValueError('absolute non-symlink paths required')
    return p

def run(manifest):
    lock = safe_path(manifest['gpu_lock'])
    output = safe_path(manifest['output'])
    prefix = safe_path(manifest['external_prefix'])
    paths = [safe_path(str(prefix)+s) for s in ('.training.log','.launcher.pid','.exit_code')]
    if output == prefix or output in prefix.parents or prefix in output.parents:
        raise ValueError('logs must be independent of output')
    fd = os.open(lock, os.O_RDWR | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError('registered regular lock required')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(os.path.lexists(p) for p in [output,*paths]):
            raise FileExistsError('registered paths already exist')
        for name, expected in manifest['pinned_files'].items():
            if hashlib.sha256(safe_path(name).read_bytes()).hexdigest() != expected:
                raise ValueError('source hash mismatch')
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('GPU conflict')
        rows = subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()[1:]
        for row in rows:
            fields = row.strip().split(None,1)
            if len(fields)==2 and int(fields[0])!=os.getpid() and any(t in fields[1] for t in manifest['conflict_tokens']):
                raise RuntimeError('process conflict')
        if shutil.disk_usage(output.parent).free < manifest['minimum_free_bytes']:
            raise RuntimeError('insufficient disk')
        # Reserve metadata before spawning. Live child identity stays owned by
        # bounded helper; final PID is written after its group has been cleaned.
        with paths[1].open('x') as record:
            json.dump({'supervisor':os.getpid(),'status':'reserved_not_launched'},record)
            record.flush()
            os.fsync(record.fileno())
        with paths[0].open('x') as log:
            try:
                result = run_with_signal_cleanup(manifest['command'], seconds=manifest['walltime_seconds'],
                    grace_seconds=manifest['grace_seconds'],cwd=manifest['cwd'],stdout=log)
            except BaseException as error:
                with paths[2].open('x') as terminal:
                    json.dump({'status':'signal' if isinstance(error,SupervisorSignal) else 'error',
                               'error_type':type(error).__name__},terminal)
                raise
        with paths[2].open('x') as terminal:
            json.dump(result,terminal)
        return result
    finally:
        os.close(fd)

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    args=parser.parse_args()
    result=run(json.loads(Path(args.manifest).read_text()))
    sys.exit(124 if result['timed_out'] else result['returncode'])
