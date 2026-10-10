"""Cooperating-process lease and permanent registration; never launches work."""
import fcntl
import json
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from resume_entry_contract_v1 import require_resume_entry_contract


@contextmanager
def reserve_resume(cfg, accelerator, *, lock_path, receipt, **paths):
    lock_path, receipt = Path(lock_path), Path(receipt)
    for p in (lock_path, receipt):
        if not p.is_absolute() or any(parent.is_symlink() for parent in p.parents):
            raise ValueError('absolute nonredirected registration paths required')
    protected = [Path(p).resolve() for p in paths.values()]
    for p in (lock_path.resolve(), receipt.resolve()):
        if any(p == other or other in p.parents or p in other.parents for other in protected):
            raise ValueError('registration path overlaps evidence/output paths')
    if lock_path.resolve() == receipt.resolve():
        raise ValueError('receipt aliases lock')
    # Require an already registered regular lock file; never unlink/replace it.
    fd = os.open(lock_path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError('regular lock required')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_resume_entry_contract(cfg, accelerator, **paths)
        registration = {'status': 'reserved_not_launched', 'pid': os.getpid(),
                        'paths': {k: str(v) for k, v in paths.items()}}
        receipt_fd = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(receipt_fd, 'w') as stream:
            json.dump(registration, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Retain receipt on success or failure; reuse is prohibited.
        yield registration
    finally:
        os.close(fd)
