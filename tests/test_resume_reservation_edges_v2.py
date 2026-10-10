"""Negative tests for evidence-preserving lease failure paths; no training."""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_resume_reservation_v1 import ReservationTests
from resume_reservation_v1 import reserve_resume


class ReservationEdgeTests(ReservationTests):
    def test_missing_lock_not_created(self):
        lock = self.root / 'missing.lock'
        with self.assertRaises(FileNotFoundError):
            with reserve_resume(self.cfg, self.acc, **{**self.kw, 'lock_path': lock}):
                self.fail('missing lock accepted')
        self.assertFalse(lock.exists())
        self.assertFalse(self.kw['receipt'].exists())

    def test_symlink_lock_target_preserved(self):
        alias = self.root / 'alias.lock'
        alias.symlink_to(self.kw['lock_path'])
        before = self.kw['lock_path'].read_bytes()
        with self.assertRaises(OSError):
            with reserve_resume(self.cfg, self.acc, **{**self.kw, 'lock_path': alias}):
                self.fail('symlink accepted')
        self.assertEqual(before, self.kw['lock_path'].read_bytes())
        self.assertFalse(self.kw['receipt'].exists())

    def test_symlink_receipt_target_preserved(self):
        target = self.root / 'evidence'; target.write_bytes(b'old evidence')
        self.kw['receipt'].symlink_to(target)
        with self.assertRaises(FileExistsError):
            with reserve_resume(self.cfg, self.acc, **self.kw):
                self.fail('receipt symlink accepted')
        self.assertEqual(target.read_bytes(), b'old evidence')

    def test_exception_releases_lock_without_erasing_receipt(self):
        with self.assertRaises(RuntimeError):
            with reserve_resume(self.cfg, self.acc, **self.kw):
                raise RuntimeError('controlled failure')
        receipt = self.kw['receipt'].read_bytes()
        with reserve_resume(self.cfg, self.acc, **{**self.kw, 'receipt': self.root/'next.receipt'}):
            self.assertEqual(receipt, self.kw['receipt'].read_bytes())
        self.assertEqual(os.stat(self.kw['receipt']).st_mode & 0o777, 0o600)
        self.assertFalse(self.paths['output'].exists())


if __name__ == '__main__':
    unittest.main()
