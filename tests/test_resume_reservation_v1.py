import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from resume_reservation_v1 import reserve_resume


class ReservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name).resolve()
        cp = self.root/'old'; cp.mkdir()
        lock = self.root/'gpu.lock'; lock.touch()
        self.paths = dict(checkpoint=cp, output=self.root/'output', log=self.root/'log',
                          pid=self.root/'pid', exit_code=self.root/'exit')
        self.kw = dict(lock_path=lock, receipt=self.root/'receipt', **self.paths)
        self.cfg = NS(resume=True, steps=160000, batch_size=2, num_workers=4, env=None,
            output_dir=self.paths['output'], checkpoint_path=cp,
            policy=NS(pretrained_path=cp/'pretrained_model', push_to_hub=False),
            dataset=NS(streaming=False, image_transforms=NS(enable=False)), wandb=NS(enable=False))
        self.acc = NS(num_processes=1)

    def tearDown(self):
        self.temp.cleanup()

    def test_contention_rejected_no_second_receipt(self):
        with reserve_resume(self.cfg, self.acc, **self.kw):
            with self.assertRaises(BlockingIOError):
                with reserve_resume(self.cfg, self.acc, **{**self.kw, 'receipt': self.root/'second'}):
                    self.fail('contending lease acquired')
        self.assertFalse((self.root/'second').exists())
        self.assertFalse(self.paths['output'].exists())

    def test_failure_receipt_preserved_and_reuse_rejected(self):
        with self.assertRaises(RuntimeError):
            with reserve_resume(self.cfg, self.acc, **self.kw):
                raise RuntimeError('synthetic failure')
        before = self.kw['receipt'].read_bytes()
        with self.assertRaises(FileExistsError):
            with reserve_resume(self.cfg, self.acc, **self.kw):
                pass
        self.assertEqual(before, self.kw['receipt'].read_bytes())

    def test_existing_output_rejected_before_receipt(self):
        self.paths['output'].mkdir()
        with self.assertRaises(FileExistsError):
            with reserve_resume(self.cfg, self.acc, **self.kw):
                pass
        self.assertFalse(self.kw['receipt'].exists())


if __name__ == '__main__':
    unittest.main()
