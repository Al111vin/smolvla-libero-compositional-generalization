import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import torch
except ImportError:
    torch = None
from resume_training_hooks_v1 import registered_resume_hooks


@unittest.skipIf(torch is None, "requires PyTorch; run on registered CPU runtime")
class HooksTests(unittest.TestCase):
    def setUp(self):
        self.cfg = NS(resume=True, steps=40000, batch_size=2, num_workers=4, env=None,
            dataset=NS(repo_id="probe", streaming=False, image_transforms=NS(enable=False)),
            policy=NS(push_to_hub=False))
        self.trainer = NS(make_optimizer_and_scheduler=lambda *a: None,
            load_training_state=lambda *a: None, cycle=lambda x: iter(x),
            update_policy=lambda *a, **k: k["lr_scheduler"].step())

    def test_exception_restores_all_hooks(self):
        saved = dict(vars(self.trainer))
        init = torch.utils.data.DataLoader.__init__
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with registered_resume_hooks(self.trainer, self.cfg, factor=0.5):
                raise RuntimeError("sentinel")
        self.assertEqual(vars(self.trainer), saved)
        self.assertIs(torch.utils.data.DataLoader.__init__, init)

    def test_skipped_update_cannot_advance_scheduler(self):
        calls = []
        scheduler = NS(step=lambda: calls.append(True))
        with registered_resume_hooks(self.trainer, self.cfg, factor=1.0):
            with self.assertRaisesRegex(RuntimeError, "optimizer update skipped"):
                self.trainer.update_policy(None, None, None, NS(step_was_skipped=True),
                                           lr_scheduler=scheduler)
        self.assertEqual(calls, [])

    def test_executed_update_advances_scheduler_once(self):
        calls = []
        with registered_resume_hooks(self.trainer, self.cfg, factor=1.0):
            self.trainer.update_policy(None, None, None, NS(step_was_skipped=False),
                                      lr_scheduler=NS(step=lambda: calls.append(True)))
        self.assertEqual(calls, [True])

    def test_bad_worker_configuration_rejected_before_dataset_access(self):
        with registered_resume_hooks(self.trainer, self.cfg, factor=0.5):
            with self.assertRaisesRegex(ValueError, "worker or sampler"):
                torch.utils.data.DataLoader(NS(repo_id="probe"), batch_size=2,
                                            num_workers=0, prefetch_factor=None)


if __name__ == "__main__":
    unittest.main()
