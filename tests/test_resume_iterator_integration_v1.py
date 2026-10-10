import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from resume_iterator_integration_v1 import guarded_resume_iterator


class IntegrationTests(unittest.TestCase):
    def trainer(self, step=20000):
        return SimpleNamespace(load_training_state=lambda *a, **k: (
            step, object(), SimpleNamespace(last_epoch=step)), cycle=lambda x: iter(x))

    def test_verified_restore_then_finite_budget(self):
        trainer = self.trainer()
        restore, cycle = trainer.load_training_state, trainer.cycle
        with guarded_resume_iterator(trainer, expected_step=20000, total_steps=40000):
            trainer.load_training_state(None)
            iterator = trainer.cycle(range(20000))
            with self.assertRaisesRegex(RuntimeError, "cannot be recreated"):
                trainer.cycle(range(20000))
            for i in range(20000):
                self.assertEqual(next(iterator), i)
            with self.assertRaisesRegex(RuntimeError, "cycling forbidden"):
                next(iterator)
            with self.assertRaisesRegex(RuntimeError, "more than once"):
                trainer.load_training_state(None)
        self.assertIs(trainer.load_training_state, restore)
        self.assertIs(trainer.cycle, cycle)

    def test_restore_required_and_exception_restores_hooks(self):
        trainer = self.trainer()
        restore, cycle = trainer.load_training_state, trainer.cycle
        with self.assertRaisesRegex(RuntimeError, "before verified restore"):
            with guarded_resume_iterator(trainer, expected_step=20000, total_steps=40000):
                trainer.cycle(range(20000))
        self.assertIs(trainer.load_training_state, restore)
        self.assertIs(trainer.cycle, cycle)

    def test_wrong_restore_and_unregistered_budget_rejected(self):
        trainer = self.trainer(0)
        with guarded_resume_iterator(trainer, expected_step=20000, total_steps=40000):
            with self.assertRaisesRegex(RuntimeError, "boundary mismatch"):
                trainer.load_training_state(None)
            with self.assertRaisesRegex(RuntimeError, "before verified restore"):
                trainer.cycle(range(20000))
        with self.assertRaises(ValueError):
            with guarded_resume_iterator(trainer, expected_step=0, total_steps=40000):
                pass


if __name__ == "__main__":
    unittest.main()
