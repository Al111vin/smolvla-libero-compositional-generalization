import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from resume_lr_guard_v1 import apply_post_restore_factor


class GuardTests(unittest.TestCase):
    def objects(self):
        opt = SimpleNamespace(param_groups=[{"lr": 0.00005125}], state={"moments": 17})
        sch = SimpleNamespace(last_epoch=20000, lr_lambdas=[lambda k: k / 40000],
                              _last_lr=[0.00005125], base_lrs=[0.0001])
        sch.get_last_lr = lambda: sch._last_lr
        sch.optimizer = opt
        return opt, sch

    def apply(self, opt, sch, **overrides):
        args = dict(restored_step=20000, expected_step=20000,
                    expected_lrs=[0.00005125], factor=0.5)
        args.update(overrides)
        return apply_post_restore_factor(opt, sch, **args)

    def test_scales_current_and_future_not_moments(self):
        opt, sch = self.objects()
        self.assertEqual(self.apply(opt, sch), [0.000025625])
        self.assertEqual(sch.lr_lambdas[0](30000), 0.375)
        self.assertEqual(opt.state, {"moments": 17})
        self.assertEqual(sch.base_lrs, [0.0001])

    def test_mismatch_fails_before_mutation(self):
        opt, sch = self.objects()
        opt.param_groups[0]["lr"] = 0.1
        with self.assertRaises(ValueError): self.apply(opt, sch)
        self.assertEqual(sch.lr_lambdas[0](30000), 0.75)

    def test_wrong_step_and_repeated_application_rejected(self):
        opt, sch = self.objects()
        with self.assertRaises(ValueError): self.apply(opt, sch, restored_step=19999)
        self.apply(opt, sch)
        with self.assertRaises(ValueError): self.apply(opt, sch)

    def test_invalid_inputs_fail_without_mutation(self):
        for override in ({"factor": 0.25}, {"expected_lrs": [float("nan")]},
                         {"expected_lrs": []}, {"expected_lrs": [0.0]}):
            opt, sch = self.objects()
            with self.assertRaises(ValueError): self.apply(opt, sch, **override)
            self.assertEqual(opt.param_groups[0]["lr"], 0.00005125)
            self.assertEqual(sch.lr_lambdas[0](30000), 0.75)

    def test_wrong_optimizer_and_scheduler_step_rejected(self):
        opt, sch = self.objects()
        sch.optimizer = object()
        with self.assertRaises(ValueError): self.apply(opt, sch)
        sch.optimizer = opt
        sch.last_epoch = 19999
        with self.assertRaises(ValueError): self.apply(opt, sch)

    def test_original_lr_control_is_unchanged(self):
        opt, sch = self.objects()
        self.apply(opt, sch, factor=1.0)
        self.assertEqual(opt.param_groups[0]["lr"], 0.00005125)
        self.assertEqual(sch.lr_lambdas[0](30000), 0.75)


if __name__ == "__main__": unittest.main()
