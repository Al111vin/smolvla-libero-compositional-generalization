import unittest
from types import SimpleNamespace
from unittest.mock import patch
from scripts import fresh_seed_joint_train_wrapper_v1 as wrapper


class Values:
    def __init__(self, task):
        self.task = task
    def reshape(self, *args):
        return self
    def tolist(self):
        return [self.task, self.task]


def original_update(train_metrics, policy, batch, optimizer, grad_clip_norm,
                    accelerator, lr_scheduler=None, lock=None, rabc_weights_provider=None):
    optimizer.calls += 1
    return train_metrics, {}


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.trainer = SimpleNamespace(make_optimizer_and_scheduler=lambda *a: None,
                                       update_policy=original_update)
        with patch.object(wrapper, "install_homogeneous_loader_patch", return_value=None):
            wrapper.install(self.trainer)
        self.optimizer = SimpleNamespace(calls=0, step_was_skipped=False)
        self.accelerator = SimpleNamespace(num_processes=1, gradient_accumulation_steps=1)

    def call(self, task):
        return self.trainer.update_policy(None, None, {"task_index": Values(task)},
                                         self.optimizer, 1, accelerator=self.accelerator)

    def test_two_balanced_blocks(self):
        for task in [3, 1, 0, 2, 0, 2, 3, 1]:
            self.call(task)
        self.assertEqual(self.optimizer.calls, 8)

    def test_duplicate_rejected_before_update(self):
        self.call(0)
        with self.assertRaises(ValueError):
            self.call(0)
        self.assertEqual(self.optimizer.calls, 1)

    def test_skipped_update_stops(self):
        self.optimizer.step_was_skipped = True
        with self.assertRaises(RuntimeError):
            self.call(0)

    def test_distributed_rejected(self):
        self.accelerator.num_processes = 2
        with self.assertRaises(ValueError):
            self.call(0)
        self.assertEqual(self.optimizer.calls, 0)

    def test_full_config_guard_runs_before_factory(self):
        with patch.object(wrapper, "validate_registered_config", side_effect=ValueError("changed")) as guard:
            with self.assertRaises(ValueError):
                self.trainer.make_optimizer_and_scheduler(None, None)
            guard.assert_called_once_with(None)


if __name__ == "__main__":
    unittest.main()
