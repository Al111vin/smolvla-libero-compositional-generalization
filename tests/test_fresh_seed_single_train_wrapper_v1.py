import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from scripts import fresh_seed_single_train_wrapper_v1 as wrapper

class SingleEntryTests(unittest.TestCase):
    def test_native_factory_preserved(self):
        original = Mock(return_value=('optimizer', 'scheduler'))
        trainer = SimpleNamespace(make_optimizer_and_scheduler=original)
        self.assertIs(wrapper.install(trainer), original)
        cfg = SimpleNamespace(steps=40000, seed=2001, resume=False, env=None)
        with patch.object(wrapper, 'validate_registered_config') as guard:
            self.assertEqual(trainer.make_optimizer_and_scheduler(cfg, 'policy'), ('optimizer','scheduler'))
        guard.assert_called_once_with(cfg)
        original.assert_called_once_with(cfg, 'policy')

    def test_full_guard_failure_prevents_factory(self):
        original = Mock()
        trainer = SimpleNamespace(make_optimizer_and_scheduler=original)
        wrapper.install(trainer)
        with patch.object(wrapper, 'validate_registered_config', side_effect=ValueError('drift')):
            with self.assertRaises(ValueError):
                trainer.make_optimizer_and_scheduler(None,None)
        original.assert_not_called()

    def test_wrong_seed_prevents_factory(self):
        original = Mock()
        trainer = SimpleNamespace(make_optimizer_and_scheduler=original)
        wrapper.install(trainer)
        cfg = SimpleNamespace(steps=40000, seed=1000, resume=False, env=None)
        with patch.object(wrapper, 'validate_registered_config'):
            with self.assertRaises(ValueError):
                trainer.make_optimizer_and_scheduler(cfg,None)
        original.assert_not_called()
