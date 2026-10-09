import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('summary', Path(__file__).resolve().parents[1]/'scripts/summarize_cpu_contact_observers_v1.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ContactTests(unittest.TestCase):
    def test_target_bowl_and_order(self):
        pad = 'gripper0_finger1_pad_collision'
        self.assertTrue(module.contact_with([[pad, 'akita_black_bowl_1_g24']], pad))
        self.assertTrue(module.contact_with([['akita_black_bowl_1_g24', pad]], pad))
        self.assertFalse(module.contact_with([[pad, 'akita_black_bowl_2_g24']], pad))
        self.assertFalse(module.contact_with([[None, pad]], pad))


if __name__ == '__main__':
    unittest.main()
