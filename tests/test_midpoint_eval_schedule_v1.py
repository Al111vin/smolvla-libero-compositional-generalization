import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from midpoint_eval_schedule_v1 import schedule, validate_protocol


class MidpointTests(unittest.TestCase):
    def test_complete_prespecified_pairs(self):
        rows = schedule()
        self.assertEqual(len(rows), 40)
        for i in range(20):
            pair = rows[2*i:2*i+2]
            self.assertEqual([r['model'] for r in pair], ['single20k','joint80k'])
            self.assertTrue(all(r['init'] == i and r['repeat'] == 0 for r in pair))
            protocol = dict(init=i, cli_seed=12345+i, effective_seed=12345+2*i,
                environment_seed=12351, wait=10, max_steps=300, action_steps=25)
            for row in pair:
                validate_protocol(row, protocol)
            protocol['max_steps'] = 301
            with self.assertRaises(AssertionError):
                validate_protocol(pair[0], protocol)


if __name__ == '__main__':
    unittest.main()
