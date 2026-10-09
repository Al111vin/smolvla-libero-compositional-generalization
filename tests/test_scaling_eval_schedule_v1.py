import unittest
from scripts.scaling_eval_schedule_v1 import schedule, validate_protocol, validate_rollout, gate


class ScheduleTests(unittest.TestCase):
    def test_full_rollout_evidence_and_wrong_task_rejected(self):
        row = schedule()[2]
        summary = dict(suite='libero_spatial', task_id=2, init_source='benchmark',
                       init_index=0, seed=12345, wait_steps=10, n_action_steps=25,
                       success='True', steps=1, total_reward=1)
        action = dict(step=0, reward=1)
        action.update({f'{p}_{j}': 0 for p in
                       ('raw_action', 'processed_action', 'applied_action') for j in range(7)})
        action.update({f'state_{j}': 0 for j in range(15)})
        self.assertTrue(validate_rollout(row, summary, [action]))
        for changes in ({'task_id': 0}, {'success': 'False'}, {'steps': 301},
                        {'total_reward': 2}):
            with self.assertRaises(ValueError):
                validate_rollout(row, dict(summary, **changes), [action])
        for changes in ({'step': 1}, {'raw_action_0': float('nan')},
                        {'applied_action_0': 0.5}):
            with self.assertRaises(ValueError):
                validate_rollout(row, summary, [dict(action, **changes)])

    def test_complete_interleaved_budget(self):
        rows = schedule()
        self.assertEqual(len(rows), 96)
        self.assertEqual(len({r['key'] for r in rows}), 96)
        for offset in range(0, 96, 4):
            self.assertEqual([r['task_id'] for r in rows[offset:offset+4]], list(range(4)))
        for task in range(4):
            selected = [r for r in rows if r['task_id'] == task]
            self.assertEqual([r['init'] for r in selected if r['kind'] == 'paired'], list(range(20)))
            self.assertEqual(sum(r['init'] == 3 for r in selected), 5)

    def test_every_protocol_field_is_guarded(self):
        row = schedule()[0]
        validate_protocol(row, row)
        for key in ('task_id', 'init', 'cli_seed', 'effective_seed',
                    'environment_seed', 'wait', 'max_steps', 'action_steps'):
            bad = dict(row)
            bad[key] += 1
            with self.assertRaises(ValueError):
                validate_protocol(row, bad)

    def test_complete_results_required(self):
        with self.assertRaises(ValueError):
            gate({})
        outcomes = {r['key']: True for r in schedule()}
        self.assertTrue(gate(outcomes)['passed'])
        outcomes[schedule()[12]['key']] = False  # paired task0 init3
        self.assertFalse(gate(outcomes)['passed'])
        outcomes[schedule()[0]['key']] = 'True'
        with self.assertRaises(ValueError):
            gate(outcomes)


if __name__ == '__main__':
    unittest.main()
