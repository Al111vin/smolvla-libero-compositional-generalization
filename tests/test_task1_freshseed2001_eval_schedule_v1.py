import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from task1_freshseed2001_eval_schedule_v1 import gate_task1, schedule


class Task1FreshSeed2001ScheduleTests(unittest.TestCase):
    def test_registered_schedule_has_20_paired_and_four_fixed_init3_repeats(self):
        rows = schedule()
        self.assertEqual(len(rows), 24)
        self.assertTrue(all(row["task_id"] == 1 for row in rows))
        self.assertEqual(sum(row["kind"] == "paired" for row in rows), 20)
        self.assertEqual(sum(row["kind"] == "extra_init3" for row in rows), 4)
        self.assertEqual(sum(row["init"] == 3 for row in rows), 5)
        self.assertEqual([row["cli_seed"] for row in rows if row["kind"] == "extra_init3"], [12348] * 4)
        self.assertEqual([row["effective_seed"] for row in rows if row["kind"] == "extra_init3"], [12351] * 4)

    def test_threshold_is_diagnostic_only_and_does_not_unlock_progression(self):
        rows = schedule()
        outcomes = {row["key"]: True for row in rows}
        result = gate_task1(outcomes)
        self.assertTrue(result["passed"])
        self.assertFalse(result["automatic_next_stage"])
        self.assertEqual(result["fold02"], "LOCKED")
        outcomes = {row["key"]: False for row in rows}
        for row in [r for r in rows if r["kind"] == "paired"][:9]:
            outcomes[row["key"]] = True
        self.assertFalse(gate_task1(outcomes)["passed"])
        outcomes = {row["key"]: True for row in rows}
        outcomes[next(row["key"] for row in rows if row["kind"] == "extra_init3")] = False
        self.assertFalse(gate_task1(outcomes)["passed"])


if __name__ == "__main__":
    unittest.main()
