import unittest
from scripts.controlled_eval_schedule_v1 import schedule, validate_protocol


class ScheduleTests(unittest.TestCase):
    def test_finite_balanced_interleaving(self):
        rows = schedule()
        self.assertEqual(len(rows), 48)
        for model in ("40k", "80k"):
            ds = [r for r in rows if r["model"] == model]
            self.assertEqual([r["init"] for r in ds if r["kind"] == "paired"], list(range(20)))
            self.assertEqual(sum(r["init"] == 3 for r in ds), 5)
        self.assertTrue(all(rows[i]["model"] == "40k" and rows[i+1]["model"] == "80k" for i in range(0,48,2)))

    def test_protocol_drift_rejected(self):
        r=schedule()[6]
        p=dict(init=r["init"],cli_seed=r["cli_seed"],effective_seed=12345+2*r["init"],
               environment_seed=12351,wait=10,max_steps=300,action_steps=25)
        validate_protocol(r,p)
        p["wait"]=0
        with self.assertRaises(AssertionError): validate_protocol(r,p)


if __name__ == "__main__": unittest.main()
