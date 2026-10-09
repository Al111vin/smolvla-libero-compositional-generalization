import unittest
from scripts.controlled_eval_schedule_v1 import schedule, validate_protocol, validate_rollout


class ScheduleTests(unittest.TestCase):
    def test_bad_actions_rejected(self):
        row=schedule()[0]
        summary=dict(suite="libero_spatial",task_id="0",init_source="benchmark",init_index="0",
            seed="12345",wait_steps="10",n_action_steps="25",success="False",steps="1",total_reward="0")
        action={"step":"0","reward":"0"}
        action.update({f"{prefix}_{j}":"0" for prefix in ("raw_action","processed_action","applied_action") for j in range(7)})
        action.update({f"state_{j}":"0" for j in range(15)})
        validate_rollout(row,summary,[action])
        action["raw_action_0"]="nan"
        with self.assertRaises(AssertionError): validate_rollout(row,summary,[action])
        action["raw_action_0"]="0"
        action["applied_action_0"]="0.5"
        with self.assertRaises(AssertionError): validate_rollout(row,summary,[action])

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
