import unittest
from scripts.summarize_reset_render_contrast_v1 import summarize


class SummaryTests(unittest.TestCase):
    def records(self):
        return [{"mode": mode, "seed": 12351 if mode == "preseed" else None,
                 "policy_calls": 0, "predicted_actions": 0, "wait_steps": 10,
                 "init": 3, "state": "same", "model_arrays": {"body_pos": str(i) if mode == "frozen_order" else "same"},
                 "cameras": {"agentview_image": str(i) if mode == "frozen_order" else "same",
                             "robot0_eye_in_hand_image": "same"}}
                for mode in ("frozen_order", "preseed") for i in range(3)]

    def test_counts_not_causal_claim(self):
        s = summarize(self.records(), 0)
        self.assertEqual(s["frozen_order"]["model_unique_counts"]["body_pos"], 3)
        self.assertEqual(s["preseed"]["camera_unique_counts"]["agentview_image"], 1)

    def test_incomplete_or_failed_rejected(self):
        for records, code in ((self.records()[:-1], 0), (self.records(), 1)):
            with self.assertRaises(AssertionError):
                summarize(records, code)


if __name__ == "__main__":
    unittest.main()
