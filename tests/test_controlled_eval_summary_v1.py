import unittest
from scripts.controlled_eval_schedule_v1 import schedule
from scripts.summarize_controlled_eval_v1 import summarize


class SummaryTests(unittest.TestCase):
    def rows(self):
        return [dict(r,success=True,steps=100) for r in schedule()]

    def test_complete_gate(self):
        s=summarize(self.rows(),0)
        self.assertTrue(s["models"]["80k"]["gate_passed"])
        self.assertEqual(s["mcnemar_exact_p"],1.)

    def test_missing_duplicate_failure_rejected(self):
        for rows,code in ((self.rows()[:-1],0),(self.rows(),1),([self.rows()[0]]*48,0)):
            with self.assertRaises(AssertionError): summarize(rows,code)

    def test_one_fixed_failure_blocks_gate(self):
        rows=self.rows(); rows[-1]["success"]=False
        self.assertFalse(summarize(rows,0)["models"]["80k"]["gate_passed"])


if __name__=="__main__": unittest.main()
