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

    def test_paired_threshold_boundary(self):
        for count in (10,11):
            rows=self.rows()
            successful=set([3]+[i for i in range(20) if i!=3][:count-1])
            for row in rows:
                if row["kind"]=="paired": row["success"]=row["init"] in successful
            self.assertEqual(summarize(rows,0)["models"]["80k"]["gate_passed"],count==11)

    def test_invalid_steps_rejected(self):
        for value in (True,1.5,0,301):
            rows=self.rows(); rows[0]["steps"]=value
            with self.assertRaises(AssertionError): summarize(rows,0)

    def test_discordant_pair_statistics(self):
        rows=self.rows()
        for row in rows:
            if row["kind"]=="paired" and row["model"]=="40k" and row["init"]<6:
                row["success"]=False
        result=summarize(rows,0)
        self.assertEqual(result["gains"],list(range(6)))
        self.assertEqual(result["losses"],[])
        self.assertEqual(result["mcnemar_exact_p"],0.03125)


if __name__=="__main__": unittest.main()
