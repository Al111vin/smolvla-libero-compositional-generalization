import unittest
from scripts.compare_midpoint_outcomes_v1 import compare


class PairedComparisonTests(unittest.TestCase):
    def test_registered_discordance(self):
        ref={i:i in (0,12,13,14) for i in range(20)}
        new={i:i==6 for i in range(20)}
        result=compare(ref,new)
        self.assertEqual(result['gains'],[6])
        self.assertEqual(result['losses'],[0,12,13,14])
        self.assertEqual(result['exact_mcnemar_two_sided_p'],.375)
        self.assertFalse(result['automatic_checkpoint_selection'])
    def test_identical(self):
        ref={i:False for i in range(20)}
        self.assertEqual(compare(ref,ref)['exact_mcnemar_two_sided_p'],1.)
    def test_incomplete_or_nonboolean(self):
        ref={i:False for i in range(20)}
        with self.assertRaises(ValueError):compare(ref,{0:False})
        with self.assertRaises(ValueError):compare(ref,{i:0 for i in range(20)})


if __name__=='__main__':unittest.main()
