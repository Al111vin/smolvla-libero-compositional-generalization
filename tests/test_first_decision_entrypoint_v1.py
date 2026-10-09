"""Static budget checks; not a substitute for real processor/runtime tests."""
import ast
from pathlib import Path
import unittest


class EntrypointBudgetTests(unittest.TestCase):
    def test_only_zero_wait_actions_enter_environment(self):
        source = Path("scripts/diagnose_first_decision_v1.py").read_text()
        tree = ast.parse(source)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        steps = [n for n in calls if isinstance(n.func, ast.Attribute)
                 and n.func.attr == "step"]
        self.assertEqual(len(steps), 1)
        self.assertEqual(ast.unparse(steps[0].args[0]), "np.zeros(7, dtype=np.float32)")
        self.assertIn("range(10)", source)
        self.assertEqual(sum(isinstance(n.func, ast.Name) and
                             n.func.id == "paired_first_calls" for n in calls), 1)


if __name__ == "__main__":
    unittest.main()
