import ast
from pathlib import Path
import unittest


class CaptureBudgetTests(unittest.TestCase):
    def test_no_policy_or_training_calls(self):
        tree = ast.parse(Path("scripts/capture_reset_render_provenance_v1.py").read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        forbidden = {"select_action", "from_pretrained", "backward", "train"}
        self.assertFalse(any(isinstance(c.func, ast.Attribute) and c.func.attr in forbidden for c in calls))
        steps = [c for c in calls if isinstance(c.func, ast.Attribute) and c.func.attr == "step"]
        self.assertEqual(len(steps), 1)
        self.assertEqual(ast.unparse(steps[0].args[0]), "np.zeros(7, dtype=np.float32)")


if __name__ == "__main__":
    unittest.main()
