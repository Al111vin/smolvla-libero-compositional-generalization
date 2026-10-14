import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "seed2002_builder", ROOT / "scripts/build_fresh_seed2002_matched_pair_configs_v1.py"
)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)

class Seed2002ConfigBuilderTests(unittest.TestCase):
    def test_only_registered_fields_change(self):
        for arm, spec in builder.ARMS.items():
            with self.subTest(arm=arm):
                source = json.loads((ROOT / spec["source"]).read_text())
                result = builder.build(arm)
                changed = {k for k in source if source[k] != result[k]}
                self.assertEqual(changed, {"seed", "job_name", "output_dir"})
                self.assertEqual(result["seed"], 2002)
                self.assertEqual(result["steps"], spec["steps"])
                self.assertEqual(result["batch_size"], 2)
                self.assertIsNone(result["env"])
                self.assertFalse(result["resume"])

    def test_invalid_arm_and_source_drift_fail_closed(self):
        with self.assertRaises(ValueError):
            builder.build("task8")
        original = builder.ARMS["single"]["source_sha256"]
        try:
            builder.ARMS["single"]["source_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                builder.build("single")
        finally:
            builder.ARMS["single"]["source_sha256"] = original

if __name__ == "__main__":
    unittest.main()
