import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_teacher_native_spatial_task0_80k_retry_config_v1 import (  # noqa: E402
    OUTPUT_DIR,
    RUN_NAME,
    build_retry,
    changed_paths,
)


class PrepareTask0EightykRetryConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "configs/teacher_native_spatial_task0_single_current_recipe_80k_batch2_v1.json"
        cls.raw = cls.path.read_bytes()
        cls.source = json.loads(cls.raw)
        cls.digest = hashlib.sha256(cls.raw).hexdigest()

    def test_retry_changes_only_identity_and_fresh_output_path(self):
        retry, audit = build_retry(copy.deepcopy(self.source), self.digest)
        self.assertEqual(changed_paths(self.source, retry), {"job_name", "output_dir"})
        self.assertEqual(retry["job_name"], RUN_NAME)
        self.assertEqual(retry["output_dir"], OUTPUT_DIR)
        self.assertEqual(retry["steps"], 80000)
        self.assertEqual(retry["batch_size"], 2)
        self.assertEqual(retry["seed"], 1000)
        self.assertIsNone(retry["env"])
        self.assertFalse(audit["policy_evaluation_authorized"])
        self.assertEqual(audit["fold02"], "LOCKED")
        self.assertEqual(audit["task_count_expansion"], "LOCKED")

    def test_rejects_wrong_source_hash(self):
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            build_retry(self.source, "0" * 64)

    def test_rejects_eval_or_dataset_drift(self):
        modified = copy.deepcopy(self.source)
        modified["env"] = {"type": "libero"}
        with self.assertRaisesRegex(ValueError, "not authorized"):
            build_retry(modified, self.digest)
        modified = copy.deepcopy(self.source)
        modified["dataset"]["eval_split"] = 0.1
        with self.assertRaisesRegex(ValueError, "not authorized"):
            build_retry(modified, self.digest)


if __name__ == "__main__":
    unittest.main()
