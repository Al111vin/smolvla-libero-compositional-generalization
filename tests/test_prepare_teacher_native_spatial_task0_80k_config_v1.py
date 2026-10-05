import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_teacher_native_spatial_task0_80k_config_v1 import (  # noqa: E402
    RUN_NAME,
    build_candidate,
    changed_paths,
    write_exclusive,
)


class PrepareTask0EightykConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = json.loads(
            (ROOT / "configs/teacher_native_spatial_task0_single_40k_batch2_20261003.json").read_text()
        )
        cls.design = json.loads(
            (ROOT / "results/teacher_native_spatial_task0_post40k_single_variable_budget_extension_design_v1_20261005.json").read_text()
        )

    def design_for_modified_source(self, source):
        design = copy.deepcopy(self.design)
        source_bytes = (json.dumps(source, indent=2, ensure_ascii=False) + "\n").encode()
        design["evidence_basis"]["current_40k_config"]["sha256"] = hashlib.sha256(source_bytes).hexdigest()
        return design

    def test_only_budget_run_identity_and_output_path_change(self):
        candidate, audit = build_candidate(copy.deepcopy(self.source), self.design)
        self.assertEqual(changed_paths(self.source, candidate), {"job_name", "output_dir", "steps"})
        self.assertEqual(candidate["job_name"], RUN_NAME)
        self.assertEqual(candidate["steps"], 80000)
        self.assertEqual(candidate["batch_size"], self.source["batch_size"])
        self.assertEqual(candidate["dataset"], self.source["dataset"])
        self.assertEqual(candidate["optimizer"], self.source["optimizer"])
        self.assertEqual(candidate["scheduler"], self.source["scheduler"])
        self.assertEqual(candidate["policy"], self.source["policy"])
        self.assertIsNone(candidate["env"])
        self.assertEqual(candidate["dataset"].get("eval_split", 0), 0)
        self.assertEqual(candidate.get("eval_steps", 0), 0)
        self.assertEqual(audit["task0_sample_draws"], 160000)
        self.assertEqual(audit["scheduler_expectation"]["expected_effective_warmup_steps"], 2666)
        self.assertTrue(audit["policy_evaluation_guard"]["runtime_confirmation_required"])

    def test_rejects_altered_recipe(self):
        source = copy.deepcopy(self.source)
        source["optimizer"]["lr"] *= 2
        with self.assertRaisesRegex(ValueError, "Source config hash"):
            build_candidate(source, self.design)

    def test_rejects_task_slice_or_batch_mismatch(self):
        source = copy.deepcopy(self.source)
        source["dataset"]["episodes"] = list(range(49))
        with self.assertRaisesRegex(ValueError, "Source config hash"):
            build_candidate(source, self.design)
        source = copy.deepcopy(self.source)
        source["batch_size"] = 4
        with self.assertRaisesRegex(ValueError, "Source config hash"):
            build_candidate(source, self.design)

    def test_rejects_unlock_or_eval_authorization_drift(self):
        design = copy.deepcopy(self.design)
        design["execution_state"]["fold02"] = "UNLOCKED"
        with self.assertRaisesRegex(ValueError, "Fold 02"):
            build_candidate(self.source, design)
        design = copy.deepcopy(self.design)
        design["execution_state"]["policy_evaluation_authorized"] = True
        with self.assertRaisesRegex(ValueError, "Policy evaluation"):
            build_candidate(self.source, design)

    def test_rejects_any_environment_policy_rollout_config(self):
        source = copy.deepcopy(self.source)
        source["env"] = {"type": "libero"}
        design = self.design_for_modified_source(source)
        with self.assertRaisesRegex(ValueError, "Policy environment evaluation"):
            build_candidate(source, design)

    def test_rejects_offline_held_out_policy_evaluation(self):
        source = copy.deepcopy(self.source)
        source["dataset"]["eval_split"] = 0.1
        design = self.design_for_modified_source(source)
        with self.assertRaisesRegex(ValueError, "Offline held-out evaluation"):
            build_candidate(source, design)
        source = copy.deepcopy(self.source)
        source["eval_steps"] = 1000
        design = self.design_for_modified_source(source)
        with self.assertRaisesRegex(ValueError, "Offline held-out evaluation"):
            build_candidate(source, design)

    def test_exclusive_writer_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "existing.json"
            target.write_text("original\n")
            with self.assertRaises(FileExistsError):
                write_exclusive(target, {"new": True})
            self.assertEqual(target.read_text(), "original\n")


if __name__ == "__main__":
    unittest.main()
