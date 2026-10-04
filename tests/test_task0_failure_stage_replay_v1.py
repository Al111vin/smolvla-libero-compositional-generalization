from __future__ import annotations

import importlib.util
import csv
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "replay_teacher_native_spatial_task0_failure_stages_v1.py"
SPEC = importlib.util.spec_from_file_location("task0_failure_stage_replay", SCRIPT)
REPLAY = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(REPLAY)


def manifest_records():
    rows = []
    for index in range(20):
        rows.append(
            {
                "trace_id": f"paired_init{index:02d}",
                "group": "paired",
                "init_index": index,
                "cli_seed": 12345 + index,
                "effective_seed": 12345 + 2 * index,
                "expected_success": index % 2 == 0 or index == 1,
                "summary_csv": f"paired/{index}_summary.csv",
                "actions_csv": f"paired/{index}_actions.csv",
            }
        )
    for repeat in range(4):
        rows.append(
            {
                "trace_id": f"fixed_init3_repeat{repeat + 1:02d}",
                "group": "fixed_init3_repeat",
                "init_index": 3,
                "cli_seed": 12348,
                "effective_seed": 12351,
                "expected_success": repeat != 0,
                "summary_csv": f"repeats/{repeat}_summary.csv",
                "actions_csv": f"repeats/{repeat}_actions.csv",
            }
        )
    return rows


class Task0FailureStageReplayTests(unittest.TestCase):
    def test_registered_24_trace_inventory(self):
        records = manifest_records()
        result = REPLAY.validate_manifest_shape(
            {
                "schema_version": 1,
                "experiment_id": "teacher_native_spatial_task0_failure_stage_replay_v1_20261004",
                "task_suite": "libero_spatial",
                "task_id": 0,
                "traces": records,
            }
        )
        self.assertEqual(len(result), 24)
        self.assertEqual(sum(row["group"] == "paired" for row in result), 20)
        self.assertEqual(sum(row["group"] == "fixed_init3_repeat" for row in result), 4)

    def test_rejects_missing_paired_initialization(self):
        records = manifest_records()
        records = [row for row in records if row["init_index"] != 19 or row["group"] != "paired"]
        with self.assertRaises(REPLAY.ProtocolError):
            REPLAY.validate_manifest_shape(
                {
                    "schema_version": 1,
                    "experiment_id": "teacher_native_spatial_task0_failure_stage_replay_v1_20261004",
                    "task_suite": "libero_spatial",
                    "task_id": 0,
                    "traces": records,
                }
            )

    def test_stage_label_is_first_unreached_proxy_only(self):
        events = {
            "approach": True,
            "grasp": True,
            "lift": True,
            "transport": False,
            "placement_terminal": False,
        }
        self.assertEqual(REPLAY.first_unreached_stage(events), "transport")
        self.assertIsNone(
            REPLAY.first_unreached_stage({key: True for key in events})
        )

    def test_nonmonotone_proxy_sequence_is_ambiguous(self):
        events = {
            "approach": False,
            "grasp": True,
            "lift": False,
            "transport": True,
            "placement_terminal": False,
        }
        first_steps = {
            "approach": None,
            "grasp": 12,
            "lift": None,
            "transport": 9,
            "placement_terminal": None,
        }
        label, monotone = REPLAY.classify_stage_proxies(events, first_steps)
        self.assertEqual(label, "ambiguous_nonmonotone_sequence")
        self.assertFalse(monotone)

    def test_monotone_sequence_keeps_first_missing_proxy(self):
        events = {
            "approach": True,
            "grasp": True,
            "lift": False,
            "transport": False,
            "placement_terminal": False,
        }
        first_steps = {
            "approach": 2,
            "grasp": 4,
            "lift": None,
            "transport": None,
            "placement_terminal": None,
        }
        label, monotone = REPLAY.classify_stage_proxies(events, first_steps)
        self.assertEqual(label, "lift")
        self.assertTrue(monotone)

    def test_bool_parser_fails_closed(self):
        self.assertTrue(REPLAY.parse_bool("True", "success"))
        self.assertFalse(REPLAY.parse_bool("0", "success"))
        with self.assertRaises(REPLAY.ProtocolError):
            REPLAY.parse_bool("unknown", "success")

    def test_simulator_execution_requires_explicit_flag(self):
        with self.assertRaisesRegex(REPLAY.ProtocolError, "--authorize-execution"):
            REPLAY.require_execution_authorization(
                __import__("argparse").Namespace(authorize_execution=False)
            )
        REPLAY.require_execution_authorization(
            __import__("argparse").Namespace(authorize_execution=True)
        )

    def test_source_csv_preflight_checks_registered_run_and_outcome_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records = manifest_records()
            paired_success_indices = set(range(11)) - {3} | {11}
            for record in records:
                source_dir = root / Path(record["summary_csv"]).parent
                source_dir.mkdir(parents=True, exist_ok=True)
                source_success = record["expected_success"]
                summary_path = root / record["summary_csv"]
                summary_row = {
                    "suite": "libero_spatial",
                    "task_id": 0,
                    "init_source": "benchmark",
                    "init_index": record["init_index"],
                    "language": "pick up the black bowl and place it on the plate",
                    "success": str(source_success).lower(),
                    "total_reward": 1 if source_success else 0,
                    "steps": 1,
                    "wait_steps": 10,
                    "n_action_steps": 25,
                    "seed": record["effective_seed"],
                    "checkpoint": "/r/teacher_native_spatial_task0_single_current_recipe_40k_batch2_v1/checkpoints/040000/pretrained_model",
                }
                if record["group"] == "paired":
                    summary_row["success"] = str(record["init_index"] in paired_success_indices).lower()
                    summary_row["total_reward"] = 1 if record["init_index"] in paired_success_indices else 0
                    record["expected_success"] = record["init_index"] in paired_success_indices
                with summary_path.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=sorted(REPLAY.SUMMARY_FIELDS))
                    writer.writeheader()
                    writer.writerow(summary_row)
                actions_path = root / record["actions_csv"]
                actions_path.parent.mkdir(parents=True, exist_ok=True)
                action_row = {"step": 0, "reward": summary_row["total_reward"]}
                action_row.update({f"state_{i}": 0 for i in range(15)})
                action_row.update({f"applied_action_{i}": 0 for i in range(7)})
                with actions_path.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=sorted(REPLAY.ACTION_FIELDS))
                    writer.writeheader()
                    writer.writerow(action_row)

            manifest_path = root / "manifest.json"
            normalized = {
                "schema_version": 1,
                "experiment_id": "teacher_native_spatial_task0_failure_stage_replay_v1_20261004",
                "task_suite": "libero_spatial",
                "task_id": 0,
                "traces": records,
            }
            manifest_path.write_text(__import__("json").dumps(normalized), encoding="utf-8")
            ordered = REPLAY.validate_manifest_shape(normalized)
            cases, inventory = REPLAY.validate_source_records(manifest_path, ordered)
            self.assertEqual(inventory["trace_count"], 24)
            self.assertEqual(inventory["paired_count"], 20)
            self.assertEqual(inventory["fixed_init3_repeat_count"], 4)
            output = root / "new_output"
            REPLAY.validate_output_root(output, cases)
            output.mkdir()
            with self.assertRaises(REPLAY.ProtocolError):
                REPLAY.validate_output_root(output, cases)


if __name__ == "__main__":
    unittest.main()
