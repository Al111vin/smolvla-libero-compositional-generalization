from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_fresh_seed_stage_manifest_v1 import (  # noqa: E402
    CHECKPOINT_SHA256,
    EVALUATOR_SHA256,
    ManifestError,
    build_manifest,
    expected_rows,
)
from replay_fresh_seed_cpu_stage_v1 import ReplayError, _read_source, run  # noqa: E402


def _write_csv(path: Path, fields: list[str], row: dict[str, object]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def make_fixture(root: Path) -> None:
    summary_fields = ["suite", "task_id", "init_source", "init_index", "language",
                      "success", "total_reward", "steps", "wait_steps", "n_action_steps",
                      "seed", "checkpoint"]
    action_fields = ["step", "reward"]
    action_fields += [f"{prefix}_{index}" for prefix in
                      ("raw_action", "processed_action", "applied_action") for index in range(7)]
    action_fields += [f"state_{index}" for index in range(15)]

    for row in expected_rows():
        model, init, key = row["model"], row["init"], row["key"]
        # The registered discordant pairs are the only joint failures.
        success = init in {0, 1, 2, 3, 4, 5, 6, 7, 9, 10} if model == "joint" else init not in {14, 15, 16, 17, 18, 19}
        if model == "single" and init in {8, 11, 12, 13}:
            success = True
        trace = root / model / key
        trace.mkdir(parents=True)
        checkpoint = f"/private/{model}/model.safetensors"
        protocol_row = {key: value for key, value in row.items() if key != "model"}
        protocol_row["key"] = row["key"].removeprefix(model + "_")
        protocol = dict(protocol_row)
        protocol.update(checkpoint=checkpoint, model_sha=CHECKPOINT_SHA256[model],
                        evaluator_sha=EVALUATOR_SHA256, wrapper_sha="fixture-wrapper")
        (trace / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
        state_dict = f"{model}-{init}"
        capture = {
            "model_arrays": {"model": "same"},
            "simulator_state": "same",
            "rng": {"np": "same"},
            "processor": {"p": "same"},
            "images": {"image": "same"},
            "input": {"input": "same"},
            "loaded_model": {"state_dict": state_dict},
        }
        (trace / "first_input.json").write_text(json.dumps(capture), encoding="utf-8")
        reward = 1.0 if success else 0.0
        summary = {
            "suite": "libero_spatial", "task_id": 0, "init_source": "benchmark",
            "init_index": init,
            "language": "pick up the black bowl between the plate and the ramekin and place it on the plate",
            "success": str(success), "total_reward": reward, "steps": 1,
            "wait_steps": 10, "n_action_steps": 25, "seed": row["effective_seed"],
            "checkpoint": checkpoint,
        }
        _write_csv(trace / f"{key}{'_summary.csv'}", summary_fields, summary)
        action = {"step": 0, "reward": reward}
        action.update({f"{prefix}_{index}": 0.0 for prefix in
                      ("raw_action", "processed_action", "applied_action") for index in range(7)})
        action.update({f"state_{index}": 0.0 for index in range(15)})
        _write_csv(trace / f"{key}{'_actions.csv'}", action_fields, action)


class ManifestBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self.temp = tempfile.TemporaryDirectory()
        self.source_root = Path(self.temp.name) / "source"
        self.source_root.mkdir()
        make_fixture(self.source_root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_hash_pins_complete_40_trace_inventory_and_matched_starts(self) -> None:
        manifest = build_manifest(self.source_root)
        self.assertEqual(manifest["source_inventory_count"], 40)
        self.assertEqual(manifest["discordant_task0_inits"],
                         {"joint_losses": [8, 11, 12, 13], "joint_gains": []})
        self.assertEqual(len(manifest["traces"]), 40)
        self.assertEqual(manifest["evaluator_sha256"], EVALUATOR_SHA256)
        self.assertIn("replay_fresh_seed_cpu_stage_v1.py", manifest["code_hashes"])
        self.assertTrue(all(set(row["sha256"]) == {
            "summary_csv", "actions_csv", "protocol_json", "first_input_json"
        } for row in manifest["traces"]))

    def test_fails_closed_if_paired_saved_start_differs(self) -> None:
        joint_capture = self.source_root / "joint" / "joint_task0_paired_0_r0_160k" / "first_input.json"
        data = json.loads(joint_capture.read_text())
        data["processor"]["p"] = "mismatch"
        joint_capture.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ManifestError, "Paired first-input mismatch"):
            build_manifest(self.source_root)

    def test_fails_closed_if_registered_protocol_changes(self) -> None:
        path = self.source_root / "joint" / "joint_task0_paired_0_r0_160k" / "protocol.json"
        data = json.loads(path.read_text())
        data["effective_seed"] += 1
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Protocol mismatch: effective_seed"):
            build_manifest(self.source_root)

    def test_saved_trace_hash_is_rechecked_immediately_before_read(self) -> None:
        source = self.source_root / "mutable.csv"
        source.write_text("changed", encoding="utf-8")
        trace = {"key": "fixture", "paths": {"actions_csv": str(source)},
                 "sha256": {"actions_csv": "0" * 64}}
        with self.assertRaisesRegex(ReplayError, "immediately before replay"):
            _read_source(trace)

    def test_replay_requires_explicit_execution_flag_before_remote_paths(self) -> None:
        with self.assertRaisesRegex(ReplayError, "Explicit --authorize-execution"):
            run(Path("/nonexistent/manifest.json"), Path("/nonexistent/output"), False)


if __name__ == "__main__":
    unittest.main()
