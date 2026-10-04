#!/usr/bin/env python3
"""Build a hash-pinned manifest for the private, already-recorded task-0 traces.

This is an inventory/preflight utility only: it reads existing summary/action
CSVs, writes one new manifest, and never opens LIBERO or starts a simulator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


RUN_ID = "teacher_native_spatial_task0_single_current_recipe_40k_batch2_v1"
EXPERIMENT_ID = "teacher_native_spatial_task0_failure_stage_replay_v1_20261004"
TASK_TEXT = "pick up the black bowl between the plate and the ramekin and place it on the plate"
RUNNER_PATH = Path(__file__).resolve().with_name(
    "replay_teacher_native_spatial_task0_failure_stages_v1.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("task0_failure_stage_replay", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load replay validator: {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _single_summary(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        row = next(reader, None)
        if row is None or next(reader, None) is not None:
            raise ValueError(f"Expected exactly one summary row: {path}")
        return row


def _collect_group(root_arg: Path, group: str) -> list[dict[str, Any]]:
    root = root_arg.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    summaries = sorted(root.rglob("*_summary.csv"))
    if not summaries:
        raise FileNotFoundError(f"No *_summary.csv files beneath {root}")
    records = []
    for candidate in summaries:
        summary_path = candidate.resolve(strict=True)
        if not summary_path.is_relative_to(root):
            raise ValueError(f"Summary path escapes its selected group root: {candidate}")
        action_path = summary_path.with_name(summary_path.name.removesuffix("_summary.csv") + "_actions.csv")
        if not action_path.is_file():
            raise FileNotFoundError(f"Matching action CSV is missing: {action_path}")
        action_path = action_path.resolve(strict=True)
        if not action_path.is_relative_to(root):
            raise ValueError(f"Action path escapes its selected group root: {action_path}")
        row = _single_summary(summary_path)
        if row.get("suite") != "libero_spatial" or row.get("task_id") != "0":
            raise ValueError(f"Unexpected suite/task in {summary_path}")
        if row.get("language", "").strip() != TASK_TEXT:
            raise ValueError(f"Unexpected task language in {summary_path}")
        if row.get("init_source") != "benchmark":
            raise ValueError(f"Unexpected initialization source in {summary_path}")
        if RUN_ID not in row.get("checkpoint", "") or not any(
            step in row.get("checkpoint", "") for step in ("040000", "40000")
        ):
            raise ValueError(f"Not a registered final-checkpoint trace: {summary_path}")
        init_index = int(row["init_index"])
        if group == "fixed_init3_repeat" and init_index != 3:
            raise ValueError(f"Fixed-repeat root contains non-init3 trace: {summary_path}")
        effective_seed = int(row["seed"])
        cli_seed = effective_seed - init_index
        if cli_seed != 12345 + init_index or effective_seed != 12345 + 2 * init_index:
            raise ValueError(f"Unexpected registered seed mapping in {summary_path}")
        success_text = row["success"].strip().lower()
        if success_text not in {"1", "true", "yes", "0", "false", "no"}:
            raise ValueError(f"Unrecognized success value in {summary_path}: {row['success']!r}")
        expected_success = success_text in {"1", "true", "yes"}
        rel = summary_path.relative_to(root).with_name(summary_path.name.removesuffix("_summary.csv"))
        trace_id = f"{group}_{rel.as_posix().replace('/', '_')}"
        records.append(
            {
                "trace_id": trace_id,
                "group": group,
                "init_index": init_index,
                "cli_seed": cli_seed,
                "effective_seed": effective_seed,
                "expected_success": expected_success,
                "summary_csv": str(summary_path),
                "summary_sha256": sha256(summary_path),
                "actions_csv": str(action_path),
                "actions_sha256": sha256(action_path),
            }
        )
    return records


def build_manifest(paired_root: Path, repeats_root: Path) -> dict[str, Any]:
    records = _collect_group(paired_root, "paired") + _collect_group(
        repeats_root, "fixed_init3_repeat"
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "task_suite": "libero_spatial",
        "task_id": 0,
        "traces": records,
    }
    runner = load_runner()
    normalized = runner.validate_manifest_shape(manifest)
    _, inventory = runner.validate_source_records(Path("/private-manifest-root/manifest.json"), normalized)
    return {**manifest, "inventory_preflight": inventory}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-root", type=Path, required=True)
    parser.add_argument("--repeats-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {output}")
    manifest = build_manifest(args.paired_root, args.repeats_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "manifest": str(output),
                "trace_count": manifest["inventory_preflight"]["trace_count"],
                "paired_count": manifest["inventory_preflight"]["paired_count"],
                "fixed_init3_repeat_count": manifest["inventory_preflight"]["fixed_init3_repeat_count"],
                "inputs_hash_pinned": True,
                "simulator_started": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        import sys

        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
