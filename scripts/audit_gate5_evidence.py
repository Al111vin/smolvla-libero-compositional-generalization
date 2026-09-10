#!/usr/bin/env python3
"""Audit current LIBERO-36 Gate-5 evidence without modifying it.

Gate-5 controllers predate a single result schema.  This reader recognizes
the pass fields used by their task summaries, requires a clean failed-check
list whenever one is recorded, and reports every unproven task explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate5-dir",
        type=Path,
        default=Path("results/libero36_source_xy_redesign_diagnostic/gate5"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"expected object at {path}")
    return value


def clean_checks(summary: dict) -> bool:
    failed = summary.get("failed_checks")
    return failed is None or failed == []


def summary_passed(summary: dict) -> bool:
    if not clean_checks(summary):
        return False
    direct = (
        summary.get("passed"),
        summary.get("success"),
        summary.get("attempt_succeeded"),
        summary.get("replay_passed"),
        summary.get("exact_replay_passed"),
        summary.get("gate5_task_passed"),
    )
    if any(value is True for value in direct):
        return True
    return bool(
        summary.get("recording_passed")
        and summary.get("replay_passed")
    )


def task_evidence(task_dir: Path) -> tuple[bool, str]:
    summary_path = task_dir / "summary.json"
    replay_summary_path = task_dir / "replay_summary.json"
    if not summary_path.is_file():
        return False, "missing summary"
    summary = read_json(summary_path)

    # Group controllers store the recording and replay reports inside one
    # top-level summary object.
    if "trajectory" in summary and "replay" in summary:
        trajectory = summary["trajectory"]
        replay = summary["replay"]
        passed = (
            isinstance(trajectory, dict)
            and isinstance(replay, dict)
            and summary_passed(trajectory)
            and bool(replay.get("replay_passed"))
            and clean_checks(replay)
        )
        return passed, "combined trajectory/replay summary"

    if not summary_passed(summary):
        return False, "summary has no clean pass signal"
    if replay_summary_path.is_file():
        replay = read_json(replay_summary_path)
        if not summary_passed(replay):
            return False, "replay summary is not a clean pass"
    return True, "task summary"


def main() -> None:
    args = parse_args()
    gate5 = args.gate5_dir
    evidence: dict[int, dict] = {}
    for task_dir in sorted(gate5.glob("task_*_*")):
        if not task_dir.is_dir():
            continue
        summary_path = task_dir / "summary.json"
        if not summary_path.is_file():
            continue
        summary = read_json(summary_path)
        task_id = summary.get("task_id")
        if not isinstance(task_id, int):
            continue
        passed, reason = task_evidence(task_dir)
        evidence[task_id] = {
            "passed": passed,
            "path": str(task_dir),
            "reason": reason,
        }

    # Task 26's established formal record predates per-task subdirectories.
    root_summary = gate5 / "summary.json"
    root_replay = gate5 / "formal_helper_summary.json"
    if root_summary.is_file() and root_replay.is_file():
        recorded = read_json(root_summary)
        replayed = read_json(root_replay)
        if recorded.get("task_id") == 26:
            evidence[26] = {
                "passed": bool(recorded.get("success"))
                and bool(replayed.get("passed"))
                and clean_checks(recorded)
                and clean_checks(replayed),
                "path": str(gate5),
                "reason": "root trajectory plus formal replay helper",
            }

    passed = sorted(task_id for task_id, item in evidence.items() if item["passed"])
    missing = [task_id for task_id in range(36) if task_id not in passed]
    report = {
        "expected_task_ids": list(range(36)),
        "passed_task_ids": passed,
        "passed_count": len(passed),
        "missing_task_ids": missing,
        "evidence": {str(key): value for key, value in sorted(evidence.items())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
