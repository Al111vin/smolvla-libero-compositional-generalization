#!/usr/bin/env python3
"""Hash-pin the 40 already-recorded fresh-seed task-0 traces; no simulator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from fresh_seed_eval_schedule_v1 import schedule
from scaling_eval_schedule_v1 import validate_protocol, validate_rollout


EVALUATOR_SHA256 = "89fd36a89dc45a56382219d4c3f6e5d12a3b1689abb3440ec29be346815af7cb"
TASK_TEXT = "pick up the black bowl between the plate and the ramekin and place it on the plate"
CHECKPOINT_SHA256 = {
    "single": "06e778a6a21510d7fccf11924a09c2b23485bf25883ffd566a0fb44e27137e03",
    "joint": "4b4127e49c6a3a467ea8c29055fce762effbe04fc466bdd6fadc51ef717ca628",
}
EXPECTED_LOSSES = [8, 11, 12, 13]
SUMMARY_SUFFIX = "_summary.csv"
ACTION_SUFFIX = "_actions.csv"
CODE_FILES = (
    "build_fresh_seed_stage_manifest_v1.py",
    "replay_fresh_seed_cpu_stage_v1.py",
    "controlled_cpu_replay_support_v1.py",
    "capture_reset_render_provenance_v1.py",
    "first_decision_provenance_v1.py",
    "fresh_seed_eval_schedule_v1.py",
    "scaling_eval_schedule_v1.py",
    "eval_v3_task0.py",
)


class ManifestError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_file(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ManifestError(f"Trace file escapes source root or is not a file: {path}")
    cursor = path
    while cursor != root:
        if cursor.is_symlink():
            raise ManifestError(f"Symlinked trace path is not allowed: {cursor}")
        cursor = cursor.parent
    return resolved


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not fields or not rows:
        raise ManifestError(f"Empty or malformed CSV: {path}")
    return fields, rows


def _without_model_weights(value: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(value, allow_nan=False))
    loaded = copied.get("loaded_model")
    if not isinstance(loaded, dict) or "state_dict" not in loaded:
        raise ManifestError("first_input must contain loaded_model.state_dict")
    del loaded["state_dict"]
    return copied


def _difference_paths(left: Any, right: Any, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        keys = set(left) | set(right)
        result: set[str] = set()
        for key in keys:
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                result.add(child)
            else:
                result.update(_difference_paths(left[key], right[key], child))
        return result
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return {prefix}
        result: set[str] = set()
        for index, (a, b) in enumerate(zip(left, right)):
            result.update(_difference_paths(a, b, f"{prefix}[{index}]"))
        return result
    return set() if type(left) is type(right) and left == right else {prefix}


def expected_rows() -> list[dict[str, Any]]:
    rows = [row for row in schedule()
            if row["task_id"] == 0 and row["kind"] == "paired"
            and row["model"] in CHECKPOINT_SHA256]
    rows.sort(key=lambda row: (row["init"], row["model"] != "single"))
    if len(rows) != 40 or len({row["key"] for row in rows}) != 40:
        raise ManifestError("Frozen schedule did not yield exactly 40 unique task-0 traces")
    return rows


def build_manifest(source_root: Path) -> dict[str, Any]:
    if source_root.is_symlink():
        raise ManifestError(f"Source root may not be a symlink: {source_root}")
    root = source_root.resolve(strict=True)
    if not root.is_dir():
        raise ManifestError(f"Source root is not a directory: {root}")

    records: list[dict[str, Any]] = []
    first_inputs: dict[tuple[str, int], dict[str, Any]] = {}
    protocols: dict[tuple[str, int], dict[str, Any]] = {}
    outcomes: dict[str, dict[int, bool]] = {"single": {}, "joint": {}}

    for row in expected_rows():
        model, init = str(row["model"]), int(row["init"])
        trace_dir = root / model / str(row["key"])
        if not trace_dir.is_dir() or trace_dir.is_symlink():
            raise ManifestError(f"Missing or symlinked trace directory: {trace_dir}")
        summary_candidates = sorted(trace_dir.glob(f"*{SUMMARY_SUFFIX}"))
        action_candidates = sorted(trace_dir.glob(f"*{ACTION_SUFFIX}"))
        if len(summary_candidates) != 1 or len(action_candidates) != 1:
            raise ManifestError(f"Expected one summary/actions CSV for {row['key']}")
        paths = {
            "summary_csv": _safe_file(summary_candidates[0], root),
            "actions_csv": _safe_file(action_candidates[0], root),
            "protocol_json": _safe_file(trace_dir / "protocol.json", root),
            "first_input_json": _safe_file(trace_dir / "first_input.json", root),
        }

        protocol = json.loads(paths["protocol_json"].read_text(encoding="utf-8"))
        validate_protocol(row, protocol)
        if protocol.get("model_sha") != CHECKPOINT_SHA256[model]:
            raise ManifestError(f"Checkpoint hash identity mismatch for {row['key']}")
        if protocol.get("evaluator_sha") != EVALUATOR_SHA256:
            raise ManifestError(f"Evaluator hash identity mismatch for {row['key']}")
        protocol_pair = (model, init)
        protocols[protocol_pair] = protocol

        first_input = json.loads(paths["first_input_json"].read_text(encoding="utf-8"))
        for field in ("model_arrays", "simulator_state", "rng", "processor", "images",
                      "input", "loaded_model"):
            if field not in first_input:
                raise ManifestError(f"Missing first-input field {field!r} for {row['key']}")
        first_inputs[protocol_pair] = first_input

        summary_fields, summary_rows = _read_csv(paths["summary_csv"])
        if len(summary_rows) != 1:
            raise ManifestError(f"Expected one summary row for {row['key']}")
        summary = summary_rows[0]
        if summary.get("language", "").strip() != TASK_TEXT:
            raise ManifestError(f"Task text mismatch for {row['key']}")
        action_fields, actions = _read_csv(paths["actions_csv"])
        if not {"reward", "step", *[f"state_{i}" for i in range(15)],
                *[f"applied_action_{i}" for i in range(7)],
                *[f"raw_action_{i}" for i in range(7)],
                *[f"processed_action_{i}" for i in range(7)]}.issubset(action_fields):
            raise ManifestError(f"Incomplete action schema for {row['key']}")
        success = validate_rollout(row, summary, actions)
        if summary.get("checkpoint") != protocol.get("checkpoint"):
            raise ManifestError(f"Summary/protocol checkpoint identity mismatch for {row['key']}")
        outcomes[model][init] = success

        records.append({
            "model": model,
            "key": row["key"],
            "init": init,
            "cli_seed": row["cli_seed"],
            "effective_seed": row["effective_seed"],
            "expected_success": success,
            "paths": {name: str(path) for name, path in paths.items()},
            "sha256": {name: sha256_file(path) for name, path in paths.items()},
        })

    for init in range(20):
        pair = ("single", init), ("joint", init)
        if any(key not in first_inputs for key in pair):
            raise ManifestError(f"Incomplete first-input pair at init {init}")
        if _difference_paths(*[_without_model_weights(first_inputs[key]) for key in pair]):
            raise ManifestError(f"Paired first-input mismatch at init {init}")
        protocol_differences = _difference_paths(protocols[pair[0]], protocols[pair[1]])
        if protocol_differences != {"checkpoint", "model_sha"}:
            raise ManifestError(
                f"Paired protocol differences unexpected at init {init}: {sorted(protocol_differences)}"
            )

    if sum(outcomes["single"].values()) != 14 or sum(outcomes["joint"].values()) != 10:
        raise ManifestError("Saved paired outcome counts differ from registered result")
    losses = [init for init in range(20)
              if outcomes["single"][init] and not outcomes["joint"][init]]
    gains = [init for init in range(20)
             if outcomes["joint"][init] and not outcomes["single"][init]]
    if losses != EXPECTED_LOSSES or gains:
        raise ManifestError(f"Saved discordant outcomes differ from registered result: {losses=}, {gains=}")

    scripts_root = Path(__file__).resolve().parent
    code_hashes = {name: sha256_file(scripts_root / name) for name in CODE_FILES}
    if code_hashes["eval_v3_task0.py"] != EVALUATOR_SHA256:
        raise ManifestError("Frozen task-0 evaluator code hash mismatch")

    payload = {
        "schema_version": 1,
        "experiment_id": "teacher_native_spatial_fresh_seed_stage_cpu_v1_20261012",
        "source_root": str(root),
        "source_inventory_count": len(records),
        "source_input_audit_digest": "d369a6212d91503b0bfa11abc1033314c55874ace048d728b2acf06226d42d13",
        "evaluator_sha256": EVALUATOR_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "code_hashes": code_hashes,
        "paired_outcomes": {model: {str(init): ok for init, ok in values.items()}
                             for model, values in outcomes.items()},
        "discordant_task0_inits": {"joint_losses": losses, "joint_gains": gains},
        "traces": records,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode("utf-8")
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing existing manifest path: {output}")
    payload = build_manifest(args.source_root)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "hash_pinned", "trace_count": len(payload["traces"]),
                      "manifest_sha256": payload["manifest_sha256"],
                      "simulator_started": False, "policy_inference": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
