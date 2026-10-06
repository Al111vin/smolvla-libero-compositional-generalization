#!/usr/bin/env python3
"""Prepare a fresh isolated 80k retry config; never launches training."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_CONFIG_SHA256 = "c79765254e18d35577b5dfe65f8acb35d85ed458d7b92f53459e481d78cae628"
RUN_NAME = "teacher_native_spatial_task0_single_current_recipe_80k_batch2_v1_retry1_20261006"
OUTPUT_DIR = (
    "/root/smolvla-training-prep/results/training/"
    "teacher_native_spatial_task0_single_current_recipe_80k_batch2_v1_retry1_20261006"
)
OLD_FAILED_OUTPUT_DIR = (
    "/root/smolvla-training-prep/results/training/"
    "teacher_native_spatial_task0_single_current_recipe_80k_batch2_v1"
)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def changed_paths(left: Any, right: Any, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        result: set[str] = set()
        for key in left.keys() | right.keys():
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                result.add(path)
            else:
                result |= changed_paths(left[key], right[key], path)
        return result
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return {prefix}
        result = set()
        for index, (a, b) in enumerate(zip(left, right)):
            result |= changed_paths(a, b, f"{prefix}[{index}]")
        return result
    return set() if left == right else {prefix}


def build_retry(source: dict[str, Any], source_hash: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if source_hash != SOURCE_CONFIG_SHA256:
        raise ValueError("Registered 80k source config hash mismatch")
    if source.get("job_name") != "teacher_native_spatial_task0_single_current_recipe_80k_batch2_v1":
        raise ValueError("Source is not the registered 80k candidate")
    if source.get("steps") != 80000 or source.get("batch_size") != 2 or source.get("seed") != 1000:
        raise ValueError("80k training budget, batch size, or seed drifted")
    if source.get("env") is not None or source.get("eval_steps", 0) != 0:
        raise ValueError("Policy evaluation/inference is not authorized")
    if source.get("dataset", {}).get("eval_split", 0) != 0:
        raise ValueError("Offline policy evaluation is not authorized")
    if source.get("output_dir") != OLD_FAILED_OUTPUT_DIR:
        raise ValueError("The preserved failed-attempt output path no longer matches registration")

    retry = copy.deepcopy(source)
    retry["job_name"] = RUN_NAME
    retry["output_dir"] = OUTPUT_DIR
    if changed_paths(source, retry) != {"job_name", "output_dir"}:
        raise ValueError("Unexpected retry config changes")

    audit = {
        "schema_version": 1,
        "experiment_id": RUN_NAME,
        "status": "local_retry_config_prepared_remote_preflight_pending",
        "training_started": False,
        "user_authorization": "User explicitly authorized one corrected 80k training attempt in this chat on 2026-10-06.",
        "source_80k_config_sha256": source_hash,
        "retry_config_sha256": sha256_bytes(canonical_bytes(retry)),
        "changed_config_paths_vs_registered_80k_candidate": ["job_name", "output_dir"],
        "preserved_failed_attempt_output_dir_do_not_touch": OLD_FAILED_OUTPUT_DIR,
        "new_output_dir_must_be_absent_before_launch": OUTPUT_DIR,
        "log_placement": "Outside the training output directory; never mkdir the training output directory before LeRobot validation.",
        "training": {"steps": 80000, "batch_size": 2, "seed": 1000},
        "policy_evaluation_authorized": False,
        "policy_evaluation_started": False,
        "fold02": "LOCKED",
        "task_count_expansion": "LOCKED",
        "remote_preflight_required": [
            "Resolve the correct registered conversion manifest path and verify its hash against the 8bb81e1d...11ba1 source-manifest reference; do not compare it to the distinct source/preflight_manifest.json.",
            "Verify converted dataset tree, metadata/statistics, exact task-0 slice, frozen data and model hashes.",
            "Verify installed trainer/scheduler hashes and runtime-resolved config/warmup/decay.",
            "Verify GPU/process exclusivity, disk, lock availability, and absence of retry config/log/PID/output paths.",
            "Verify trainer env=null and no policy rollout/evaluation path is enabled.",
        ],
        "launch_guard": "Single attempt only after all remote preflight checks pass; use a new absent output path, keep the log outside it, and let LeRobot create the output directory.",
    }
    return retry, audit


def write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--config-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args()
    if args.config_out.exists() or args.audit_out.exists():
        raise FileExistsError("Refusing to overwrite an existing retry config or audit")
    source_bytes = args.source_config.read_bytes()
    source = json.loads(source_bytes)
    candidate, audit = build_retry(source, sha256_bytes(source_bytes))
    config_bytes = canonical_bytes(candidate)
    audit["retry_config_sha256"] = sha256_bytes(config_bytes)
    audit_bytes = canonical_bytes(audit)
    if args.config_out.exists() or args.audit_out.exists():
        raise FileExistsError("Refusing to overwrite an existing retry config or audit")
    write_exclusive(args.config_out, config_bytes)
    write_exclusive(args.audit_out, audit_bytes)
    print(json.dumps({"status": audit["status"], "training_started": False,
                      "retry_config_sha256": audit["retry_config_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
