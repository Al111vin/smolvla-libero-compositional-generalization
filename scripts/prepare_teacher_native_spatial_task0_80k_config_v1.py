#!/usr/bin/env python3
"""Prepare (but never launch) the registered single-task 80k contrast config."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


RUN_NAME = "teacher_native_spatial_task0_single_current_recipe_80k_batch2_v1"
BASELINE_NAME = "teacher_native_spatial_task0_single_current_recipe_40k_batch2_v1"
EXPECTED_ALLOWED_DIFFS = {"job_name", "output_dir", "steps"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def changed_paths(left: Any, right: Any, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        changed = set()
        for key in left.keys() | right.keys():
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                changed.add(path)
            else:
                changed |= changed_paths(left[key], right[key], path)
        return changed
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return {prefix}
        changed = set()
        for index, (a, b) in enumerate(zip(left, right)):
            changed |= changed_paths(a, b, f"{prefix}[{index}]")
        return changed
    return set() if left == right else {prefix}


def build_candidate(source: dict, design: dict) -> tuple[dict, dict]:
    source_bytes = (json.dumps(source, indent=2, ensure_ascii=False) + "\n").encode()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    expected_source_hash = design["evidence_basis"]["current_40k_config"]["sha256"]
    if source_hash != expected_source_hash:
        raise ValueError("Source config hash differs from the registered 40k baseline")
    if source.get("job_name") != BASELINE_NAME or source.get("steps") != 40000:
        raise ValueError("Source config is not the registered 40k baseline")
    if source.get("batch_size") != 2 or source.get("seed") != 1000:
        raise ValueError("Baseline batch size or seed differs from the registered recipe")
    if source.get("dataset", {}).get("episodes") != list(range(50)):
        raise ValueError("Baseline must use only frozen task-0 episodes 0-49")

    auth = design.get("user_authorization_record", {})
    protocol = design.get("proposed_protocol_if_later_authorized", {})
    execution = design.get("execution_state", {})
    if not (auth.get("scope") and execution.get("training_authorized") is True):
        raise ValueError("No explicit 80k training authorization recorded")
    if execution.get("policy_evaluation_authorized") is not False:
        raise ValueError("Policy evaluation authorization state is missing or unsafe")
    if execution.get("task_count_expansion") != "LOCKED" or execution.get("fold02") != "LOCKED":
        raise ValueError("Task expansion and Fold 02 must remain locked")

    # The 80k authorization covers training only, not policy rollout/inference.
    # LeRobot's training loop gates environment rollouts on both a positive
    # evaluation frequency and a configured environment; preserve the baseline's
    # null environment and reject any offline eval split/step schedule as well.
    if source.get("env") is not None:
        raise ValueError("Policy environment evaluation is not authorized")
    dataset_cfg = source.get("dataset", {})
    if dataset_cfg.get("eval_split", 0) != 0:
        raise ValueError("Offline held-out evaluation is not authorized")
    if source.get("eval_steps", 0) != 0:
        raise ValueError("Offline held-out evaluation steps must remain disabled")

    proposed = copy.deepcopy(source)
    proposed["job_name"] = RUN_NAME
    proposed["output_dir"] = protocol["output_isolation"]["training_root"]
    proposed["steps"] = 80000

    actual_diffs = changed_paths(source, proposed)
    if actual_diffs != EXPECTED_ALLOWED_DIFFS:
        raise ValueError(f"Unexpected recipe changes: {sorted(actual_diffs)}")

    scheduler = source["scheduler"]
    expected = protocol["80k_expected_exposure"]
    warmup = int(scheduler["num_warmup_steps"] * (80000 / scheduler["num_decay_steps"]))
    if warmup != expected["effective_warmup_steps_expected_from_same_scaling_rule"]:
        raise ValueError(f"Expected warmup mismatch: computed {warmup}")
    if scheduler["num_decay_steps"] != 90000:
        raise ValueError("Baseline scheduler family/decay parameter changed")

    audit = {
        "schema_version": 1,
        "experiment_id": RUN_NAME,
        "status": "config_prepared_static_only_remote_preflight_pending",
        "training_started": False,
        "policy_evaluation_authorized": False,
        "policy_evaluation_started": False,
        "task_count_expansion": "LOCKED",
        "fold02": "LOCKED",
        "source_config_sha256": None,
        "design_sha256": None,
        "candidate_config_sha256": None,
        "changed_config_paths": sorted(actual_diffs),
        "task0_sample_draws": proposed["steps"] * proposed["batch_size"],
        "scheduler_expectation": {
            "configured_warmup_steps": scheduler["num_warmup_steps"],
            "configured_decay_steps": scheduler["num_decay_steps"],
            "expected_effective_warmup_steps": warmup,
            "expected_effective_decay_horizon_steps": 80000,
            "runtime_confirmation_required": True,
        },
        "policy_evaluation_guard": {
            "env": proposed["env"],
            "dataset_eval_split": proposed["dataset"].get("eval_split", 0),
            "eval_steps": proposed.get("eval_steps", 0),
            "eval_freq_retained_from_baseline": proposed.get("eval_freq"),
            "interpretation": "A positive environment-evaluation frequency must not trigger rollouts while env is null; the resolved runtime config and trainer behavior must be freshly verified before training.",
            "runtime_confirmation_required": True,
        },
        "remote_checks_not_performed": [
            "V3 replay result reconciliation and preserved unique output/log",
            "fresh dataset root, HDF5/manifest/checksum, and task-0 slice verification",
            "fresh base initialization hash and resolved trainer config",
            "installed scheduler implementation and observed LR at warmup/midpoint/final",
            "GPU workload state, free disk, and exclusive output/log/config path absence",
        ],
        "candidate": proposed,
    }
    return proposed, audit


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--config-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args()
    if args.config_out.exists() or args.audit_out.exists():
        raise FileExistsError("Refusing to overwrite an existing config or audit")

    source = json.loads(args.source_config.read_text(encoding="utf-8"))
    design = json.loads(args.design.read_text(encoding="utf-8"))
    expected_source_hash = design["evidence_basis"]["current_40k_config"]["sha256"]
    actual_source_hash = sha256(args.source_config)
    if actual_source_hash != expected_source_hash:
        raise ValueError(f"Registered 40k config hash mismatch: {actual_source_hash}")
    candidate, audit = build_candidate(source, design)
    audit["source_config_sha256"] = actual_source_hash
    audit["design_sha256"] = sha256(args.design)
    candidate_bytes = (json.dumps(candidate, indent=2, ensure_ascii=False) + "\n").encode()
    audit["candidate_config_sha256"] = hashlib.sha256(candidate_bytes).hexdigest()

    # Check both paths before either exclusive write; any partial failure is preserved.
    if args.config_out.exists() or args.audit_out.exists():
        raise FileExistsError("Refusing to overwrite an existing config or audit")
    write_exclusive(args.config_out, candidate)
    write_exclusive(args.audit_out, audit)
    print(json.dumps({
        "status": audit["status"],
        "config": str(args.config_out),
        "config_sha256": audit["candidate_config_sha256"],
        "audit": str(args.audit_out),
        "training_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
