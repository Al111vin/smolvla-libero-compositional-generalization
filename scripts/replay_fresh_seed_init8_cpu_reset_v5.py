#!/usr/bin/env python3
"""V5 path-isolated launcher over the frozen, hash-pinned V4 CPU runner."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import replay_fresh_seed_init8_cpu_reset_v4 as base  # noqa: E402

EXPERIMENT_ID = "teacher_native_spatial_fresh_seed_init8_cpu_reset_v5_20261013"
MANIFEST_PATH = Path(
    "/root/smolvla-eval-prep/recovery/teacher_native_spatial_fresh_seed_init8_cpu_reset_manifest_v5_20261013.json"
)
OUTPUT_ROOT = Path(
    "/root/smolvla-eval-prep/results/teacher_native_spatial_fresh_seed_init8_cpu_reset_v5_20261013"
)

# The V4 implementation itself is immutable in this version. V5 changes only
# identity/path parameters and explicitly pins both launcher and base runner.
base.EXPERIMENT_ID = EXPERIMENT_ID
base.MANIFEST_PATH = MANIFEST_PATH
base.OUTPUT_ROOT = OUTPUT_ROOT
_base_code_paths = base._code_paths


def _code_paths() -> dict[str, Path]:
    paths = dict(_base_code_paths())
    paths["base_runner"] = paths.pop("runner")
    paths.pop("unit_tests")
    paths["runner"] = Path(__file__).resolve()
    paths["unit_tests"] = SCRIPTS.parent / "tests" / "test_fresh_seed_init8_cpu_reset_v5.py"
    return paths


def _code_hashes() -> dict[str, str]:
    paths = _code_paths()
    missing = [name for name, path in paths.items() if not path.is_file() or path.is_symlink()]
    if missing:
        raise base.ReplayError(f"Required code/test inputs are missing or symlinked: {missing}")
    return {name: base.sha256_file(path) for name, path in paths.items()}


_base_build_run_manifest = base._build_run_manifest


def _build_run_manifest() -> dict[str, Any]:
    payload = _base_build_run_manifest()
    payload["runner_sha256"] = base.sha256_file(Path(__file__).resolve())
    payload["code_sha256"] = _code_hashes()
    return payload


base._code_paths = _code_paths
base._code_hashes = _code_hashes
base._build_run_manifest = _build_run_manifest

if __name__ == "__main__":
    raise SystemExit(base.main())
