#!/usr/bin/env python3
"""V11: isolated V4-engine wrapper for a two-trace, CPU-only reset smoke."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import replay_fresh_seed_init8_cpu_reset_v4 as engine  # noqa: E402

EXPERIMENT_ID = "teacher_native_spatial_fresh_seed_init8_cpu_reset_v11_20261014"
MANIFEST_PATH = Path("/root/smolvla-eval-prep/recovery/teacher_native_spatial_fresh_seed_init8_cpu_reset_manifest_v11_20261014.json")
OUTPUT_ROOT = Path("/root/smolvla-eval-prep/results/teacher_native_spatial_fresh_seed_init8_cpu_reset_v11_20261014")

# Patch the frozen V4 engine exactly once. No V5-V10 wrapper or verifier is imported.
engine.EXPERIMENT_ID = EXPERIMENT_ID
engine.MANIFEST_PATH = MANIFEST_PATH
engine.OUTPUT_ROOT = OUTPUT_ROOT
_base_code_paths = engine._code_paths
_base_build_run_manifest = engine._build_run_manifest


def _code_paths() -> dict[str, Path]:
    paths = dict(_base_code_paths())
    paths["base_runner_v4"] = paths.pop("runner")
    paths["base_test_v4"] = paths.pop("unit_tests")
    paths["runner"] = Path(__file__).resolve()
    paths["unit_tests"] = SCRIPTS.parent / "tests" / "test_fresh_seed_init8_cpu_reset_v11.py"
    return paths


def _code_hashes() -> dict[str, str]:
    paths = _code_paths()
    missing = [name for name, path in paths.items() if not path.is_file() or path.is_symlink()]
    if missing:
        raise engine.ReplayError(f"Required V11 code/test inputs are missing or symlinked: {missing}")
    return {name: engine.sha256_file(path) for name, path in paths.items()}


def _build_run_manifest() -> dict[str, Any]:
    payload = _base_build_run_manifest()
    payload["runner_sha256"] = engine.sha256_file(Path(__file__).resolve())
    payload["code_sha256"] = _code_hashes()
    return payload


engine._code_paths = _code_paths
engine._code_hashes = _code_hashes
engine._build_run_manifest = _build_run_manifest

if __name__ == "__main__":
    raise SystemExit(engine.main())
