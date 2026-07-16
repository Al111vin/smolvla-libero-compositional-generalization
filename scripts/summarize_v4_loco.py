from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None


PROTOCOL_VERSION = "v4_loco_official_v1"
FROZEN_GIT_COMMIT = "a352a4b90aba1a3cb716f9d365bec1cfd139cbcd"

MODEL_SPECS = {
    "task3_holdout": {
        "heldout_task_id": 3,
        "seen_task_id": 6,
    },
    "task6_holdout": {
        "heldout_task_id": 6,
        "seen_task_id": 3,
    },
}

TRACE_KEYS = {
    "raw_actions",
    "processed_actions",
    "applied_actions",
    "rewards",
    "official_success",
    "done",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate and summarize the frozen V4 LIBERO-Spatial "
            "LOCO evaluation."
        )
    )
    parser.add_argument(
        "--results-dir",
        default="results/v4_loco_rollouts",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
    )
    parser.add_argument(
        "--logs-dir",
        default="logs/v4_loco_eval",
    )
    return parser.parse_args()


def require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def parse_success(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    require(
        set(normalized.unique()) <= {"true", "false"},
        f"Unexpected success values: {sorted(normalized.unique())}",
    )
    return normalized.eq("true")


def wilson_interval(
    successes: int,
    rollouts: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    proportion = successes / rollouts
    denominator = 1.0 + z * z / rollouts
    center = (
        proportion + z * z / (2.0 * rollouts)
    ) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / rollouts
            + z * z / (4.0 * rollouts * rollouts)
        )
        / denominator
    )
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    if low < 1e-15:
        low = 0.0
    if 1.0 - high < 1e-15:
        high = 1.0
    return low, high


def exact_mcnemar_p(seen_only: int, heldout_only: int) -> float:
    discordant = seen_only + heldout_only
    if discordant == 0:
        return 1.0

    smaller = min(seen_only, heldout_only)
    tail = sum(
        math.comb(discordant, index)
        for index in range(smaller + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file_set(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(
            str(path.relative_to(root)).encode("utf-8")
        )
        with path.open("rb") as file:
            while chunk := file.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def evaluator_fingerprint(repo_root: Path) -> str:
    paths = [
        repo_root / "scripts/eval_v4_loco.py",
        repo_root / "scripts/eval_v3_task0.py",
        repo_root / "results/V4_LOCO_EVAL_PROTOCOL.md",
    ]
    digest = hashlib.sha256()
    for path in paths:
        require(
            path.is_file(),
            f"Missing evaluator dependency: {path}",
        )
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def validate_trace(
    trace_path: Path,
    expected_steps: int,
):
    require(
        trace_path.is_file(),
        f"Missing action trace: {trace_path}",
    )

    with np.load(trace_path) as trace:
        require(
            set(trace.files) == TRACE_KEYS,
            f"Unexpected trace fields: {trace_path}",
        )

        for key in [
            "raw_actions",
            "processed_actions",
            "applied_actions",
        ]:
            require(
                trace[key].shape == (expected_steps, 7),
                f"Unexpected {key} shape in {trace_path}: "
                f"{trace[key].shape}",
            )

        for key in [
            "rewards",
            "official_success",
            "done",
        ]:
            require(
                trace[key].shape == (expected_steps,),
                f"Unexpected {key} shape in {trace_path}: "
                f"{trace[key].shape}",
            )


def load_model_results(
    results_dir: Path,
    model_label: str,
    spec: dict[str, int],
) -> pd.DataFrame:
    summary_path = (
        results_dir / f"{model_label}_formal_summary.csv"
    )
    require(
        summary_path.is_file(),
        f"Missing formal summary: {summary_path}",
    )

    df = pd.read_csv(summary_path)
    require(len(df) == 100, f"{model_label}: expected 100 rows")
    require(
        set(df["task_id"]) == {3, 6},
        f"{model_label}: expected tasks 3 and 6",
    )
    require(
        df.groupby("task_id").size().to_dict()
        == {3: 50, 6: 50},
        f"{model_label}: expected 50 rows per task",
    )
    require(
        not df.duplicated(["task_id", "init_index"]).any(),
        f"{model_label}: duplicate task/init rows",
    )

    for task_id in [3, 6]:
        indices = sorted(
            df.loc[df["task_id"] == task_id, "init_index"]
            .astype(int)
            .tolist()
        )
        require(
            indices == list(range(50)),
            f"{model_label} task {task_id}: "
            "expected init indices 0..49",
        )

    fixed_fields = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": FROZEN_GIT_COMMIT,
        "evaluation_mode": "formal",
        "model_label": model_label,
        "heldout_task_id": spec["heldout_task_id"],
        "suite": "libero_spatial",
        "init_source": "benchmark",
        "max_steps": 280,
        "wait_steps": 10,
        "n_action_steps": 25,
    }
    for column, expected in fixed_fields.items():
        require(
            set(df[column]) == {expected},
            f"{model_label}: unexpected {column}",
        )

    expected_roles = {
        spec["heldout_task_id"]: "heldout",
        spec["seen_task_id"]: "seen_control",
    }
    for task_id, role in expected_roles.items():
        require(
            set(
                df.loc[
                    df["task_id"] == task_id,
                    "evaluation_role",
                ]
            )
            == {role},
            f"{model_label} task {task_id}: unexpected role",
        )

    expected_env_seeds = (
        1000 + df["init_index"].astype(int)
    )
    expected_policy_seeds = (
        2000 + df["init_index"].astype(int)
    )
    require(
        np.array_equal(
            df["env_seed"].astype(int).to_numpy(),
            expected_env_seeds.to_numpy(),
        ),
        f"{model_label}: environment seed mismatch",
    )
    require(
        np.array_equal(
            df["policy_seed"].astype(int).to_numpy(),
            expected_policy_seeds.to_numpy(),
        ),
        f"{model_label}: policy seed mismatch",
    )

    require(
        df["evaluator_sha256"].nunique() == 1,
        f"{model_label}: mixed evaluator hashes",
    )
    require(
        df["checkpoint_sha256"].nunique() == 1,
        f"{model_label}: mixed checkpoint hashes",
    )
    require(
        df["steps"].between(1, 280).all(),
        f"{model_label}: invalid rollout step count",
    )

    df = df.copy()
    df["success_bool"] = parse_success(df["success"])

    actions_dir = (
        results_dir / f"{model_label}_formal_actions"
    )
    trace_paths = sorted(
        actions_dir.glob("task*_init*.npz")
    )
    require(
        len(trace_paths) == 100,
        f"{model_label}: expected 100 action traces",
    )

    action_trace_paths = []
    for row in df.itertuples():
        trace_path = actions_dir / (
            f"task{int(row.task_id)}_"
            f"init{int(row.init_index):02d}.npz"
        )
        validate_trace(trace_path, int(row.steps))
        action_trace_paths.append(str(trace_path))

    df["action_trace"] = action_trace_paths

    return df


def make_cell_summary(
    all_results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (model_label, task_id), part in all_results.groupby(
        ["model_label", "task_id"],
        sort=True,
    ):
        successes = int(part["success_bool"].sum())
        rollouts = len(part)
        low, high = wilson_interval(successes, rollouts)
        successful_steps = part.loc[
            part["success_bool"],
            "steps",
        ]

        rows.append(
            {
                "model_label": model_label,
                "heldout_task_id": int(
                    part.iloc[0]["heldout_task_id"]
                ),
                "task_id": int(task_id),
                "evaluation_role": part.iloc[0][
                    "evaluation_role"
                ],
                "successes": successes,
                "rollouts": rollouts,
                "success_rate": successes / rollouts,
                "wilson95_low": low,
                "wilson95_high": high,
                "mean_success_steps": (
                    float(successful_steps.mean())
                    if len(successful_steps)
                    else None
                ),
                "median_success_steps": (
                    float(successful_steps.median())
                    if len(successful_steps)
                    else None
                ),
            }
        )

    return pd.DataFrame(rows)


def make_paired_summary(
    all_results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for task_id in [3, 6]:
        heldout_model = f"task{task_id}_holdout"
        seen_model = (
            "task6_holdout"
            if task_id == 3
            else "task3_holdout"
        )

        heldout = all_results[
            (all_results["model_label"] == heldout_model)
            & (all_results["task_id"] == task_id)
        ][["init_index", "success_bool"]].rename(
            columns={"success_bool": "heldout_success"}
        )
        seen = all_results[
            (all_results["model_label"] == seen_model)
            & (all_results["task_id"] == task_id)
        ][["init_index", "success_bool"]].rename(
            columns={"success_bool": "seen_success"}
        )

        paired = seen.merge(
            heldout,
            on="init_index",
            validate="one_to_one",
        )
        require(
            len(paired) == 50,
            f"Task {task_id}: expected 50 paired states",
        )

        both_success = int(
            (
                paired["seen_success"]
                & paired["heldout_success"]
            ).sum()
        )
        seen_only = int(
            (
                paired["seen_success"]
                & ~paired["heldout_success"]
            ).sum()
        )
        heldout_only = int(
            (
                ~paired["seen_success"]
                & paired["heldout_success"]
            ).sum()
        )
        both_fail = int(
            (
                ~paired["seen_success"]
                & ~paired["heldout_success"]
            ).sum()
        )
        seen_rate = float(paired["seen_success"].mean())
        heldout_rate = float(
            paired["heldout_success"].mean()
        )

        rows.append(
            {
                "task_id": task_id,
                "seen_model": seen_model,
                "heldout_model": heldout_model,
                "paired_states": len(paired),
                "both_success": both_success,
                "seen_only_success": seen_only,
                "heldout_only_success": heldout_only,
                "both_fail": both_fail,
                "seen_success_rate": seen_rate,
                "heldout_success_rate": heldout_rate,
                "generalization_gap": (
                    seen_rate - heldout_rate
                ),
                "mcnemar_exact_p": exact_mcnemar_p(
                    seen_only,
                    heldout_only,
                ),
            }
        )

    summary = pd.DataFrame(rows)

    order = summary["mcnemar_exact_p"].sort_values().index
    adjusted = {}
    previous = 0.0
    hypotheses = len(summary)
    for rank, index in enumerate(order):
        value = min(
            1.0,
            (hypotheses - rank)
            * float(summary.loc[index, "mcnemar_exact_p"]),
        )
        value = max(previous, value)
        adjusted[index] = value
        previous = value
    summary["holm_adjusted_p"] = pd.Series(adjusted)

    return summary


def make_macro_summary(
    all_results: pd.DataFrame,
) -> pd.DataFrame:
    heldout = all_results[
        all_results["evaluation_role"] == "heldout"
    ]
    seen = all_results[
        all_results["evaluation_role"] == "seen_control"
    ]

    heldout_rates = heldout.groupby(
        "model_label"
    )["success_bool"].mean()
    seen_rates = seen.groupby(
        "model_label"
    )["success_bool"].mean()

    heldout_successes = int(heldout["success_bool"].sum())
    seen_successes = int(seen["success_bool"].sum())
    heldout_low, heldout_high = wilson_interval(
        heldout_successes,
        len(heldout),
    )
    seen_low, seen_high = wilson_interval(
        seen_successes,
        len(seen),
    )

    return pd.DataFrame(
        [
            {
                "heldout_macro_success_rate":
                    float(heldout_rates.mean()),
                "seen_control_macro_success_rate":
                    float(seen_rates.mean()),
                "macro_generalization_gap":
                    float(
                        seen_rates.mean()
                        - heldout_rates.mean()
                    ),
                "heldout_successes": heldout_successes,
                "heldout_rollouts": len(heldout),
                "heldout_pooled_wilson95_low": heldout_low,
                "heldout_pooled_wilson95_high": heldout_high,
                "seen_control_successes": seen_successes,
                "seen_control_rollouts": len(seen),
                "seen_control_pooled_wilson95_low": seen_low,
                "seen_control_pooled_wilson95_high": seen_high,
            }
        ]
    )


def percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def format_p(value: float) -> str:
    if value < 0.001:
        return f"{value:.3e}"
    return f"{value:.4f}"


def make_report(
    all_results: pd.DataFrame,
    cell_summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
    macro_summary: pd.DataFrame,
) -> str:
    macro = macro_summary.iloc[0]
    evaluator_hash = all_results[
        "evaluator_sha256"
    ].iloc[0]

    lines = [
        "# V4 LIBERO Spatial LOCO Results",
        "",
        "## Primary result",
        "",
        (
            "The frozen V4 SmolVLA policies achieved "
            f"**{percent(macro['heldout_macro_success_rate'])}** "
            "mean success on the two held-out compositional folds, "
            "versus "
            f"**{percent(macro['seen_control_macro_success_rate'])}** "
            "on matched seen-task controls. The macro "
            "generalization gap was "
            f"**{percent(macro['macro_generalization_gap'])}**."
        ),
        "",
        "## Per-cell results",
        "",
        (
            "| Model | Task | Role | Success | "
            "95% Wilson CI | Mean successful steps |"
        ),
        "|---|---:|---|---:|---:|---:|",
    ]

    for row in cell_summary.itertuples():
        mean_steps = (
            "—"
            if pd.isna(row.mean_success_steps)
            else f"{row.mean_success_steps:.2f}"
        )
        lines.append(
            f"| `{row.model_label}` | {row.task_id} | "
            f"{row.evaluation_role} | "
            f"{row.successes}/{row.rollouts} "
            f"({percent(row.success_rate)}) | "
            f"[{percent(row.wilson95_low)}, "
            f"{percent(row.wilson95_high)}] | "
            f"{mean_steps} |"
        )

    lines.extend(
        [
            "",
            "## Paired seen-versus-held-out comparison",
            "",
            (
                "For each target task, outcomes are paired by the same "
                "official benchmark initial-state index."
            ),
            "",
            (
                "| Task | Seen-only successes | "
                "Held-out-only successes | Both fail | "
                "Gap | Exact McNemar p | Holm-adjusted p |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in paired_summary.itertuples():
        lines.append(
            f"| {row.task_id} | "
            f"{row.seen_only_success} | "
            f"{row.heldout_only_success} | "
            f"{row.both_fail} | "
            f"{percent(row.generalization_gap)} | "
            f"{format_p(row.mcnemar_exact_p)} | "
            f"{format_p(row.holm_adjusted_p)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "- Both held-out cells were 0/50, while their matched "
                "seen controls were 42/50 and 44/50."
            ),
            (
                "- The strong seen controls show that the result is not "
                "explained by a generally broken policy or evaluation "
                "pipeline."
            ),
            (
                "- Under these two audited target-role LOCO folds, the "
                "policy learned the training combinations but did not "
                "transfer that behavior to the held-out composition."
            ),
            (
                "- Aggregating the two equally sized held-out cells "
                f"gives {int(macro['heldout_successes'])}/"
                f"{int(macro['heldout_rollouts'])}; the descriptive "
                "pooled Wilson 95% upper bound is "
                f"{percent(macro['heldout_pooled_wilson95_high'])}."
            ),
            "",
            "## Frozen protocol and provenance",
            "",
            f"- Protocol: `{PROTOCOL_VERSION}`",
            f"- Evaluation commit: `{FROZEN_GIT_COMMIT}`",
            f"- Evaluator fingerprint: `{evaluator_hash}`",
            "- Checkpoint: preselected step 90,000 for both folds",
            "- Official benchmark states: 0 through 49 per cell",
            "- Policy horizon: 280 actions",
            "- Stabilization: 10 steps with `[0,0,0,0,0,0,-1]`",
            "- Action execution horizon: `n_action_steps=25`",
            "- Success: `env.check_success()` only",
            "",
            "## Limitations",
            "",
            (
                "- Each fold has one training seed. Confidence intervals "
                "describe benchmark-state outcomes, not variation across "
                "training runs."
            ),
            (
                "- The 50 states per task are a fixed benchmark set; "
                "Wilson intervals and McNemar p-values are descriptive "
                "of state-level uncertainty under this protocol rather "
                "than a guarantee for an unrestricted task population."
            ),
            (
                "- The conclusion applies to the two audited "
                "LIBERO-Spatial target-role folds (tasks 3 and 6), not "
                "to every possible form of compositional generalization."
            ),
            (
                "- The action horizon was selected on task 0 before "
                "target evaluation, but that earlier ablation used the "
                "legacy all-zero stabilization action."
            ),
            (
                "- Post-hoc diagnostics may explain the failures, but "
                "must not be used to replace or retune these frozen "
                "primary results."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def write_csv(df: pd.DataFrame, path: Path):
    df.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    )


def make_manifest(
    all_results: pd.DataFrame,
    results_dir: Path,
    logs_dir: Path,
    output_paths: dict[str, Path],
    repo_root: Path,
) -> dict:
    models = {}

    for model_label in sorted(MODEL_SPECS):
        part = all_results[
            all_results["model_label"] == model_label
        ]
        actions_dir = (
            results_dir / f"{model_label}_formal_actions"
        )
        trace_paths = sorted(
            actions_dir.glob("task*_init*.npz")
        )
        summary_path = (
            results_dir / f"{model_label}_formal_summary.csv"
        )
        log_path = logs_dir / f"{model_label}_formal.log"

        models[model_label] = {
            "heldout_task_id": int(
                part.iloc[0]["heldout_task_id"]
            ),
            "checkpoint": str(part.iloc[0]["checkpoint"]),
            "checkpoint_sha256": str(
                part.iloc[0]["checkpoint_sha256"]
            ),
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "log": str(log_path),
            "log_sha256": (
                sha256_file(log_path)
                if log_path.is_file()
                else None
            ),
            "action_trace_count": len(trace_paths),
            "action_trace_bundle_sha256": sha256_file_set(
                trace_paths,
                actions_dir,
            ),
        }

    protocol_path = (
        repo_root / "results/V4_LOCO_EVAL_PROTOCOL.md"
    )
    output_hashes = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for name, path in output_paths.items()
        if name != "manifest"
    }

    return {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_commit": FROZEN_GIT_COMMIT,
        "evaluator_sha256": str(
            all_results.iloc[0]["evaluator_sha256"]
        ),
        "protocol_document": {
            "path": str(protocol_path.relative_to(repo_root)),
            "sha256": (
                sha256_file(protocol_path)
                if protocol_path.is_file()
                else None
            ),
        },
        "rollout_protocol": {
            "suite": "libero_spatial",
            "tasks": [3, 6],
            "init_indices": list(range(50)),
            "max_steps": 280,
            "wait_steps": 10,
            "stabilization_action": [
                0,
                0,
                0,
                0,
                0,
                0,
                -1,
            ],
            "n_action_steps": 25,
            "base_env_seed": 1000,
            "base_policy_seed": 2000,
            "success_criterion": "env.check_success()",
        },
        "models": models,
        "evaluation_environment": {
            "DISPLAY": ":99",
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "LIBGL_DRIVERS_PATH":
                "/usr/lib/x86_64-linux-gnu/dri",
            "LD_PRELOAD":
                "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
            "MUJOCO_GL": "glx",
            "PYOPENGL_PLATFORM": "glx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "summarizer_runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": (
                torch.__version__
                if torch is not None
                else "not-installed"
            ),
            "cuda_runtime": (
                torch.version.cuda
                if torch is not None
                else None
            ),
            "cuda_device": (
                torch.cuda.get_device_name(0)
                if (
                    torch is not None
                    and torch.cuda.is_available()
                )
                else None
            ),
            "packages": {
                name: package_version(name)
                for name in [
                    "lerobot",
                    "libero",
                    "robosuite",
                    "mujoco",
                    "numpy",
                    "pandas",
                ]
            },
        },
        "outputs": output_hashes,
    }


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = [
        load_model_results(
            results_dir,
            model_label,
            spec,
        )
        for model_label, spec in MODEL_SPECS.items()
    ]
    all_results = pd.concat(
        frames,
        ignore_index=True,
    )
    require(
        len(all_results) == 200,
        "Expected 200 formal rollouts",
    )
    require(
        all_results["evaluator_sha256"].nunique() == 1,
        "Formal runs used different evaluator fingerprints",
    )
    recorded_evaluator_hash = str(
        all_results.iloc[0]["evaluator_sha256"]
    )
    current_evaluator_hash = evaluator_fingerprint(repo_root)
    require(
        recorded_evaluator_hash == current_evaluator_hash,
        "Tracked evaluator, adapter, or protocol changed after "
        "formal evaluation",
    )

    all_results = all_results.sort_values(
        ["model_label", "task_id", "init_index"]
    ).reset_index(drop=True)

    cell_summary = make_cell_summary(all_results)
    paired_summary = make_paired_summary(all_results)
    macro_summary = make_macro_summary(all_results)

    rollout_columns = [
        "protocol_version",
        "git_commit",
        "evaluator_sha256",
        "model_label",
        "heldout_task_id",
        "evaluation_role",
        "suite",
        "task_id",
        "init_index",
        "language",
        "success_bool",
        "steps",
        "first_success_step",
        "total_reward",
        "max_steps",
        "wait_steps",
        "n_action_steps",
        "env_seed",
        "policy_seed",
        "checkpoint_label",
        "checkpoint_sha256",
        "checkpoint",
        "action_trace",
    ]
    tracked_rollouts = all_results[
        rollout_columns
    ].rename(columns={"success_bool": "success"})

    output_paths = {
        "rollouts": (
            output_dir
            / "eval_v4_loco_formal_rollouts.csv"
        ),
        "cells": (
            output_dir
            / "eval_v4_loco_cell_summary.csv"
        ),
        "paired": (
            output_dir
            / "eval_v4_loco_paired_summary.csv"
        ),
        "macro": (
            output_dir
            / "eval_v4_loco_macro_summary.csv"
        ),
        "report": output_dir / "V4_LOCO_RESULTS.md",
        "manifest": (
            output_dir / "eval_v4_loco_manifest.json"
        ),
    }

    write_csv(tracked_rollouts, output_paths["rollouts"])
    write_csv(cell_summary, output_paths["cells"])
    write_csv(paired_summary, output_paths["paired"])
    write_csv(macro_summary, output_paths["macro"])
    output_paths["report"].write_text(
        make_report(
            all_results,
            cell_summary,
            paired_summary,
            macro_summary,
        ),
        encoding="utf-8",
    )
    manifest = make_manifest(
        all_results,
        results_dir,
        Path(args.logs_dir),
        output_paths,
        repo_root,
    )
    output_paths["manifest"].write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Cell summary")
    print("=" * 80)
    print(cell_summary.to_string(index=False))
    print()
    print("Paired summary")
    print("=" * 80)
    print(paired_summary.to_string(index=False))
    print()
    print("Macro summary")
    print("=" * 80)
    print(macro_summary.to_string(index=False))
    print()
    print("Validated action traces: 200")
    for name, path in output_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
