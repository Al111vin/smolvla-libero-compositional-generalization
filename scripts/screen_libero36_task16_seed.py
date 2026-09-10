from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--push-scale", type=float, default=0.6)
    parser.add_argument("--orientation-tolerance", type=float, default=0.060)
    parser.add_argument(
        "--controller",
        type=Path,
        default=Path("/tmp/run_mug_push_task16_horiz.py"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/tmp/libero36_task16_seed_screen"),
    )
    args = parser.parse_args()

    source = args.controller.read_text(encoding="utf-8")
    marker = "calibrator.main()"
    if source.count(marker) != 1:
        raise RuntimeError(
            f"Expected one {marker!r} call, found {source.count(marker)}"
        )

    # Load the frozen Task-16 controller configuration without starting its
    # command-line calibration driver. The pilot uses the formal-v6 open
    # gripper convention and a slightly wider pose convergence tolerance.
    source = source.replace(
        "calibrator.ACTIVE_GRIPPER_ACTION = -1.0",
        "calibrator.ACTIVE_GRIPPER_ACTION = 0.0",
        1,
    )
    source = source.replace(marker, "", 1)
    namespace: dict[str, object] = {
        "__file__": str(args.controller),
        "__name__": "task16_seed_screen_controller",
    }
    exec(compile(source, str(args.controller), "exec"), namespace)

    calibrator = namespace["calibrator"]
    namespace["PUSH_TRANSLATION_SCALE"] = args.push_scale
    calibrator.ORIENTATION_TOLERANCE_RAD = args.orientation_tolerance
    calibrator.reset_validator.SOURCE_SAMPLER_RADII_M[
        "white_yellow_mug"
    ] = 0.075

    rows = calibrator.reset_validator.read_layout_spec(
        Path("data/libero_36/layout_spec.csv")
    )
    row = next(
        dict(item)
        for item in rows
        if int(item["task_id"]) == 16 and int(item["layout_id"]) == 1
    )
    candidate = namespace["selected_candidates_for"](row)[0]
    original = calibrator.run_original_attempt(row, candidate, args.seed)
    arrays = original["arrays"]

    root = args.output_root / f"seed_{args.seed}"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "trajectory.npz"
    np.savez_compressed(path, **arrays)

    terminal_relation = bool(
        np.asarray(arrays["relation"], dtype=bool)[-20:].all()
    )
    terminal_xy = bool(
        np.asarray(arrays["xy_in_target"], dtype=bool)[-20:].all()
    )
    safe = bool(
        np.asarray(arrays["table_support"], dtype=bool).all()
        and not np.asarray(arrays["grasp_proxy"], dtype=bool).any()
        and not np.asarray(
            arrays["robot_distractor_contact"], dtype=bool
        ).any()
        and not np.asarray(
            arrays["unexpected_object_contact"], dtype=bool
        ).any()
    )
    passed = bool(terminal_relation and terminal_xy and safe)
    print(
        "SCREEN",
        {
            "seed": args.seed,
            "push_scale": args.push_scale,
            "orientation_tolerance": args.orientation_tolerance,
            "actions": int(len(arrays["actions"])),
            "terminal_relation": terminal_relation,
            "terminal_xy": terminal_xy,
            "safe": safe,
            "passed": passed,
            "trajectory": str(path),
        },
        flush=True,
    )
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
