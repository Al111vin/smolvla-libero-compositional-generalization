from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import validate_libero_36_envs as validation
except ModuleNotFoundError:
    from scripts import validate_libero_36_envs as validation


DEFAULT_FOVY_DEGREES = [60.0, 65.0, 70.0, 75.0, 80.0]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render the same settled LIBERO-36 states at candidate "
            "agentview vertical fields of view. This is a diagnostic "
            "only and does not change the frozen v4 protocol."
        )
    )
    parser.add_argument(
        "--layout-spec",
        type=Path,
        default=Path("data/libero_36/layout_spec.csv"),
    )
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument(
        "--layout-ids",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
    )
    parser.add_argument(
        "--fovy-degrees",
        type=float,
        nargs="+",
        default=DEFAULT_FOVY_DEGREES,
    )
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument(
        "--seed-base",
        type=int,
        default=validation.FULL_SEED_BASE,
    )
    parser.add_argument(
        "--max-reset-attempts",
        type=int,
        default=validation.FULL_MAX_RESET_ATTEMPTS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/libero36_fovy_sweep"),
    )
    return parser.parse_args()


def validate_args(args):
    if args.task_id < 0:
        raise ValueError("--task-id cannot be negative")
    if len(args.layout_ids) != len(set(args.layout_ids)):
        raise ValueError("--layout-ids contains duplicates")
    if len(args.fovy_degrees) != len(set(args.fovy_degrees)):
        raise ValueError("--fovy-degrees contains duplicates")
    if args.settle_steps < 0:
        raise ValueError("--settle-steps cannot be negative")
    if args.max_reset_attempts <= 0:
        raise ValueError("--max-reset-attempts must be positive")
    for fovy in args.fovy_degrees:
        if not 0.0 < fovy < 180.0:
            raise ValueError(f"Invalid field of view: {fovy}")


def apply_agentview_fovy(env, fovy: float):
    camera_id = env.sim.model.camera_name2id("agentview")
    env.sim.model.cam_fovy[camera_id] = float(fovy)
    env.sim.forward()
    return env.env._get_observations(force_update=True)


def save_contact_sheet(
    output_dir: Path,
    task_id: int,
    fovy: float,
    layout_ids: list[int],
):
    images = []
    for layout_id in layout_ids:
        path = output_dir / (
            f"task_{task_id:03d}_layout_{layout_id}_"
            f"fovy_{fovy:g}_agentview.png"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing preview for task {task_id}, "
                f"layout {layout_id}, fovy {fovy}: {path}"
            )
        images.append(Image.open(path).convert("RGB"))

    widths, heights = zip(*(image.size for image in images))
    if len(set(widths)) != 1 or len(set(heights)) != 1:
        raise ValueError("Preview image sizes do not match")
    sheet = Image.new(
        "RGB",
        (sum(widths), max(heights)),
    )
    x_offset = 0
    for image in images:
        sheet.paste(image, (x_offset, 0))
        x_offset += image.width
    sheet.save(
        output_dir / f"fovy_{fovy:g}_contact_sheet.png"
    )


def main():
    args = parse_args()
    validate_args(args)
    validation.validate_registered_geometry()

    layout_spec = validation.resolve_repo_path(args.layout_spec)
    all_rows = validation.read_layout_spec(layout_spec)
    requested_layouts = set(args.layout_ids)
    rows = [
        row
        for row in all_rows
        if row["task_id"] == args.task_id
        and row["layout_id"] in requested_layouts
    ]
    rows.sort(key=lambda row: row["layout_id"])
    actual_layouts = [row["layout_id"] for row in rows]
    if actual_layouts != sorted(args.layout_ids):
        raise ValueError(
            f"Requested layouts {sorted(args.layout_ids)}, "
            f"found {actual_layouts}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    print("=" * 80)
    print("LIBERO-36 agentview FOV diagnostic")
    print("task:", args.task_id)
    print("layouts:", args.layout_ids)
    print("candidate fovy values:", args.fovy_degrees)
    print("output:", output_dir)
    print("NOTE: this diagnostic does not modify the v4 protocol.")

    for row in rows:
        env = validation.make_environment(row["bddl_path"])
        try:
            seed = validation.seed_for(row, 0, args.seed_base)
            obs, reset_attempts, frozen_camera = validation.safe_reset(
                env,
                seed,
                args.max_reset_attempts,
            )
            receiver = validation.receiver_for_skill(row["skill"])
            validation.settle_environment(
                env,
                obs,
                row,
                receiver,
                args.settle_steps,
            )

            for fovy in args.fovy_degrees:
                rendered = apply_agentview_fovy(env, fovy)
                image = np.asarray(rendered["agentview_image"])
                if image.shape != (128, 128, 3):
                    raise ValueError(
                        f"Unexpected agentview shape: {image.shape}"
                    )
                path = output_dir / (
                    f"task_{args.task_id:03d}_"
                    f"layout_{row['layout_id']}_"
                    f"fovy_{fovy:g}_agentview.png"
                )
                Image.fromarray(image).save(path)
                records.append(
                    {
                        "task_id": args.task_id,
                        "layout_id": row["layout_id"],
                        "seed": seed,
                        "reset_attempts": reset_attempts,
                        "frozen_camera_before_candidate": frozen_camera,
                        "candidate_fovy_degrees": float(fovy),
                        "output_path": str(path),
                    }
                )
                print(
                    f"task={args.task_id:02d} "
                    f"layout={row['layout_id']} "
                    f"frozen_fovy={frozen_camera['fovy_deg']:g} "
                    f"candidate_fovy={fovy:g} "
                    f"saved={path.name}"
                )
        finally:
            env.close()

    for fovy in args.fovy_degrees:
        save_contact_sheet(
            output_dir,
            args.task_id,
            fovy,
            args.layout_ids,
        )

    manifest = {
        "diagnostic_only": True,
        "frozen_protocol_unchanged": True,
        "protocol_version": validation.PROTOCOL_VERSION,
        "layout_spec": str(layout_spec),
        "task_id": args.task_id,
        "layout_ids": args.layout_ids,
        "candidate_fovy_degrees": args.fovy_degrees,
        "settle_steps": args.settle_steps,
        "records": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("=" * 80)
    print("Diagnostic complete")
    print("individual previews:", len(records))
    print("contact sheets:", len(args.fovy_degrees))
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()
