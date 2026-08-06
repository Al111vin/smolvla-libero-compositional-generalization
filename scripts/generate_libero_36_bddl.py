from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

try:
    import libero_36_camera as camera_config
except ModuleNotFoundError:
    from scripts import libero_36_camera as camera_config


PROTOCOL_VERSION = "libero_36_proxy_tabletop_draft_v4"
BENCHMARK_NAME = "libero_registered_object_36_proxy_tabletop"
DEFAULT_OUTPUT_DIR = Path("data/libero_36")
LAYOUT_COUNT = 4

# Recognized v0/v1/v2/v3 signatures are accepted only so an explicit
# --overwrite can atomically migrate those generated outputs to v4.
LEGACY_OVERWRITE_SIGNATURES = {
    (
        "libero_36_proxy_draft_v0",
        "libero_registered_object_36_proxy",
    ),
    (
        "libero_36_proxy_tabletop_draft_v1",
        "libero_registered_object_36_proxy_tabletop",
    ),
    (
        "libero_36_proxy_tabletop_draft_v2",
        "libero_registered_object_36_proxy_tabletop",
    ),
    (
        "libero_36_proxy_tabletop_draft_v3",
        "libero_registered_object_36_proxy_tabletop",
    ),
}

# Provisional only. This must be calibrated and frozen before formal
# evaluation from passive-settle and valid scripted push trajectories.
PROVISIONAL_PUSH_MAX_LIFT_M = 0.03

OBJECTS = {
    "akita_black_bowl": {
        "instance": "akita_black_bowl_1",
        "display": "akita black bowl",
    },
    "white_yellow_mug": {
        "instance": "white_yellow_mug_1",
        "display": "yellow and white mug",
    },
    "alphabet_soup": {
        "instance": "alphabet_soup_1",
        "display": "alphabet soup",
    },
    "cream_cheese": {
        "instance": "cream_cheese_1",
        "display": "cream cheese box",
    },
}

# Frozen values read from each registered object's
# horizontal_radius_site. robosuite uses the first position component
# as the TableRegionSampler boundary inset. Keeping this check in the
# generator makes an asset or dependency change fail before any BDDL is
# replaced.
SOURCE_SAMPLER_RADII_M = {
    "akita_black_bowl": 0.025,
    "white_yellow_mug": 0.025,
    "alphabet_soup": 0.025,
    "cream_cheese": 0.030,
}
TABLE_HALF_EXTENTS_M = (0.50, 0.60)

SKILLS = ("put_on_top", "put_inside", "push_to")
REGIONS = ("left", "middle", "right")

SOURCE_SLOT_RANGES = {
    0: (0.20, -0.395, 0.27, -0.325),
    1: (0.20, -0.155, 0.27, -0.085),
    2: (0.20, 0.085, 0.27, 0.155),
    3: (0.20, 0.325, 0.27, 0.395),
}

TARGET_REGION_RANGES = {
    "left": (-0.20, -0.28, -0.04, -0.14),
    "middle": (-0.20, -0.07, -0.04, 0.07),
    "right": (-0.20, 0.14, -0.04, 0.28),
}

TARGET_REGION_RGBA = (0.10, 0.45, 1.00, 0.25)

TASK_SPEC_FIELDS = [
    "protocol_version",
    "benchmark_name",
    "task_id",
    "tuple",
    "object",
    "object_instance",
    "skill",
    "spatial_region",
    "language",
    "problem_name",
    "scene_type",
    "target_type",
    "target_instance",
    "receiver_region",
    "goal_expression",
    "goal_predicates_json",
    "draft_success_definition",
    "layout_count",
    "bddl_glob",
    "push_no_bilateral_grasp",
    "provisional_push_max_lift_m",
    "push_constraint_status",
    "all_four_objects_present",
    "proxy_benchmark",
    "factor_interpretation",
    "source_slot_ranges_json",
    "target_region_ranges_json",
    "target_region_rgba_json",
    camera_config.OBSERVATION_CAMERA_SPEC_FIELD,
]

LAYOUT_SPEC_FIELDS = [
    "protocol_version",
    "benchmark_name",
    "task_id",
    "layout_id",
    "tuple",
    "object",
    "skill",
    "spatial_region",
    "language",
    "bddl_path",
    "bddl_sha256",
    "object_to_slot_json",
    "goal_expression",
    "goal_predicates_json",
    "receiver_region",
    "source_slot_ranges_json",
    "target_region_ranges_json",
    camera_config.OBSERVATION_CAMERA_SPEC_FIELD,
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate 36 logical LIBERO proxy tasks and four balanced "
            "source-layout BDDL variants per task."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def validate_output_path(path: Path):
    resolved = path.resolve()
    dangerous = {
        Path("/").resolve(),
        Path.home().resolve(),
        Path.cwd().resolve(),
        repo_root().resolve(),
        repo_root().resolve().parent,
    }
    if resolved in dangerous or len(resolved.parts) < 3:
        raise ValueError(
            f"Refusing unsafe output directory: {resolved}"
        )


def validate_existing_output_for_overwrite(path: Path):
    required = [
        path / "task_spec.csv",
        path / "layout_spec.csv",
        path / "bddl",
    ]
    if not all(item.exists() for item in required):
        raise ValueError(
            "Refusing to overwrite a directory that is not a "
            f"recognized generator output: {path}"
        )
    with (path / "task_spec.csv").open(
        newline="",
        encoding="utf-8",
    ) as file:
        first = next(csv.DictReader(file), None)
    if first is None:
        signature = None
    else:
        signature = (
            first.get("protocol_version"),
            first.get("benchmark_name"),
        )
    allowed_signatures = LEGACY_OVERWRITE_SIGNATURES | {
        (PROTOCOL_VERSION, BENCHMARK_NAME),
    }
    if signature not in allowed_signatures:
        raise ValueError(
            "Refusing to overwrite output with a different protocol "
            f"signature: {path}"
        )


def region_block(
    name: str,
    target: str,
    region_bounds: tuple[float, float, float, float] | None,
    rgba: tuple[float, float, float, float] | None = None,
) -> str:
    lines = [
        f"      ({name}",
        f"          (:target {target})",
    ]
    if region_bounds is not None:
        values = " ".join(
            f"{value:.3f}" for value in region_bounds
        )
        lines.extend(
            [
                "          (:ranges (",
                f"              ({values})",
                "            )",
                "          )",
            ]
        )
    if rgba is not None:
        values = " ".join(f"{value:.2f}" for value in rgba)
        lines.append(f"          (:rgba ({values}))")
    lines.append("      )")
    return "\n".join(lines)


def language_for(
    object_name: str,
    skill: str,
    region: str,
) -> str:
    display = OBJECTS[object_name]["display"]
    if skill == "put_on_top":
        return (
            f"Pick up the {display} and place it on the "
            f"plate in the {region} region"
        )
    if skill == "put_inside":
        return (
            f"Pick up the {display} and place it inside the "
            f"basket in the {region} region"
        )
    if skill == "push_to":
        return (
            f"Push the {display} into the "
            f"{region} target region"
        )
    raise ValueError(f"Unknown skill: {skill}")


def balanced_layout(layout_id: int) -> dict[str, int]:
    if not 0 <= layout_id < LAYOUT_COUNT:
        raise ValueError(f"Invalid layout ID: {layout_id}")
    return {
        object_name: (object_index + layout_id) % LAYOUT_COUNT
        for object_index, object_name in enumerate(OBJECTS)
    }


def goal_definition(
    object_name: str,
    skill: str,
    region: str,
) -> dict[str, str]:
    target = OBJECTS[object_name]["instance"]
    receiver_region = f"main_table_target_{region}_region"

    if skill == "put_on_top":
        return {
            "target_type": "plate",
            "target_instance": "plate_1",
            "receiver_region": receiver_region,
            "goal_expression": f"(And (On {target} plate_1))",
            "goal_predicates": [
                ["on", target, "plate_1"],
            ],
            "draft_success_definition": (
                f"On({target},plate_1) AND "
                f"terminal_center_xy(plate_1) in "
                f"{receiver_region}"
            ),
        }
    if skill == "put_inside":
        return {
            "target_type": "basket",
            "target_instance": "basket_1",
            "receiver_region": receiver_region,
            "goal_expression": (
                f"(And (In {target} basket_1_contain_region))"
            ),
            "goal_predicates": [
                ["in", target, "basket_1_contain_region"],
            ],
            "draft_success_definition": (
                f"In({target},basket_1_contain_region) AND "
                f"terminal_center_xy(basket_1) in "
                f"{receiver_region}"
            ),
        }
    if skill == "push_to":
        push_region = f"main_table_target_{region}_region"
        return {
            "target_type": "table_region",
            "target_instance": push_region,
            "receiver_region": push_region,
            "goal_expression": f"(And (On {target} {push_region}))",
            "goal_predicates": [
                ["on", target, push_region],
            ],
            "draft_success_definition": (
                f"On({target},{push_region}) AND "
                "no_bilateral_grasp_during_trajectory AND "
                f"max_lift_from_stabilized_height<="
                f"{PROVISIONAL_PUSH_MAX_LIFT_M:.3f}m "
                "(provisional threshold)"
            ),
        }
    raise ValueError(f"Unknown skill: {skill}")


def object_declarations(skill: str) -> list[str]:
    declarations = [
        f"    {values['instance']} - {object_name}"
        for object_name, values in OBJECTS.items()
    ]
    if skill == "put_on_top":
        declarations.append("    plate_1 - plate")
    elif skill == "put_inside":
        declarations.append("    basket_1 - basket")
    return declarations


def make_bddl(
    object_name: str,
    skill: str,
    region: str,
    layout_id: int,
) -> tuple[str, dict[str, str], dict[str, int]]:
    language = language_for(object_name, skill, region)
    goal = goal_definition(object_name, skill, region)
    layout = balanced_layout(layout_id)

    region_blocks = []
    for slot_id, slot_range in SOURCE_SLOT_RANGES.items():
        region_blocks.append(
            region_block(
                f"source_slot_{slot_id}_region",
                "main_table",
                slot_range,
            )
        )
    for target_region, target_range in TARGET_REGION_RANGES.items():
        region_blocks.append(
            region_block(
                f"target_{target_region}_region",
                "main_table",
                target_range,
                rgba=TARGET_REGION_RGBA,
            )
        )
    if skill == "put_inside":
        region_blocks.append(
            region_block(
                "contain_region",
                "basket_1",
                None,
            )
        )

    init_lines = [
        (
            f"    (On {OBJECTS[name]['instance']} "
            f"main_table_source_slot_{slot_id}_region)"
        )
        for name, slot_id in layout.items()
    ]
    if skill == "put_on_top":
        init_lines.append(
            f"    (On plate_1 {goal['receiver_region']})"
        )
        interest = [
            OBJECTS[object_name]["instance"],
            "plate_1",
        ]
    elif skill == "put_inside":
        init_lines.append(
            f"    (On basket_1 {goal['receiver_region']})"
        )
        interest = [
            OBJECTS[object_name]["instance"],
            "basket_1",
        ]
    elif skill == "push_to":
        interest = [
            OBJECTS[object_name]["instance"],
            goal["receiver_region"],
        ]
    else:
        raise ValueError(f"Unknown skill: {skill}")

    text = "\n".join(
        [
            "(define (problem LIBERO_Tabletop_Manipulation)",
            "  (:domain robosuite)",
            f"  (:language {language})",
            "    (:regions",
            *region_blocks,
            "    )",
            "",
            "  (:fixtures",
            "    main_table - table",
            "  )",
            "",
            "  (:objects",
            *object_declarations(skill),
            "  )",
            "",
            "  (:obj_of_interest",
            *[f"    {item}" for item in interest],
            "  )",
            "",
            "  (:init",
            *init_lines,
            "  )",
            "",
            "  (:goal",
            f"    {goal['goal_expression']}",
            "  )",
            "",
            ")",
            "",
        ]
    )
    return text, goal, layout


def validate_registry():
    import libero.libero.envs.objects  # noqa: F401
    from libero.libero.envs.base_object import OBJECTS_DICT

    required = set(OBJECTS) | {"plate", "basket"}
    missing = sorted(required - set(OBJECTS_DICT))
    if missing:
        raise RuntimeError(
            f"Required LIBERO objects are not registered: {missing}"
        )

    for object_name, expected_radius in SOURCE_SAMPLER_RADII_M.items():
        model = OBJECTS_DICT[object_name](
            name=f"geometry_audit_{object_name}"
        )
        actual_radius = float(model.horizontal_radius)
        if not math.isclose(
            actual_radius,
            expected_radius,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                f"Registered object radius changed for {object_name}: "
                f"{actual_radius} != {expected_radius}"
            )


def validate_region_geometry():
    if sorted(SOURCE_SLOT_RANGES) != list(range(LAYOUT_COUNT)):
        raise ValueError("Source slot IDs must be exactly 0..3")

    rectangles = [
        ("source", str(slot_id), bounds)
        for slot_id, bounds in SOURCE_SLOT_RANGES.items()
    ] + [
        ("target", region, bounds)
        for region, bounds in TARGET_REGION_RANGES.items()
    ]
    for group, name, (x1, y1, x2, y2) in rectangles:
        if not (x1 < x2 and y1 < y2):
            raise ValueError(
                f"Invalid {group} rectangle {name}: "
                f"{(x1, y1, x2, y2)}"
            )

        half_x, half_y = TABLE_HALF_EXTENTS_M
        if not (
            -half_x <= x1 < x2 <= half_x
            and -half_y <= y1 < y2 <= half_y
        ):
            raise ValueError(
                f"{group.title()} rectangle {name} exceeds the "
                f"table bounds: {(x1, y1, x2, y2)}"
            )

    max_source_radius = max(SOURCE_SAMPLER_RADII_M.values())
    for slot_id, (x1, y1, x2, y2) in SOURCE_SLOT_RANGES.items():
        if min(x2 - x1, y2 - y1) <= 2 * max_source_radius:
            raise ValueError(
                f"Source slot {slot_id} is too small for the frozen "
                f"sampler radii: {(x1, y1, x2, y2)}"
            )

    for left_index, left in enumerate(rectangles):
        for right in rectangles[left_index + 1 :]:
            left_x1, left_y1, left_x2, left_y2 = left[2]
            right_x1, right_y1, right_x2, right_y2 = right[2]
            overlaps = (
                max(left_x1, right_x1) < min(left_x2, right_x2)
                and max(left_y1, right_y1) < min(left_y2, right_y2)
            )
            if overlaps:
                raise ValueError(
                    f"Regions overlap: {left[:2]} and {right[:2]}"
                )

    target_centers_y = {
        name: (bounds[1] + bounds[3]) / 2
        for name, bounds in TARGET_REGION_RANGES.items()
    }
    if not (
        target_centers_y["left"]
        < target_centers_y["middle"]
        < target_centers_y["right"]
    ):
        raise ValueError(
            "Target regions must be ordered left-to-right in world y"
        )


def validate_bddl(
    text: str,
    task_row: dict,
    layout: dict[str, int],
):
    if text.count("(") != text.count(")"):
        raise ValueError(
            f"Unbalanced BDDL for task {task_row['task_id']}"
        )
    required_fragments = [
        "(problem LIBERO_Tabletop_Manipulation)",
        f"(:language {task_row['language']})",
        task_row["goal_expression"],
    ]
    for object_name, values in OBJECTS.items():
        required_fragments.extend(
            [
                f"{values['instance']} - {object_name}",
                (
                    f"(On {values['instance']} "
                    f"main_table_source_slot_"
                    f"{layout[object_name]}_region)"
                ),
            ]
        )
    for fragment in required_fragments:
        if fragment not in text:
            raise ValueError(
                f"Task {task_row['task_id']} is missing: {fragment}"
            )


def normalize_predicates(predicates) -> list[tuple[str, ...]]:
    return sorted(
        tuple(str(token).strip().lower() for token in predicate)
        for predicate in predicates
    )


def normalized_text(value) -> str:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    return " ".join(str(value).strip().lower().split())


def validate_parsed_bddl(
    path: Path,
    task_row: dict,
    layout: dict[str, int],
):
    from libero.libero.envs.bddl_utils import robosuite_parse_problem

    parsed = robosuite_parse_problem(str(path))
    if normalized_text(parsed["problem_name"]) != (
        "libero_tabletop_manipulation"
    ):
        raise ValueError("Parsed LIBERO problem name is incorrect")
    if normalized_text(parsed["language_instruction"]) != normalized_text(
        task_row["language"]
    ):
        raise ValueError(
            f"Parsed language mismatch for task {task_row['task_id']}"
        )

    expected_goal = json.loads(task_row["goal_predicates_json"])
    if normalize_predicates(parsed["goal_state"]) != normalize_predicates(
        expected_goal
    ):
        raise ValueError(
            f"Parsed goal mismatch for task {task_row['task_id']}"
        )

    expected_init = [
        [
            "on",
            OBJECTS[object_name]["instance"],
            f"main_table_source_slot_{slot_id}_region",
        ]
        for object_name, slot_id in layout.items()
    ]
    if task_row["skill"] == "put_on_top":
        expected_init.append(
            ["on", "plate_1", task_row["receiver_region"]]
        )
    elif task_row["skill"] == "put_inside":
        expected_init.append(
            ["on", "basket_1", task_row["receiver_region"]]
        )
    if normalize_predicates(parsed["initial_state"]) != normalize_predicates(
        expected_init
    ):
        raise ValueError(
            f"Parsed init mismatch for task {task_row['task_id']}"
        )

    actual_objects = {
        instance
        for instances in parsed["objects"].values()
        for instance in instances
    }
    expected_objects = {
        values["instance"]
        for values in OBJECTS.values()
    }
    if task_row["skill"] == "put_on_top":
        expected_objects.add("plate_1")
    elif task_row["skill"] == "put_inside":
        expected_objects.add("basket_1")
    if actual_objects != expected_objects:
        raise ValueError(
            f"Parsed object mismatch for task {task_row['task_id']}"
        )

    actual_fixtures = {
        instance
        for instances in parsed["fixtures"].values()
        for instance in instances
    }
    if actual_fixtures != {"main_table"}:
        raise ValueError(
            f"Parsed fixture mismatch for task {task_row['task_id']}"
        )

    expected_regions = {
        f"main_table_source_slot_{slot_id}_region"
        for slot_id in SOURCE_SLOT_RANGES
    } | {
        f"main_table_target_{region}_region"
        for region in TARGET_REGION_RANGES
    }
    if task_row["skill"] == "put_inside":
        expected_regions.add("basket_1_contain_region")
    if set(parsed["regions"]) != expected_regions:
        raise ValueError(
            f"Parsed region mismatch for task {task_row['task_id']}"
        )

    target = task_row["object_instance"]
    if task_row["skill"] == "put_on_top":
        expected_interest = {target, "plate_1"}
    elif task_row["skill"] == "put_inside":
        expected_interest = {target, "basket_1"}
    else:
        expected_interest = {target, task_row["receiver_region"]}
    if set(parsed["obj_of_interest"]) != expected_interest:
        raise ValueError(
            "Parsed objects of interest mismatch for task "
            f"{task_row['task_id']}"
        )


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict],
):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_task_rows(rows: list[dict]):
    if len(rows) != 36:
        raise ValueError(f"Expected 36 tasks, found {len(rows)}")
    if [int(row["task_id"]) for row in rows] != list(range(36)):
        raise ValueError("Task IDs are not exactly 0..35")
    for row in rows:
        camera_config.validate_frozen_camera_spec_row(row)
    for field in ["tuple", "language", "bddl_glob"]:
        values = [row[field] for row in rows]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate task field: {field}")
    expected = set(itertools.product(OBJECTS, SKILLS, REGIONS))
    actual = {
        (
            row["object"],
            row["skill"],
            row["spatial_region"],
        )
        for row in rows
    }
    if actual != expected:
        raise ValueError("Task factor coverage is incomplete")

    def count_by(fields: tuple[str, ...]) -> dict[tuple[str, ...], int]:
        counts = {}
        for row in rows:
            key = tuple(str(row[field]) for field in fields)
            counts[key] = counts.get(key, 0) + 1
        return counts

    expected_uniform_counts = {
        ("object",): 9,
        ("skill",): 12,
        ("spatial_region",): 12,
        ("object", "skill"): 3,
        ("object", "spatial_region"): 3,
        ("skill", "spatial_region"): 4,
    }
    for fields, expected_count in expected_uniform_counts.items():
        counts = count_by(fields)
        if set(counts.values()) != {expected_count}:
            raise ValueError(
                f"Unbalanced factor coverage for {fields}: {counts}"
            )


def validate_layout_rows(
    rows: list[dict],
    task_rows: list[dict],
    temporary_bddl_dir: Path,
):
    expected_count = 36 * LAYOUT_COUNT
    if len(rows) != expected_count:
        raise ValueError(
            f"Expected {expected_count} layout rows, found {len(rows)}"
        )
    keys = [
        (int(row["task_id"]), int(row["layout_id"]))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate task/layout key")
    for field in ["bddl_path", "bddl_sha256"]:
        values = [row[field] for row in rows]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate layout field: {field}")

    task_by_id = {
        int(row["task_id"]): row
        for row in task_rows
    }
    layouts_by_task = {
        task_id: []
        for task_id in range(36)
    }
    for row in rows:
        camera_config.validate_frozen_camera_spec_row(row)
        task_id = int(row["task_id"])
        layout_id = int(row["layout_id"])
        layouts_by_task[task_id].append(layout_id)
        task = task_by_id[task_id]
        for field in [
            "tuple",
            "object",
            "skill",
            "spatial_region",
            "language",
            "goal_expression",
            "goal_predicates_json",
            "receiver_region",
            camera_config.OBSERVATION_CAMERA_SPEC_FIELD,
        ]:
            if row[field] != task[field]:
                raise ValueError(
                    f"Task/layout mismatch for task={task_id}, "
                    f"layout={layout_id}, field={field}"
                )
        temporary_path = (
            temporary_bddl_dir / Path(row["bddl_path"]).name
        )
        if not temporary_path.is_file():
            raise FileNotFoundError(
                f"Generated BDDL is missing: {temporary_path}"
            )
        if sha256_file(temporary_path) != row["bddl_sha256"]:
            raise ValueError(
                f"Generated BDDL hash mismatch: {temporary_path}"
            )

    for task_id, layout_ids in layouts_by_task.items():
        if sorted(layout_ids) != list(range(LAYOUT_COUNT)):
            raise ValueError(
                f"Layouts are not exactly 0..3 for task={task_id}: "
                f"{layout_ids}"
            )

    for object_name in OBJECTS:
        for task_id in range(36):
            slots = []
            for row in rows:
                if int(row["task_id"]) != task_id:
                    continue
                mapping = json.loads(row["object_to_slot_json"])
                slots.append(int(mapping[object_name]))
            if sorted(slots) != list(range(LAYOUT_COUNT)):
                raise ValueError(
                    f"Unbalanced source slots for task={task_id}, "
                    f"object={object_name}: {slots}"
                )


def main():
    args = parse_args()
    output_dir = args.output_dir
    validate_output_path(output_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_dir}. "
            "Use --overwrite to rebuild it."
        )
    if output_dir.exists():
        validate_existing_output_for_overwrite(output_dir)

    validate_registry()
    validate_region_geometry()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp.",
            dir=output_dir.parent,
        )
    )
    temporary_bddl_dir = temporary_dir / "bddl"
    temporary_bddl_dir.mkdir(parents=True)

    task_rows = []
    layout_rows = []

    for task_id, (object_name, skill, region) in enumerate(
        itertools.product(OBJECTS, SKILLS, REGIONS)
    ):
        language = language_for(object_name, skill, region)
        goal = goal_definition(object_name, skill, region)
        bddl_glob = (
            output_dir
            / "bddl"
            / f"task_{task_id:03d}_layout_*.bddl"
        )
        task_row = {
            "protocol_version": PROTOCOL_VERSION,
            "benchmark_name": BENCHMARK_NAME,
            "task_id": task_id,
            "tuple": f"{object_name}*{skill}*{region}",
            "object": object_name,
            "object_instance": OBJECTS[object_name]["instance"],
            "skill": skill,
            "spatial_region": region,
            "language": language,
            "problem_name": "LIBERO_Tabletop_Manipulation",
            "scene_type": "tabletop",
            "target_type": goal["target_type"],
            "target_instance": goal["target_instance"],
            "receiver_region": goal["receiver_region"],
            "goal_expression": goal["goal_expression"],
            "goal_predicates_json": json.dumps(
                goal["goal_predicates"],
                separators=(",", ":"),
            ),
            "draft_success_definition": goal[
                "draft_success_definition"
            ],
            "layout_count": LAYOUT_COUNT,
            "bddl_glob": str(bddl_glob),
            "push_no_bilateral_grasp": skill == "push_to",
            "provisional_push_max_lift_m": (
                PROVISIONAL_PUSH_MAX_LIFT_M
                if skill == "push_to"
                else ""
            ),
            "push_constraint_status": (
                "provisional_requires_pilot_calibration"
                if skill == "push_to"
                else "not_applicable"
            ),
            "all_four_objects_present": True,
            "proxy_benchmark": True,
            "factor_interpretation": (
                "object_identity*goal_schema_receiver_type*"
                "destination_location"
            ),
            "source_slot_ranges_json": json.dumps(
                SOURCE_SLOT_RANGES,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "target_region_ranges_json": json.dumps(
                TARGET_REGION_RANGES,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "target_region_rgba_json": json.dumps(
                TARGET_REGION_RGBA,
                separators=(",", ":"),
            ),
            camera_config.OBSERVATION_CAMERA_SPEC_FIELD: (
                camera_config.frozen_camera_spec_json()
            ),
        }
        task_rows.append(task_row)

        for layout_id in range(LAYOUT_COUNT):
            text, actual_goal, layout = make_bddl(
                object_name,
                skill,
                region,
                layout_id,
            )
            if actual_goal != goal:
                raise RuntimeError("Goal generation is inconsistent")
            validate_bddl(text, task_row, layout)

            file_name = (
                f"task_{task_id:03d}_layout_{layout_id}_"
                f"{object_name}_{skill}_{region}.bddl"
            )
            temporary_path = temporary_bddl_dir / file_name
            final_path = output_dir / "bddl" / file_name
            temporary_path.write_text(text, encoding="utf-8")
            validate_parsed_bddl(
                temporary_path,
                task_row,
                layout,
            )

            layout_rows.append(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "benchmark_name": BENCHMARK_NAME,
                    "task_id": task_id,
                    "layout_id": layout_id,
                    "tuple": task_row["tuple"],
                    "object": object_name,
                    "skill": skill,
                    "spatial_region": region,
                    "language": language,
                    "bddl_path": str(final_path),
                    "bddl_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                    "object_to_slot_json": json.dumps(
                        layout,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "goal_expression": goal["goal_expression"],
                    "goal_predicates_json": json.dumps(
                        goal["goal_predicates"],
                        separators=(",", ":"),
                    ),
                    "receiver_region": goal["receiver_region"],
                    "source_slot_ranges_json": json.dumps(
                        SOURCE_SLOT_RANGES,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "target_region_ranges_json": json.dumps(
                        TARGET_REGION_RANGES,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    camera_config.OBSERVATION_CAMERA_SPEC_FIELD: (
                        camera_config.frozen_camera_spec_json()
                    ),
                }
            )

    validate_task_rows(task_rows)
    validate_layout_rows(
        layout_rows,
        task_rows,
        temporary_bddl_dir,
    )
    write_csv(
        temporary_dir / "task_spec.csv",
        TASK_SPEC_FIELDS,
        task_rows,
    )
    write_csv(
        temporary_dir / "layout_spec.csv",
        LAYOUT_SPEC_FIELDS,
        layout_rows,
    )

    backup_dir = output_dir.with_name(
        f".{output_dir.name}.backup.{os.getpid()}"
    )
    if backup_dir.exists():
        raise FileExistsError(
            f"Stale backup directory exists: {backup_dir}"
        )
    if output_dir.exists():
        output_dir.replace(backup_dir)
    try:
        temporary_dir.replace(output_dir)
    except Exception:
        if backup_dir.exists() and not output_dir.exists():
            backup_dir.replace(output_dir)
        raise
    else:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)

    print("=" * 80)
    print("Generated LIBERO native-object proxy benchmark draft")
    print("protocol:", PROTOCOL_VERSION)
    print("logical tasks:", len(task_rows))
    print("layouts per task:", LAYOUT_COUNT)
    print("BDDL files:", len(layout_rows))
    print("objects:", list(OBJECTS))
    print("skills:", list(SKILLS))
    print("regions:", list(REGIONS))
    print(
        "agentview vertical FOV:",
        camera_config.AGENTVIEW_FOVY_DEG,
    )
    print("task spec:", output_dir / "task_spec.csv")
    print("layout spec:", output_dir / "layout_spec.csv")
    print("BDDL directory:", output_dir / "bddl")
    print(
        "IMPORTANT: this remains a draft until reset, oracle-goal, "
        "and one-success-trajectory gates pass."
    )
    print(
        "Push trajectory threshold is provisional and must be "
        "calibrated before formal evaluation."
    )


if __name__ == "__main__":
    main()
