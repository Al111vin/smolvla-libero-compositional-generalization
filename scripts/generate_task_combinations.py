import itertools
from pathlib import Path

import pandas as pd


OBJECTS = [
    "red_bowl",
    "white_plate",
    "black_box",
    "yellow_cup",
]

SKILLS = [
    "put_on_top",
    "put_inside",
    "push_to",
]

SPATIAL_REGIONS = [
    "left",
    "middle",
    "right",
]


def make_instruction(obj: str, skill: str, spatial: str) -> str:
    object_text = obj.replace("_", " ")
    spatial_text = spatial.replace("_", " ")

    if skill == "put_on_top":
        return f"put the {object_text} on top of the target object in the {spatial_text} region"

    if skill == "put_inside":
        return f"put the {object_text} inside the container in the {spatial_text} region"

    if skill == "push_to":
        return f"push the {object_text} to the {spatial_text} region"

    raise ValueError(f"Unknown skill: {skill}")


def main():
    rows = []

    for task_id, (obj, skill, spatial) in enumerate(
        itertools.product(OBJECTS, SKILLS, SPATIAL_REGIONS)
    ):
        rows.append(
            {
                "task_id": task_id,
                "object": obj,
                "skill": skill,
                "spatial_region": spatial,
                "tuple": f"{obj}_{skill}_{spatial}",
                "instruction": make_instruction(obj, skill, spatial),
            }
        )

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "task_combinations.csv"

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} task combinations to {output_path}")
    print(df.head(10))


if __name__ == "__main__":
    main()