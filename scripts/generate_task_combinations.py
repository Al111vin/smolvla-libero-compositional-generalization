import itertools
import pandas as pd
from pathlib import Path


OBJECTS = [
    "mug",
    "bowl",
    "plate",
    "book",
]

SKILLS = [
    "pick",
    "place",
    "open",
    "close",
]

SPATIAL_REGIONS = [
    "left",
    "right",
    "front",
    "back",
]


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
            }
        )

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "task_combinations.csv"
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} task combinations to {output_path}")
    print(df.head())


if __name__ == "__main__":
    main()