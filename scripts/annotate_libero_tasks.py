from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/libero_tasks.csv")
OUTPUT_PATH = Path("data/runnable_task_combinations.csv")


def infer_object(language: str) -> str:
    language = language.lower()

    candidates = [
        "black bowl",
        "bowl",
        "plate",
        "mug",
        "cream cheese",
        "wine bottle",
        "moka pot",
        "book",
        "alphabet soup",
        "tomato sauce",
        "butter",
        "milk",
        "orange juice",
        "ketchup",
        "bbq sauce",
        "salad dressing",
        "chocolate pudding",
    ]

    for obj in candidates:
        if obj in language:
            return obj.replace(" ", "_")

    return "unknown"


def infer_skill(language: str) -> str:
    language = language.lower()

    if "push" in language:
        return "push_to"

    if "on top of" in language:
        return "put_on_top"

    if "inside" in language or " in the " in language:
        return "put_inside"

    if "place it on" in language or " put " in language or "place" in language:
        return "pick_and_place"

    if "open" in language:
        return "open"

    if "turn on" in language:
        return "turn_on"

    return "unknown"


def infer_spatial(language: str) -> str:
    language = language.lower()

    spatial_patterns = {
        "between": "between",
        "next to": "next_to",
        "table center": "table_center",
        "on the stove": "on_stove",
        "front of the stove": "front_of_stove",
        "top drawer": "top_drawer",
        "bottom drawer": "bottom_drawer",
        "wooden cabinet": "on_or_in_cabinet",
        "on top of the cabinet": "on_top_of_cabinet",
        "on the cabinet": "on_cabinet",
        "on the plate": "on_plate",
        "in the basket": "in_basket",
        "in the bowl": "in_bowl",
        "in the microwave": "in_microwave",
        "on the rack": "on_rack",
    }

    for pattern, label in spatial_patterns.items():
        if pattern in language:
            return label

    return "unknown"


def is_simple_enough(row: pd.Series) -> bool:
    language = row["language"].lower()
    suite = row["suite"]

    if suite == "libero_10":
        return False

    if "both" in language:
        return False

    if "and close it" in language:
        return False

    return True


def main():
    df = pd.read_csv(INPUT_PATH)

    rows = []

    for _, row in df.iterrows():
        obj = infer_object(row["language"])
        skill = infer_skill(row["language"])
        spatial = infer_spatial(row["language"])
        simple_enough = is_simple_enough(row)

        is_runnable_candidate = (
            obj != "unknown"
            and skill != "unknown"
            and spatial != "unknown"
            and simple_enough
        )

        rows.append(
            {
                "suite": row["suite"],
                "libero_task_id": row["task_id"],
                "name": row["name"],
                "language": row["language"],
                "bddl_file": row["bddl_file"],
                "object": obj,
                "skill": skill,
                "spatial_relation": spatial,
                "is_runnable_candidate": is_runnable_candidate,
            }
        )

    output_df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved annotated LIBERO tasks to {OUTPUT_PATH}")
    print("\nRunnable candidates:")
    candidates = output_df[output_df["is_runnable_candidate"]]
    print(candidates[["suite", "libero_task_id", "object", "skill", "spatial_relation", "language"]])

    print("\nCounts by suite:")
    print(output_df.groupby(["suite", "is_runnable_candidate"]).size())


if __name__ == "__main__":
    main()