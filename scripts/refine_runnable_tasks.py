from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/runnable_task_combinations.csv")
OUTPUT_PATH = Path("data/refined_runnable_tasks.csv")


def infer_factor_focus(suite: str) -> str:
    if suite == "libero_spatial":
        return "spatial"
    if suite == "libero_object":
        return "object"
    if suite == "libero_goal":
        return "skill"
    return "mixed"


def infer_target_object(language: str) -> str:
    language = language.lower()

    targets = [
        "plate",
        "basket",
        "stove",
        "cabinet",
        "drawer",
        "bowl",
        "rack",
        "microwave",
    ]

    for target in targets:
        if f"on the {target}" in language:
            return target
        if f"on top of the {target}" in language:
            return target
        if f"in the {target}" in language:
            return target
        if f"inside the {target}" in language:
            return target

    return "unknown"


def infer_source_relation(language: str, suite: str) -> str:
    language = language.lower()

    if suite != "libero_spatial":
        return "none"

    source_patterns = {
        "between the plate and the ramekin": "between_plate_and_ramekin",
        "next to the ramekin": "next_to_ramekin",
        "from table center": "table_center",
        "on the cookie box": "on_cookie_box",
        "on the ramekin": "on_ramekin",
        "next to the cookie box": "next_to_cookie_box",
        "on the stove": "on_stove",
        "next to the plate": "next_to_plate",
        "on the wooden cabinet": "on_wooden_cabinet",
        "in the top drawer": "in_top_drawer",
    }

    for pattern, label in source_patterns.items():
        if pattern in language:
            return label

    return "unknown"


def is_final_selected(row: pd.Series) -> bool:
    if not bool(row["is_runnable_candidate"]):
        return False

    if row["suite"] == "libero_10":
        return False

    if row["object"] == "unknown":
        return False

    if row["skill"] == "unknown":
        return False

    if row["target_object"] == "unknown":
        return False

    if row["factor_focus"] == "spatial" and row["source_relation"] == "unknown":
        return False

    return True


def main():
    df = pd.read_csv(INPUT_PATH)

    refined_rows = []

    for _, row in df.iterrows():
        language = row["language"]
        suite = row["suite"]

        factor_focus = infer_factor_focus(suite)
        target_object = infer_target_object(language)
        source_relation = infer_source_relation(language, suite)

        refined_row = row.to_dict()
        refined_row["factor_focus"] = factor_focus
        refined_row["source_relation"] = source_relation
        refined_row["target_object"] = target_object

        refined_rows.append(refined_row)

    refined_df = pd.DataFrame(refined_rows)
    refined_df["final_selected"] = refined_df.apply(is_final_selected, axis=1)

    refined_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved refined runnable tasks to {OUTPUT_PATH}")

    print("\nSelected tasks by factor focus:")
    print(refined_df.groupby(["factor_focus", "final_selected"]).size())

    print("\nFinal selected tasks:")
    selected = refined_df[refined_df["final_selected"]]
    print(
        selected[
            [
                "suite",
                "libero_task_id",
                "object",
                "skill",
                "source_relation",
                "target_object",
                "factor_focus",
                "language",
            ]
        ]
    )


if __name__ == "__main__":
    main()