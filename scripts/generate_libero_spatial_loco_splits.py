from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/libero_spatial_task_factors.csv")
OUTPUT_DIR = Path("data/splits")

EXCLUSIONS = {
    1: (
        "visual_leakage: distractor bowl appears in "
        "main_table_next_to_box_region"
    ),
    5: (
        "runtime_instability: target bowl is not On "
        "the ramekin after standard stabilization"
    ),
}

ELIGIBLE_IDS = [0, 2, 3, 4, 6, 7, 8, 9]

SPLITS = {
    "task6": {
        "train": [0, 2, 3, 4, 7, 8, 9],
        "test": [6],
    },
    "task3": {
        "train": [0, 2, 4, 6, 7, 8, 9],
        "test": [3],
    },
}


def validate_split(table, name, train_ids, test_ids):
    train_ids = set(train_ids)
    test_ids = set(test_ids)
    eligible_ids = set(ELIGIBLE_IDS)

    if train_ids & test_ids:
        raise ValueError(f"{name}: train/test overlap")

    if train_ids | test_ids != eligible_ids:
        raise ValueError(
            f"{name}: split does not partition eligible pool"
        )

    if set(EXCLUSIONS) & (train_ids | test_ids):
        raise ValueError(
            f"{name}: excluded task entered split"
        )

    train = table[
        table["task_id"].isin(train_ids)
    ].copy().sort_values("task_id")

    test = table[
        table["task_id"].isin(test_ids)
    ].copy().sort_values("task_id")

    train_relations = set(train["source_relation"])
    train_references = set(train["source_reference"])
    train_compositions = set(train["composition"])

    for row in test.itertuples():
        if row.source_relation not in train_relations:
            raise ValueError(
                f"{name}: relation not seen in training: "
                f"{row.source_relation}"
            )

        if row.source_reference not in train_references:
            raise ValueError(
                f"{name}: reference not seen in training: "
                f"{row.source_reference}"
            )

        if row.composition in train_compositions:
            raise ValueError(
                f"{name}: test composition appears in training: "
                f"{row.composition}"
            )

    return train, test


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    table = pd.read_csv(INPUT_PATH)

    required = {
        "task_id",
        "source_relation",
        "source_reference",
        "composition",
    }

    missing = required - set(table.columns)

    if missing:
        raise ValueError(
            f"Factor table is missing columns: {sorted(missing)}"
        )

    if set(table["task_id"]) != set(range(10)):
        raise ValueError("Expected task IDs 0 through 9.")

    table["eligible_for_primary_loco"] = (
        table["task_id"].isin(ELIGIBLE_IDS)
    )
    table["exclusion_reason"] = (
        table["task_id"].map(EXCLUSIONS).fillna("")
    )

    table.to_csv(
        INPUT_PATH,
        index=False,
        lineterminator="\n",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    excluded = table[
        table["task_id"].isin(EXCLUSIONS)
    ].copy().sort_values("task_id")

    excluded_path = (
        OUTPUT_DIR
        / "libero_spatial_loco_excluded_tasks.csv"
    )

    excluded.to_csv(
        excluded_path,
        index=False,
        lineterminator="\n",
    )

    print("Eligible IDs:", ELIGIBLE_IDS)
    print("Excluded IDs:", sorted(EXCLUSIONS))
    print("Saved:", excluded_path)

    for name, split in SPLITS.items():
        train, test = validate_split(
            table,
            name,
            split["train"],
            split["test"],
        )

        train_path = (
            OUTPUT_DIR
            / f"libero_spatial_loco_{name}_train.csv"
        )
        test_path = (
            OUTPUT_DIR
            / f"libero_spatial_loco_{name}_test.csv"
        )

        train.to_csv(
            train_path,
            index=False,
            lineterminator="\n",
        )
        test.to_csv(
            test_path,
            index=False,
            lineterminator="\n",
        )

        print("=" * 80)
        print(name)
        print("train IDs:", train["task_id"].tolist())
        print("test IDs:", test["task_id"].tolist())
        print(
            test[
                [
                    "task_id",
                    "source_relation",
                    "source_reference",
                    "composition",
                ]
            ].to_string(index=False)
        )
        print("Saved:", train_path)
        print("Saved:", test_path)


if __name__ == "__main__":
    main()
