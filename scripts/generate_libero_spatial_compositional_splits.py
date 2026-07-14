from __future__ import annotations

from pathlib import Path

import pandas as pd

from libero.libero import benchmark


SUITE_NAME = "libero_spatial"
DATASET_DIR = Path("datasets/libero/datasets/libero_spatial")
OUTPUT_DIR = Path("data/splits")

FACTORS = {
    0: ("between", "plate_and_ramekin"),
    1: ("next_to", "ramekin"),
    2: ("at_center", "table"),
    3: ("on", "cookie_box"),
    4: ("inside", "top_drawer_of_wooden_cabinet"),
    5: ("on", "ramekin"),
    6: ("next_to", "cookie_box"),
    7: ("on", "stove"),
    8: ("next_to", "plate"),
    9: ("on", "wooden_cabinet"),
}

FOLDS = {
    "fold_a": {
        "train": [0, 1, 2, 3, 4, 7, 8, 9],
        "test": [5, 6],
    },
    "fold_b": {
        "train": [0, 2, 4, 5, 6, 7, 8, 9],
        "test": [1, 3],
    },
}


def validate_fold(manifest, train_ids, test_ids):
    all_ids = set(manifest["task_id"])
    train_ids = set(train_ids)
    test_ids = set(test_ids)

    if train_ids & test_ids:
        raise ValueError("Train/test task overlap")

    if train_ids | test_ids != all_ids:
        raise ValueError("Train/test tasks do not partition the suite")

    train = manifest[
        manifest["task_id"].isin(train_ids)
    ].copy()
    test = manifest[
        manifest["task_id"].isin(test_ids)
    ].copy()

    train_relations = set(train["source_relation"])
    train_references = set(train["source_reference"])
    train_compositions = set(train["composition"])

    for row in test.itertuples():
        if row.source_relation not in train_relations:
            raise ValueError(
                f"Unseen relation in test: {row.source_relation}"
            )

        if row.source_reference not in train_references:
            raise ValueError(
                f"Unseen reference in test: {row.source_reference}"
            )

        if row.composition in train_compositions:
            raise ValueError(
                f"Seen composition leaked into test: {row.composition}"
            )

    return train, test


def main():
    suite = benchmark.get_benchmark(SUITE_NAME)()
    rows = []

    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        relation, reference = FACTORS[task_id]
        hdf5_path = (
            DATASET_DIR / f"{task.name}_demo.hdf5"
        )

        if not hdf5_path.exists():
            raise FileNotFoundError(hdf5_path)

        rows.append({
            "suite": SUITE_NAME,
            "task_id": task_id,
            "task_name": task.name,
            "language": task.language,
            "source_relation": relation,
            "source_reference": reference,
            "composition": f"{relation}__{reference}",
            "hdf5_path": str(hdf5_path),
        })

    manifest = pd.DataFrame(rows).sort_values("task_id")

    Path("data").mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(
        "data/libero_spatial_task_factors.csv"
    )
    manifest.to_csv(
        manifest_path,
        index=False,
        lineterminator="\n",
    )
    print("Saved:", manifest_path)

    for fold_name, split in FOLDS.items():
        train, test = validate_fold(
            manifest,
            split["train"],
            split["test"],
        )

        train = train.assign(fold=fold_name, split="train")
        test = test.assign(fold=fold_name, split="test")

        train_path = (
            OUTPUT_DIR
            / f"libero_spatial_compositional_{fold_name}_train.csv"
        )
        test_path = (
            OUTPUT_DIR
            / f"libero_spatial_compositional_{fold_name}_test.csv"
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
        print(fold_name)
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


if __name__ == "__main__":
    main()
