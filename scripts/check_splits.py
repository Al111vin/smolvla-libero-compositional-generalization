from pathlib import Path

import pandas as pd


SPLIT_DIR = Path("data/splits")
COVERAGES = [25, 50, 75, 100]
FACTOR_COLUMNS = ["object", "skill", "spatial_region"]


def check_one_split(coverage: int) -> None:
    train_path = SPLIT_DIR / f"coverage_{coverage}_train.csv"
    test_path = SPLIT_DIR / f"coverage_{coverage}_test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("\n" + "=" * 60)
    print(f"Checking coverage {coverage}%")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows:  {len(test_df)}")

    for column in FACTOR_COLUMNS:
        print(f"\n{column} counts in train:")
        print(train_df[column].value_counts().sort_index())

        if train_df[column].nunique() == 0:
            raise ValueError(f"No values found for {column} in train split.")

    train_tuples = set(train_df["tuple"])
    test_tuples = set(test_df["tuple"])
    overlap = train_tuples & test_tuples

    print(f"\ntrain-test tuple overlap: {len(overlap)}")

    if overlap:
        raise ValueError(f"Leakage detected in coverage {coverage}%: {list(overlap)[:5]}")

    print(f"Coverage {coverage}% leakage check passed.")


def main():
    for coverage in COVERAGES:
        check_one_split(coverage)

    print("\nAll split leakage checks passed.")


if __name__ == "__main__":
    main()