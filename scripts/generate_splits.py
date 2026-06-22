from pathlib import Path

import pandas as pd


COVERAGES = [0.25, 0.50, 0.75, 1.00]
INPUT_PATH = Path("data/task_combinations.csv")
OUTPUT_DIR = Path("data/splits")
RANDOM_SEED = 42


def check_value_seen(train_df: pd.DataFrame, full_df: pd.DataFrame) -> None:
    for column in ["object", "skill", "spatial_region"]:
        full_values = set(full_df[column])
        train_values = set(train_df[column])
        missing = full_values - train_values

        print(f"{column} train counts:")
        print(train_df[column].value_counts().sort_index())

        if missing:
            raise ValueError(f"Value-seen failed for {column}. Missing: {missing}")


def check_tuple_unseen(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_tuples = set(train_df["tuple"])
    test_tuples = set(test_df["tuple"])

    overlap = train_tuples & test_tuples
    print(f"train-test tuple overlap: {len(overlap)}")

    if overlap:
        raise ValueError(f"Tuple-unseen failed. Overlap examples: {list(overlap)[:5]}")


def make_split(df: pd.DataFrame, coverage: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if coverage == 1.00:
        train_df = df.copy()
        test_df = df.iloc[0:0].copy()
        return train_df, test_df

    train_size = int(round(len(df) * coverage))

    for seed in range(RANDOM_SEED, RANDOM_SEED + 1000):
        train_df = df.sample(n=train_size, random_state=seed).sort_values("task_id")
        test_df = df.drop(train_df.index).sort_values("task_id")

        try:
            check_value_seen(train_df, df)
            check_tuple_unseen(train_df, test_df)
            return train_df, test_df
        except ValueError:
            continue

    raise RuntimeError(f"Could not create valid split for coverage={coverage}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    print(f"Loaded {len(df)} task combinations from {INPUT_PATH}")

    for coverage in COVERAGES:
        print("\n" + "=" * 60)
        print(f"Generating split for coverage={coverage:.0%}")

        train_df, test_df = make_split(df, coverage)

        train_path = OUTPUT_DIR / f"coverage_{int(coverage * 100)}_train.csv"
        test_path = OUTPUT_DIR / f"coverage_{int(coverage * 100)}_test.csv"

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"Saved train split: {train_path} ({len(train_df)} rows)")
        print(f"Saved test split:  {test_path} ({len(test_df)} rows)")

    print("\nAll splits generated successfully.")


if __name__ == "__main__":
    main()