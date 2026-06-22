from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/refined_runnable_tasks.csv")
OUTPUT_PATH = Path("data/final_task_set.csv")


def main():
    df = pd.read_csv(INPUT_PATH)

    final_df = df[
        (df["suite"] == "libero_spatial")
        & (df["final_selected"] == True)
    ].copy()

    final_df = final_df.sort_values(["suite", "libero_task_id"])

    final_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(final_df)} final tasks to {OUTPUT_PATH}")
    print(
        final_df[
            [
                "suite",
                "libero_task_id",
                "object",
                "skill",
                "source_relation",
                "target_object",
                "factor_focus",
                "language",
                "bddl_file",
            ]
        ]
    )


if __name__ == "__main__":
    main()