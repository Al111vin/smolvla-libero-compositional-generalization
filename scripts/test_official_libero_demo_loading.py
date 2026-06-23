from pathlib import Path

import pandas as pd

from libero.libero.utils.dataset_utils import get_dataset_info


FINAL_TASK_PATH = Path("data/final_task_set.csv")
DATASET_DIR = Path("datasets/libero/datasets/libero_spatial")


def main():
    df = pd.read_csv(FINAL_TASK_PATH)

    print("=" * 80)
    print("Official LIBERO demo loading test")
    print("final task count:", len(df))
    print("dataset dir:", DATASET_DIR)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    demo_files = sorted(DATASET_DIR.glob("*.hdf5"))
    print("found hdf5 files:", len(demo_files))

    if len(demo_files) == 0:
        raise FileNotFoundError(f"No hdf5 files found in {DATASET_DIR}")

    for _, row in df.iterrows():
        task_id = int(row["libero_task_id"])
        bddl_file = row["bddl_file"]
        expected_demo_name = bddl_file.replace(".bddl", "_demo.hdf5")
        expected_demo_path = DATASET_DIR / expected_demo_name

        print("=" * 80)
        print("task_id:", task_id)
        print("language:", row["language"])
        print("expected demo:", expected_demo_path)

        if not expected_demo_path.exists():
            raise FileNotFoundError(f"Missing demo file: {expected_demo_path}")

        get_dataset_info(str(expected_demo_path), verbose=False)

    print("=" * 80)
    print("Official LIBERO demo loading test passed.")


if __name__ == "__main__":
    main()