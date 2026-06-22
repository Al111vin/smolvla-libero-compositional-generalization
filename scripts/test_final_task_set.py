from pathlib import Path

import pandas as pd
from libero.libero import benchmark


FINAL_TASK_PATH = Path("data/final_task_set.csv")


def main():
    df = pd.read_csv(FINAL_TASK_PATH)

    print(f"Loaded {len(df)} final tasks from {FINAL_TASK_PATH}")

    for _, row in df.iterrows():
        suite = row["suite"]
        task_id = int(row["libero_task_id"])
        bddl_file = row["bddl_file"]

        bench_cls = benchmark.get_benchmark(suite)
        bench = bench_cls()
        task = bench.get_task(task_id)

        print("=" * 80)
        print("suite:", suite)
        print("task_id:", task_id)
        print("language:", task.language)
        print("expected bddl:", bddl_file)
        print("actual bddl:  ", task.bddl_file)

        if task.bddl_file != bddl_file:
            raise ValueError(
                f"BDDL mismatch for {suite} task {task_id}: "
                f"{task.bddl_file} != {bddl_file}"
            )

    print("\nFinal task set check passed.")


if __name__ == "__main__":
    main()