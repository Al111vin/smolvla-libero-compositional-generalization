from pathlib import Path

import pandas as pd
from libero.libero import benchmark


SUITES = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
]


def main():
    rows = []

    for suite_name in SUITES:
        bench_cls = benchmark.get_benchmark(suite_name)
        bench = bench_cls()

        for task_id in range(bench.n_tasks):
            task = bench.get_task(task_id)

            rows.append(
                {
                    "suite": suite_name,
                    "task_id": task_id,
                    "name": task.name,
                    "language": task.language,
                    "bddl_file": task.bddl_file,
                }
            )

    output_path = Path("data/libero_tasks.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} LIBERO tasks to {output_path}")
    print(df.head(20))


if __name__ == "__main__":
    main()