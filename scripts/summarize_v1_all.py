from pathlib import Path
import csv


csv_files = [
    Path("results/eval_v1_spatial_10tasks.csv"),
    Path("results/eval_v1_object_10tasks.csv"),
    Path("results/eval_v1_goal_10tasks.csv"),
]

summary_path = Path("results/eval_v1_summary_all.txt")


def load_rows(csv_path):
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def summarize(rows):
    num_tasks = len(rows)
    success_count = sum(1 for r in rows if r["success"].lower() == "true")
    success_rate = success_count / num_tasks if num_tasks > 0 else 0.0
    avg_reward = sum(float(r["total_reward"]) for r in rows) / num_tasks if num_tasks > 0 else 0.0
    avg_steps = sum(float(r["steps"]) for r in rows) / num_tasks if num_tasks > 0 else 0.0

    return {
        "num_tasks": num_tasks,
        "success_count": success_count,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
    }


all_rows = []
sections = []

for csv_path in csv_files:
    rows = load_rows(csv_path)
    all_rows.extend(rows)

    suite_name = rows[0]["suite"] if rows else csv_path.stem
    stats = summarize(rows)

    sections.append(
        f"""{suite_name}
{'-' * len(suite_name)}
Tasks: {stats['num_tasks']}
Success: {stats['success_count']} / {stats['num_tasks']}
Success rate: {stats['success_rate']:.4f}
Average reward: {stats['avg_reward']:.4f}
Average steps: {stats['avg_steps']:.2f}
"""
    )


overall = summarize(all_rows)

summary = f"""V1 Evaluation Summary
=====================

{chr(10).join(sections)}
Overall
-------
Tasks: {overall['num_tasks']}
Success: {overall['success_count']} / {overall['num_tasks']}
Success rate: {overall['success_rate']:.4f}
Average reward: {overall['avg_reward']:.4f}
Average steps: {overall['avg_steps']:.2f}

Interpretation:
The V1 SmolVLA evaluation pipeline is functional across LIBERO-Spatial, LIBERO-Object, and LIBERO-Goal.
However, the current V1 policy does not yet solve these tasks.
This result should be treated as the baseline evaluation before improving training, data coverage, action scaling, and compositional generalization.
"""

summary_path.parent.mkdir(parents=True, exist_ok=True)

with open(summary_path, "w") as f:
    f.write(summary)

print(summary)
print(f"Saved summary to: {summary_path}")
