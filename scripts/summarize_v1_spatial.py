from pathlib import Path
import csv

csv_path = Path("results/eval_v1_spatial_10tasks.csv")
summary_path = Path("results/eval_v1_spatial_summary.txt")

rows = []

with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

num_tasks = len(rows)
success_count = sum(1 for r in rows if r["success"].lower() == "true")
success_rate = success_count / num_tasks if num_tasks > 0 else 0.0

avg_reward = sum(float(r["total_reward"]) for r in rows) / num_tasks if num_tasks > 0 else 0.0
avg_steps = sum(float(r["steps"]) for r in rows) / num_tasks if num_tasks > 0 else 0.0

summary = f"""V1 LIBERO-Spatial Evaluation Summary
===================================

Suite: libero_spatial
Number of tasks: {num_tasks}
Success count: {success_count} / {num_tasks}
Success rate: {success_rate:.4f}
Average reward: {avg_reward:.4f}
Average rollout length: {avg_steps:.2f} steps

Interpretation:
The V1 SmolVLA policy was successfully evaluated on all 10 LIBERO-Spatial tasks.
The evaluation pipeline is functional, but the current V1 policy did not solve any task in this evaluation.
"""

summary_path.parent.mkdir(parents=True, exist_ok=True)

with open(summary_path, "w") as f:
    f.write(summary)

print(summary)
print(f"Saved summary to: {summary_path}")
