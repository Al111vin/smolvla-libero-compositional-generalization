import csv, glob, json, os

def read_one(path):
    rows = list(csv.DictReader(open(path)))
    summary_path = path.replace("_actions.csv", "_summary.csv")
    summary_rows = list(csv.DictReader(open(summary_path)))
    success = summary_rows and summary_rows[0].get("success", "False") == "True"
    action = [[float(r[f"applied_action_{i}"]) for i in range(7)] for r in rows]
    state = [[float(r[f"state_{i}"]) for i in range(15)] for r in rows]
    g = [x[6] for x in action]
    mean = sum(g) / len(g)
    return {
        "path": path,
        "steps": len(rows),
        "terminal_gripper_mean": sum(g[-25:]) / min(25, len(g)),
        "terminal_gripper_std": (sum((x - mean) ** 2 for x in g) / len(g)) ** 0.5,
        "gripper_sign_changes": sum(g[i] * g[i-1] < 0 for i in range(1, len(g))),
        "state_dim": len(state[0]) if state else 0,
        "action_dim": len(action[0]) if action else 0,
        "state_rows": len(state),
        "success": success,
    }

def main(root, output):
    groups = {}
    for group in ("positive", "negative"):
        groups[group] = [read_one(p) for p in sorted(glob.glob(f"{root}/{group}/init_*/**/*_actions.csv", recursive=True))]
    result = {
        "schema": "phaseN_matched_state_action_capture_v1",
        "protocol": "eval_v3_task0_n25_benchmark_init",
        "positive": groups["positive"],
        "negative": groups["negative"],
        "positive_success_count": sum(x["success"] for x in groups["positive"]),
        "negative_success_count": sum(x["success"] for x in groups["negative"]),
        "interpretation": "Diagnostic only; positive and negative checkpoints differ, so this is not causal evidence.",
    }
    os.makedirs(os.path.dirname(output), exist_ok=True)
    json.dump(result, open(output, "w"), indent=2)
    print(json.dumps({"positive_n": len(groups["positive"]), "negative_n": len(groups["negative"]), "positive_success_count": result["positive_success_count"], "negative_success_count": result["negative_success_count"]}, indent=2))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    main(a.root, a.output)
