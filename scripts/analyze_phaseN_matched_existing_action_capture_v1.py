import csv, glob, json, os

POS = "/root/smolvla-eval-prep/results/benchmark_v1_recovered_historical_control_task0_5init_20260929"
NEG = "/root/smolvla-eval-prep/results/benchmark_v1_joint32_terminal_window_weighted_5init_20260929/checkpoint_090000"

def summarize(path):
    rows = list(csv.DictReader(open(path)))
    cols = [f"applied_action_{i}" for i in range(7)]
    vals = [[float(row[c]) for c in cols] for row in rows]
    tail = vals[max(0, len(vals) - 25):]
    gripper = [row[6] for row in tail]
    mean = sum(gripper) / len(gripper)
    return {
        "path": path,
        "steps": len(vals),
        "terminal_gripper_mean": mean,
        "terminal_gripper_std": (sum((x - mean) ** 2 for x in gripper) / len(gripper)) ** 0.5,
        "terminal_gripper_min": min(gripper),
        "terminal_gripper_max": max(gripper),
        "sign_changes": sum(gripper[i] * gripper[i - 1] < 0 for i in range(1, len(gripper))),
    }

def aggregate(items):
    return {
        "n": len(items),
        "terminal_gripper_mean": sum(x["terminal_gripper_mean"] for x in items) / len(items),
        "terminal_gripper_std_mean": sum(x["terminal_gripper_std"] for x in items) / len(items),
        "sign_change_count_total": sum(x["sign_changes"] for x in items),
        "steps_mean": sum(x["steps"] for x in items) / len(items),
    }

def main(output_dir):
    positive = [summarize(p) for p in sorted(glob.glob(POS + "/init_*/**/*_actions.csv", recursive=True))]
    negative = [summarize(p) for p in sorted(glob.glob(NEG + "/init_*/**/*_actions.csv", recursive=True))]
    result = {
        "schema": "phaseN_matched_existing_action_capture_v1",
        "protocol": "eval_v3_task0_n25_benchmark_init",
        "positive": aggregate(positive),
        "negative": aggregate(negative),
        "per_positive": positive,
        "per_negative": negative,
        "paired_init_note": "Init indices overlap, but checkpoints and model families differ; diagnostic only, not causal proof.",
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"positive": result["positive"], "negative": result["negative"]}, indent=2))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    main(parser.parse_args().output_dir)
