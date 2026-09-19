#!/usr/bin/env python3
"""Generate the frozen Fold 01 train/eval manifest for the 32-task dataset.

This is a draft protocol generator. It performs no training or evaluation.
"""
import argparse
import csv
import json
from pathlib import Path

import pyarrow.parquet as pq


HELDOUT_TASK = "pick up the alphabet soup and place it inside the basket in the middle region"
SEEN_CONTROLS = [
    "pick up the alphabet soup and place it inside the basket in the left region",
    "pick up the alphabet soup and place it inside the basket in the right region",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    ep_path = args.dataset_root / "meta/episodes/chunk-000/file-000.parquet"
    rows = pq.read_table(ep_path, columns=["episode_index", "tasks"]).to_pylist()
    episodes = [(int(r["episode_index"]), r["tasks"][0]) for r in rows]
    heldout = [e for e, task in episodes if task == HELDOUT_TASK]
    if len(heldout) != 5:
        raise RuntimeError(f"expected 5 held-out episodes, found {heldout}")
    train = [e for e, task in episodes if task != HELDOUT_TASK]
    if len(train) != 155:
        raise RuntimeError(f"expected 155 training episodes, found {len(train)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "train_episodes.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["episode_index", "task"])
        for e, task in episodes:
            if e in train: w.writerow([e, task])
    manifest = {
        "protocol": "LIBERO-36 Fold 01 draft v1",
        "status": "draft_not_training",
        "dataset_root": str(args.dataset_root),
        "heldout_task": HELDOUT_TASK,
        "heldout_episode_indices": heldout,
        "training_episode_count": len(train),
        "heldout_episode_count": len(heldout),
        "seen_control_tasks": SEEN_CONTROLS,
        "excluded_from_training": [HELDOUT_TASK],
        "task_leakage_check": "held-out instruction absent from training episodes",
        "training_started": False,
        "evaluation_started": False,
    }
    (args.output_dir / "fold01_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
