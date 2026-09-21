#!/usr/bin/env python3
"""Build a read-only per-episode sampling plan from frozen provenance."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provenance", required=True)
    ap.add_argument("--window-audit", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    provenance = list(csv.DictReader(Path(args.provenance).open(newline="")))
    split = list(csv.DictReader(Path(args.split).open(newline="")))
    split_by_ep = {
        int(r["lerobot_episode_index"]): r["role"]
        for r in split
    }
    audit = json.loads(Path(args.window_audit).read_text())
    audit_by_name = {Path(r["source"]).name: r for r in audit["records"]}

    records = []
    for row in provenance:
        ep = int(row["lerobot_episode_index"])
        source = row["source_filename"]
        if source not in audit_by_name:
            raise ValueError(f"missing audit record for {source}")
        if ep not in split_by_ep:
            raise ValueError(f"missing split assignment for episode {ep}")
        a = audit_by_name[source]
        records.append(
            {
                "episode_index": ep,
                "seed": int(row["seed"]),
                "source_filename": source,
                "split": split_by_ep[ep],
                "frames": int(row["num_frames"]),
                "close_step": a["close_step"],
                "reopen_step": a["reopen_step"],
                "window_start": a["window_start"],
                "window_end_exclusive": a["window_end_exclusive"],
                "window_frames": a["window_frames"],
                "sampling_multiplier": audit["sampling_multiplier"],
            }
        )
    counts = {}
    for r in records:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    out = {
        "schema_version": 1,
        "read_only_plan": True,
        "num_episodes": len(records),
        "episodes_by_split": counts,
        "window_definition": {
            "pre": audit["pre"],
            "post": audit["post"],
            "sampling_multiplier": audit["sampling_multiplier"],
        },
        "records": records,
    }
    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"POSE_TARGET_SAMPLING_PLAN_OK episodes={len(records)} splits={counts}")


if __name__ == "__main__":
    main()
