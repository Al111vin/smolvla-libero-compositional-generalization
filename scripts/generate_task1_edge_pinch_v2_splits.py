import csv, json

PROVENANCE = "/root/smolvla-training-prep/datasets/lerobot/libero36_task1_edge_pinch_v2_demo/episode_provenance.csv"
OUT_CSV = "/root/smolvla-training-prep/datasets/lerobot/libero36_task1_edge_pinch_v2_demo/task1_edge_pinch_v2_train_val_split_v1.csv"

VAL_SEEDS = {950021, 950022, 950023, 950024, 950025}
TRAIN_SEEDS = {950001 + i for i in range(20)}  # 950001-950020

rows = list(csv.DictReader(open(PROVENANCE)))
assert len(rows) == 96, f"expected 96 provenance rows, got {len(rows)}"

out_rows = []
seen_seeds = set()
for r in rows:
    seed = int(r["seed"])
    seen_seeds.add(seed)
    if seed in VAL_SEEDS:
        role = "val"
    elif seed in TRAIN_SEEDS:
        role = "train"
    else:
        raise ValueError(f"seed {seed} not in either TRAIN_SEEDS or VAL_SEEDS -- refusing, unexpected seed")
    out_rows.append({
        "fold_id": "task1_edge_pinch_v2_split_v1",
        "role": role,
        "lerobot_episode_index": r["lerobot_episode_index"],
        "seed": seed,
        "azimuth_deg": r["azimuth_deg"],
        "standoff_m": r["standoff_m"],
    })

seed_roles = {}
for r in out_rows:
    seed_roles.setdefault(r["seed"], set()).add(r["role"])
mixed = {s: roles for s, roles in seed_roles.items() if len(roles) > 1}
if mixed:
    raise RuntimeError(f"FATAL: seed(s) split across train/val, isolation violated: {mixed}")

missing_seeds = seen_seeds - (VAL_SEEDS | TRAIN_SEEDS)
if missing_seeds:
    raise RuntimeError(f"FATAL: unassigned seeds found: {missing_seeds}")

with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["fold_id","role","lerobot_episode_index","seed","azimuth_deg","standoff_m"])
    w.writeheader()
    w.writerows(out_rows)

train_rows = [r for r in out_rows if r["role"] == "train"]
val_rows = [r for r in out_rows if r["role"] == "val"]

summary = {
    "total_episodes": len(out_rows),
    "train_episodes": len(train_rows),
    "val_episodes": len(val_rows),
    "train_seeds_count": len(set(r["seed"] for r in train_rows)),
    "val_seeds_count": len(set(r["seed"] for r in val_rows)),
    "train_seeds": sorted(set(r["seed"] for r in train_rows)),
    "val_seeds": sorted(set(r["seed"] for r in val_rows)),
    "train_episode_indices": sorted(int(r["lerobot_episode_index"]) for r in train_rows),
    "val_episode_indices": sorted(int(r["lerobot_episode_index"]) for r in val_rows),
    "seed_isolation_ok": len(mixed) == 0,
    "all_96_seeds_assigned": len(missing_seeds) == 0,
    "output_csv": OUT_CSV,
}
print(json.dumps(summary, indent=2))
print("SPLIT_GENERATION_COMPLETE")
