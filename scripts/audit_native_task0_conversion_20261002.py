"""Full read-only source/converted comparison and real dataset loader smoke."""
import io
import json
import sys
from pathlib import Path
import h5py
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

source, root = map(Path, sys.argv[1:3])
records = []
for file in sorted((root / "data").rglob("*.parquet")):
    records.extend(pq.read_table(file).to_pylist())
records.sort(key=lambda r: (r["episode_index"], r["frame_index"]))
errors = []
state_max = action_max = 0.0
images_checked = 0
with h5py.File(source, "r") as h:
    demos = h["data"]
    keys = sorted(demos.keys(), key=lambda k: int(k.removeprefix("demo_")))
    expected_count = sum(len(demos[k]["actions"]) for k in keys)
    if len(records) != expected_count:
        errors.append(f"frame_count {len(records)} != {expected_count}")
    for row in records:
        e, i = int(row["episode_index"]), int(row["frame_index"])
        demo = demos[keys[e]]
        state = np.concatenate([demo["obs/" + x][i] for x in
                                ("joint_states", "ee_pos", "ee_ori", "gripper_states")]).astype(np.float32)
        action = demo["actions"][i].astype(np.float32)
        sd = float(np.max(np.abs(np.asarray(row["observation.state"], dtype=np.float32) - state)))
        ad = float(np.max(np.abs(np.asarray(row["action"], dtype=np.float32) - action)))
        state_max, action_max = max(state_max, sd), max(action_max, ad)
        if sd or ad:
            errors.append(f"numeric_mismatch {e}:{i}")
        for converted, raw in (("agentview", "agentview_rgb"), ("wrist", "eye_in_hand_rgb")):
            image = row["observation.images." + converted]
            if image.get("bytes") is not None:
                im = Image.open(io.BytesIO(image["bytes"]))
            else:
                im = Image.open(root / image["path"])
            if not np.array_equal(np.asarray(im), demo["obs/" + raw][i]):
                errors.append(f"image_mismatch {e}:{i}:{converted}")
            images_checked += 1
    info = json.loads(demos.attrs["problem_info"])
    lang = info["language_instruction"]
    if isinstance(lang, list):
        lang = " ".join(lang)
    lang = str(lang).strip().strip('"')
tasks = pq.read_table(root / "meta/tasks.parquet").to_pandas()
task_text_matches = lang in list(tasks.index) or any(lang == str(v) for v in tasks.to_numpy().flatten())
if not task_text_matches:
    errors.append("task_text_mismatch")
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset(repo_id="local/libero_spatial_task0_reconstructed_20261002",
                        root=root, delta_timestamps={"action": [i / 20 for i in range(50)]})
smoke = []
for idx in (0, 97, 98, len(dataset) - 1):
    item = dataset[idx]
    shapes = {k: list(item[k].shape) for k in ("action", "observation.state",
              "observation.images.agentview", "observation.images.wrist")}
    finite = all(bool(item[k].isfinite().all()) for k in shapes)
    if shapes["action"] != [50, 7] or shapes["observation.state"] != [15] or not finite:
        errors.append(f"loader_bad_sample {idx}")
    smoke.append({"index": idx, "shapes": shapes, "finite": finite})
result = {"source": str(source), "root": str(root), "frames_checked": len(records),
          "episodes": dataset.num_episodes, "images_checked": images_checked,
          "state_max_abs_diff_float32": state_max, "action_max_abs_diff_float32": action_max,
          "task_text_matches": task_text_matches, "loader_smoke": smoke,
          "errors": errors, "passed": not errors and len(records) == 5068 and dataset.num_episodes == 50}
print(json.dumps(result, indent=2))
sys.exit(0 if result["passed"] else 1)
