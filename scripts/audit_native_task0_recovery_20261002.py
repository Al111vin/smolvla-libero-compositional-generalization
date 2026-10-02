"""Read-only validation of official native task0 recovery; JSON to stdout."""
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

path = Path(sys.argv[1])
expected_hash = "ff6f26121653c77280eb40a38773a74141c11a8509f3466058cb56dd2cc60ead"
expected_language = "pick up the black bowl between the plate and the ramekin and place it on the plate"
digest = hashlib.sha256()
with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
shapes = {"actions": (7,), "obs/joint_states": (7,), "obs/ee_pos": (3,),
          "obs/ee_ori": (3,), "obs/gripper_states": (2,),
          "obs/agentview_rgb": (128, 128, 3), "obs/eye_in_hand_rgb": (128, 128, 3)}
errors = []
lengths = []
with h5py.File(path, "r") as source:
    data = source["data"]
    info = json.loads(data.attrs["problem_info"])
    language = info["language_instruction"]
    if isinstance(language, list):
        language = " ".join(language)
    language = str(language).strip().strip('"')
    keys = sorted(data.keys(), key=lambda x: int(x.removeprefix("demo_")))
    for key in keys:
        demo = data[key]
        n = len(demo["actions"])
        lengths.append(n)
        for field, shape in shapes.items():
            if field not in demo:
                errors.append(f"{key}: missing {field}")
                continue
            values = demo[field][:]
            if values.shape != (n, *shape):
                errors.append(f"{key}: {field} shape {values.shape}")
            if not np.isfinite(values).all():
                errors.append(f"{key}: {field} nonfinite")
            if "rgb" in field and values.dtype != np.uint8:
                errors.append(f"{key}: {field} dtype {values.dtype}")
result = {"source": "yifengzhu-hf/LIBERO-datasets", "revision": "97773100c1474cd0d686ebd173cc0e4fd5442466",
          "path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest(),
          "official_hash_matches": digest.hexdigest() == expected_hash,
          "language": language, "task_identity_matches": language == expected_language,
          "episodes": len(lengths), "frames": sum(lengths), "episode_lengths": lengths,
          "schema_errors": errors, "original_converted_dataset_byte_identity_verified": False,
          "training_started": False, "fold02": "LOCKED"}
result["passed"] = (result["official_hash_matches"] and result["task_identity_matches"]
                    and len(lengths) == 50 and sum(lengths) == 5068 and not errors)
print(json.dumps(result, indent=2))
sys.exit(0 if result["passed"] else 1)
