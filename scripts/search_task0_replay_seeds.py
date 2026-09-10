from pathlib import Path
import sys
import numpy as np
from libero.libero.envs import OffScreenRenderEnv
from scripts import validate_libero_36_envs as v

task_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
replay = sorted(Path("results/libero36_gate5_official_v6").glob(f"task_{task_id:03d}_*/replay.npz"))[0]
row = [r for r in v.read_layout_spec(Path("data/libero_36/layout_spec.csv")) if int(r["task_id"]) == task_id and int(r["layout_id"]) == 1][0]
z = np.load(replay)
env = OffScreenRenderEnv(bddl_file_name=str(row["bddl_path"]), camera_heights=128, camera_widths=128, horizon=1000, use_camera_obs=False, has_offscreen_renderer=False)
base = int(z["seed"])
for seed in range(base, base + 31):
    v.seed_environment(env, seed)
    env.env.reset()
    for _ in range(20): env.step(np.zeros(7, dtype=np.float32))
    rewards=[]; successes=[]
    for action in z["actions"]:
        _, r, _, _ = env.step(action)
        rewards.append(float(r)); successes.append(bool(env.check_success()))
    print(seed, bool(successes[-1] or np.any(np.asarray(rewards)>0)), int(np.count_nonzero(np.asarray(rewards)>0)), bool(np.asarray(successes[-20:]).all()), flush=True)
env.close()
