from pathlib import Path
import numpy as np
from libero.libero.envs import OffScreenRenderEnv
from scripts import validate_libero_36_envs as v

row = [r for r in v.read_layout_spec(Path("data/libero_36/layout_spec.csv")) if int(r["task_id"]) == 0 and int(r["layout_id"]) == 1][0]
z = np.load("results/libero36_gate5_official_v6/task_000_put_on_top/replay.npz")
env = OffScreenRenderEnv(bddl_file_name=str(row["bddl_path"]), camera_heights=128, camera_widths=128, horizon=1000, use_camera_obs=False, has_offscreen_renderer=False)
for seed in range(361005, 361031):
    v.seed_environment(env, seed)
    env.env.reset()
    for _ in range(20): env.step(np.zeros(7, dtype=np.float32))
    rewards=[]; successes=[]
    for action in z["actions"]:
        _, r, _, _ = env.step(action)
        rewards.append(float(r)); successes.append(bool(env.check_success()))
    print(seed, bool(successes[-1] or np.any(np.asarray(rewards)>0)), int(np.count_nonzero(np.asarray(rewards)>0)), bool(np.asarray(successes[-20:]).all()), flush=True)
env.close()
