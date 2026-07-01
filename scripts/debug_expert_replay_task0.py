from pathlib import Path
import h5py
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


dataset_path = Path(
    "datasets/libero/datasets/libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
)

suite_cls = benchmark.get_benchmark("libero_spatial")
suite = suite_cls()
task = suite.get_task(0)

bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

env = OffScreenRenderEnv(
    bddl_file_name=str(bddl_path),
    camera_heights=128,
    camera_widths=128,
)

with h5py.File(dataset_path, "r") as f:
    demo = f["data/demo_0"]
    actions = np.asarray(demo["actions"])
    states = np.asarray(demo["states"])

print("=" * 80)
print("EXPERT REPLAY TASK 0")
print("=" * 80)
print("actions shape:", actions.shape)
print("states shape:", states.shape)

obs = env.reset()

# Use the exact simulator state from the dataset demo
tmp = env.set_init_state(states[0])
if tmp is not None:
    obs = tmp

total_reward = 0.0
info = {}
done = False

for t, action in enumerate(actions):
    obs, reward, done, info = env.step(action)
    total_reward += float(reward)

    if t % 50 == 0:
        print(f"step {t:04d} | reward={reward} | done={done} | info={info}")

    if done:
        break

success = bool(info.get("success", info.get("is_success", False))) or total_reward > 0

print("=" * 80)
print("REPLAY FINISHED")
print("success:", success)
print("total_reward:", total_reward)
print("steps:", t + 1)
print("final info:", info)

env.close()
