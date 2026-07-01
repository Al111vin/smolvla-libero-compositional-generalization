from pathlib import Path
import h5py
import imageio.v2 as imageio

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


out_dir = Path("results/debug_images")
out_dir.mkdir(parents=True, exist_ok=True)

dataset_path = Path(
    "datasets/libero/datasets/libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
)

with h5py.File(dataset_path, "r") as f:
    demo = f["data/demo_0"]
    dataset_agent = demo["obs/agentview_rgb"][0]
    dataset_wrist = demo["obs/eye_in_hand_rgb"][0]

imageio.imwrite(out_dir / "dataset_agentview_rgb.png", dataset_agent)
imageio.imwrite(out_dir / "dataset_wrist_rgb.png", dataset_wrist)

suite_cls = benchmark.get_benchmark("libero_spatial")
suite = suite_cls()
task = suite.get_task(0)

bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

env = OffScreenRenderEnv(
    bddl_file_name=str(bddl_path),
    camera_heights=128,
    camera_widths=128,
)

init_states = suite.get_task_init_states(0)

obs = env.reset()
tmp = env.set_init_state(init_states[0])
if tmp is not None:
    obs = tmp

env_agent = obs["agentview_image"]
env_wrist = obs["robot0_eye_in_hand_image"]

imageio.imwrite(out_dir / "env_agentview_image.png", env_agent)
imageio.imwrite(out_dir / "env_wrist_image.png", env_wrist)

env.close()

print("Saved images to:", out_dir)
print("Files:")
for p in sorted(out_dir.glob("*.png")):
    print(" ", p)
