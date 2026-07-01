from pathlib import Path
import h5py
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def print_arr(name, x):
    x = np.asarray(x)
    print(f"{name}: shape={x.shape}")
    print(x)
    print()


dataset_path = Path(
    "datasets/libero/datasets/libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
)

print("=" * 80)
print("DATASET FIRST FRAME")
print("=" * 80)

with h5py.File(dataset_path, "r") as f:
    demo = f["data/demo_0"]
    d_joint = demo["obs/joint_states"][0]
    d_ee_pos = demo["obs/ee_pos"][0]
    d_ee_ori = demo["obs/ee_ori"][0]
    d_gripper = demo["obs/gripper_states"][0]
    d_action = demo["actions"][0]

print_arr("dataset joint_states[0]", d_joint)
print_arr("dataset ee_pos[0]", d_ee_pos)
print_arr("dataset ee_ori[0]", d_ee_ori)
print_arr("dataset gripper_states[0]", d_gripper)
print_arr("dataset action[0]", d_action)

print("=" * 80)
print("ENV AFTER set_init_state")
print("=" * 80)

suite_cls = benchmark.get_benchmark("libero_spatial")
suite = suite_cls()
task_id = 0
task = suite.get_task(task_id)

bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

env = OffScreenRenderEnv(
    bddl_file_name=str(bddl_path),
    camera_heights=128,
    camera_widths=128,
)

init_states = suite.get_task_init_states(task_id)

obs = env.reset()
tmp = env.set_init_state(init_states[0])
if tmp is not None:
    obs = tmp

print_arr("env robot0_joint_pos", obs["robot0_joint_pos"])
print_arr("env robot0_eef_pos", obs["robot0_eef_pos"])
print_arr("env robot0_eef_quat", obs["robot0_eef_quat"])
print_arr("env robot0_eef_quat[:3]", obs["robot0_eef_quat"][:3])
print_arr("env robot0_gripper_qpos", obs["robot0_gripper_qpos"])

env.close()

print("=" * 80)
print("DIFFERENCES")
print("=" * 80)

print_arr("joint diff env - dataset", obs["robot0_joint_pos"] - d_joint)
print_arr("ee_pos diff env - dataset", obs["robot0_eef_pos"] - d_ee_pos)
print_arr("quat[:3] diff env - dataset ee_ori", obs["robot0_eef_quat"][:3] - d_ee_ori)
print_arr("gripper diff env - dataset", obs["robot0_gripper_qpos"] - d_gripper)
