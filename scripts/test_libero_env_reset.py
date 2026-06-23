import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "glx"

import pandas as pd

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


FINAL_TASK_PATH = Path("data/final_task_set.csv")


def main():
    df = pd.read_csv(FINAL_TASK_PATH)

    first_task = df.iloc[0]

    suite = first_task["suite"]
    task_id = int(first_task["libero_task_id"])

    print("=" * 80)
    print("Testing LIBERO environment reset")
    print("suite:", suite)
    print("task_id:", task_id)

    bench_cls = benchmark.get_benchmark(suite)
    bench = bench_cls()

    task = bench.get_task(task_id)

    print("language:", task.language)
    print("bddl_file:", task.bddl_file)
    print("problem_folder:", task.problem_folder)

    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    print("full bddl path:", bddl_path)

    if not bddl_path.exists():
        raise FileNotFoundError(f"BDDL file not found: {bddl_path}")

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=128,
        camera_widths=128,
    )

    env.seed(0)
    env.reset()

    init_states = bench.get_task_init_states(task_id)
    print("number of init states:", len(init_states))

    obs = env.set_init_state(init_states[0])

    if obs is None:
        obs = env.reset()

    print("Observation keys:")
    for key in obs.keys():
        value = obs[key]
        if hasattr(value, "shape"):
            print(f"  {key}: shape={value.shape}")
        else:
            print(f"  {key}: type={type(value)}")

    env.close()

    print("\nLIBERO environment reset test passed.")


if __name__ == "__main__":
    main()