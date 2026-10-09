"""Environment-only reset contrast; never constructs or calls a policy."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
from first_decision_provenance_v1 import digest_tree

FIELDS = ("body_pos", "body_quat", "geom_pos", "geom_quat", "geom_rgba",
          "cam_pos", "cam_quat", "cam_fovy", "light_pos", "light_dir",
          "light_diffuse", "mat_rgba", "mat_texid")
EVALUATOR_SHA = "89fd36a89dc45a56382219d4c3f6e5d12a3b1689abb3440ec29be346815af7cb"


def capture_model(model):
    arrays = {name: getattr(model, name).copy() for name in FIELDS}
    texture = next((name for name in ("tex_rgb", "tex_data") if hasattr(model, name)), None)
    if texture is None:
        raise RuntimeError("Texture capture unavailable")
    arrays[texture] = getattr(model, texture).copy()
    return {name: digest_tree(value) for name, value in arrays.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--mode", choices=("frozen_order", "preseed"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    source = Path(args.evaluator)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == EVALUATOR_SHA
    os.environ.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", HF_HUB_OFFLINE="1")
    spec = importlib.util.spec_from_file_location("frozen_v3", source)
    v3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v3)
    np = v3.np
    if args.mode == "preseed":
        random.seed(12351)
        np.random.seed(12351)
    suite = v3.benchmark.get_benchmark(v3.SUITE_NAME)()
    task = suite.get_task(0)
    env = v3.OffScreenRenderEnv(bddl_file_name=str(Path(v3.get_libero_path("bddl_files")) /
        task.problem_folder / task.bddl_file), camera_heights=128, camera_widths=128)
    try:
        obs = env.reset()
        updated = env.set_init_state(np.asarray(suite.get_task_init_states(0)[3], dtype=np.float64))
        if updated is not None:
            obs = updated
        waits = 0
        for _ in range(10):
            obs, _, done, _ = env.step(np.zeros(7, dtype=np.float32))
            waits += 1
            if done:
                break
        assert waits == 10
        model = env.sim.model._model
        record = {"mode": args.mode, "seed": 12351 if args.mode == "preseed" else None,
                  "wait_steps": waits, "init": 3, "policy_calls": 0,
                  "predicted_actions": 0, "model_arrays": capture_model(model),
                  "state": digest_tree(env.sim.get_state().flatten()),
                  "cameras": {k: digest_tree(obs[k]) for k in
                      ("agentview_image", "robot0_eye_in_hand_image")},
                  "evaluator_sha256": EVALUATOR_SHA,
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        with output.open("x") as f:
            json.dump(record, f, indent=2)
    finally:
        env.close()


if __name__ == "__main__":
    main()
