"""Two first calls only; no predicted action is sent to LIBERO."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random

from first_decision_provenance_v1 import digest_tree, paired_first_calls, processor_state

EVALUATOR_SHA = "89fd36a89dc45a56382219d4c3f6e5d12a3b1689abb3440ec29be346815af7cb"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    source = Path(args.evaluator)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == EVALUATOR_SHA
    checkpoint = Path(args.checkpoint)
    assert checkpoint.parent.name == "080000" and checkpoint.name == "pretrained_model"
    checkpoint_sha = hashlib.sha256((checkpoint / "model.safetensors").read_bytes()).hexdigest()
    assert checkpoint_sha == "51cbcf9a467d0c2d5c504b6decc7281a1d4b68337e141a8d990820081a07e96a"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["HF_HUB_OFFLINE"] = "1"
    spec = importlib.util.spec_from_file_location("frozen_v3", source)
    v3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v3)
    torch, np = v3.torch, v3.np
    policy = v3.SmolVLAPolicy.from_pretrained(args.checkpoint).to("cuda").eval()
    assert policy.config.n_action_steps == 25
    model_digest = digest_tree(dict(policy.state_dict()))
    pre, post = v3.make_pre_post_processors(policy.config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cuda"}})
    processor_digest = digest_tree({"pre": processor_state(pre), "post": processor_state(post)})
    suite = v3.benchmark.get_benchmark(v3.SUITE_NAME)()
    task = suite.get_task(0)
    env = v3.OffScreenRenderEnv(bddl_file_name=str(Path(v3.get_libero_path("bddl_files")) /
        task.problem_folder / task.bddl_file), camera_heights=128, camera_widths=128)
    try:
        obs = env.reset()
        updated = env.set_init_state(np.asarray(suite.get_task_init_states(0)[3], dtype=np.float64))
        if updated is not None:
            obs = updated
        for _ in range(10):
            obs, _, done, _ = env.step(np.zeros(7, dtype=np.float32))
            if done:
                break
        # Preserve V3 ordering: seed only after checkpoint load and environment wait.
        np.random.seed(12351)
        torch.manual_seed(12351)
        torch.cuda.manual_seed_all(12351)
        policy.reset()
        frame = v3.observation_to_frame(obs, task.language)
        batch = pre(frame)

        def snapshot():
            return {"python": random.getstate(), "numpy": np.random.get_state(),
                    "cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}

        def restore(s):
            random.setstate(s["python"])
            np.random.set_state(s["numpy"])
            torch.set_rng_state(s["cpu"])
            torch.cuda.set_rng_state_all(s["cuda"])

        def infer(p, b):
            with torch.inference_mode():
                raw = p.select_action(b)
                processed = post(raw)
            return {"raw": v3.action_to_numpy(raw).tolist(),
                    "processed": v3.action_to_numpy(processed).tolist()}

        calls = paired_first_calls(policy, batch, snapshot, restore, infer)
        result = {"model_digest": model_digest, "frame_digest": digest_tree(frame),
                  "pixels_digest": digest_tree({k: obs[k] for k in ("agentview_image", "robot0_eye_in_hand_image")}),
                  "simulator_state_digest": digest_tree(env.sim.get_state().flatten()),
                  "calls": calls, "torch_version": torch.__version__,
                  "deterministic": torch.are_deterministic_algorithms_enabled(),
                  "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                  "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                  "processor_digest": processor_digest,
                  "evaluator_sha256": EVALUATOR_SHA,
                  "instrumentation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "helper_sha256": hashlib.sha256(Path(__file__).with_name("first_decision_provenance_v1.py").read_bytes()).hexdigest(),
                  "checkpoint_model_sha256": checkpoint_sha,
                  "cuda_version": torch.version.cuda,
                  "device_name": torch.cuda.get_device_name(0),
                  "cudnn_benchmark": torch.backends.cudnn.benchmark,
                  "cudnn_deterministic": torch.backends.cudnn.deterministic,
                  "closed_loop_action_steps": 0}
        with output.open("x") as f:
            json.dump(result, f, indent=2)
    finally:
        env.close()


if __name__ == "__main__":
    main()
