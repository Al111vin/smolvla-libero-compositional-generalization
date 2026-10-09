"""Hash-pinned V3 rollout, early environment seed, first-input provenance."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
from controlled_eval_hooks_v1 import install_hooks
from first_decision_provenance_v1 import digest_tree, processor_state
from capture_reset_render_provenance_v1 import capture_model, EVALUATOR_SHA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-sha", required=True)
    parser.add_argument("--init-index", type=int, required=True)
    parser.add_argument("--cli-seed", type=int, required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    assert 0 <= args.init_index < 20
    assert args.cli_seed == 12345 + args.init_index
    root = Path(args.root)
    assert not root.exists()
    source = Path(args.evaluator)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == EVALUATOR_SHA
    assert hashlib.sha256((Path(args.checkpoint) / "model.safetensors").read_bytes()).hexdigest() == args.model_sha
    os.environ.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", HF_HUB_OFFLINE="1")
    spec = importlib.util.spec_from_file_location("frozen_v3", source)
    v3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v3)
    captured = {}
    loaded_model = {}

    def model_capture(policy):
        loaded_model["state_dict"] = digest_tree(dict(policy.state_dict()))

    def capture(env, frame, batch, pre, post):
        assert not captured
        captured.update({"model_arrays": capture_model(env.sim.model._model),
            "loaded_model": dict(loaded_model),
            "simulator_state": digest_tree(env.sim.get_state().flatten()),
            "frame": digest_tree(frame), "input": digest_tree(batch),
            "images": {k: digest_tree(v) for k,v in frame.items() if k.startswith("observation.images.")},
            "processor": digest_tree({"pre":processor_state(pre),"post":processor_state(post)}),
            "rng": {"python":digest_tree(random.getstate()), "numpy":digest_tree(v3.np.random.get_state()),
                    "torch_cpu":digest_tree(v3.torch.get_rng_state()),
                    "torch_cuda":digest_tree(v3.torch.cuda.get_rng_state_all())}})
        root.mkdir()
        with (root / "first_input.json").open("x") as f:
            json.dump(captured, f, indent=2)

    restore = install_hooks(v3, capture, model_capture=model_capture)
    episode = SimpleNamespace(checkpoint=args.checkpoint, device="cuda", n_action_steps=25,
        task_id=0, init_source="benchmark", init_index=args.init_index, wait_steps=10,
        max_steps=300, seed=args.cli_seed, capture_state=True, results_dir=str(root))
    try:
        v3.run_episode(episode)
        assert captured
        with (root / "protocol.json").open("x") as f:
            json.dump({"environment_seed":12351,"init":args.init_index,"cli_seed":args.cli_seed,
                "effective_seed":args.cli_seed+args.init_index,"wait":10,"max_steps":300,
                "action_steps":25,"checkpoint":args.checkpoint,"model_sha":args.model_sha,
                "evaluator_sha":EVALUATOR_SHA,"wrapper_sha":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},f,indent=2)
    finally:
        restore()


if __name__ == "__main__":
    main()
