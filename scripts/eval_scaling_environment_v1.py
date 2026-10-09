"""Independent four-task controlled rollout; historical task0 wrapper unchanged."""
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
from scaling_eval_schedule_v1 import schedule, validate_protocol


def registered_row(key):
    matches = [row for row in schedule() if row['key'] == key]
    if len(matches) != 1:
        raise ValueError('Unregistered rollout key')
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    for name in ('evaluator', 'checkpoint', 'model-sha', 'root', 'key'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    row = registered_row(args.key)
    root = Path(args.root)
    if root.exists():
        raise FileExistsError(root)
    source = Path(args.evaluator)
    if hashlib.sha256(source.read_bytes()).hexdigest() != EVALUATOR_SHA:
        raise ValueError('Evaluator hash mismatch')
    model = Path(args.checkpoint) / 'model.safetensors'
    if hashlib.sha256(model.read_bytes()).hexdigest() != args.model_sha:
        raise ValueError('Checkpoint hash mismatch')
    os.environ.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', HF_HUB_OFFLINE='1')
    spec = importlib.util.spec_from_file_location('frozen_v3_scaling', source)
    v3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v3)
    captured, loaded = {}, {}

    def model_capture(policy):
        loaded['state_dict'] = digest_tree(dict(policy.state_dict()))

    def capture(env, frame, batch, pre, post):
        if captured:
            raise RuntimeError('Repeated first-input capture')
        captured.update({
            'model_arrays': capture_model(env.sim.model._model),
            'loaded_model': dict(loaded),
            'simulator_state': digest_tree(env.sim.get_state().flatten()),
            'frame': digest_tree(frame), 'input': digest_tree(batch),
            'images': {k: digest_tree(v) for k, v in frame.items()
                       if k.startswith('observation.images.')},
            'processor': digest_tree({'pre': processor_state(pre), 'post': processor_state(post)}),
            'rng': {'python': digest_tree(random.getstate()),
                    'numpy': digest_tree(v3.np.random.get_state()),
                    'torch_cpu': digest_tree(v3.torch.get_rng_state()),
                    'torch_cuda': digest_tree(v3.torch.cuda.get_rng_state_all())}})
        root.mkdir()
        with (root / 'first_input.json').open('x') as f:
            json.dump(captured, f, indent=2)

    restore = install_hooks(v3, capture, model_capture=model_capture)
    episode = SimpleNamespace(checkpoint=args.checkpoint, device='cuda',
        n_action_steps=row['action_steps'], task_id=row['task_id'],
        init_source='benchmark', init_index=row['init'], wait_steps=row['wait'],
        max_steps=row['max_steps'], seed=row['cli_seed'], capture_state=True,
        results_dir=str(root))
    try:
        v3.run_episode(episode)
        if not captured:
            raise RuntimeError('Missing first-input evidence')
        protocol = dict(row, checkpoint=args.checkpoint, model_sha=args.model_sha,
                        evaluator_sha=EVALUATOR_SHA,
                        wrapper_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        validate_protocol(row, protocol)
        with (root / 'protocol.json').open('x') as f:
            json.dump(protocol, f, indent=2)
    finally:
        restore()


if __name__ == '__main__':
    main()
