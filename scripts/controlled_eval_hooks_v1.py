"""Dependency-light hooks; original evaluator owns every action and decision."""
import random


def install_hooks(v3, capture, seed=12351, model_capture=None):
    original_env = v3.OffScreenRenderEnv
    original_processors = v3.make_pre_post_processors
    original_policy = v3.SmolVLAPolicy if model_capture is not None else None
    current = {}

    def env_factory(*args, **kwargs):
        random.seed(seed)
        v3.np.random.seed(seed)
        env = original_env(*args, **kwargs)
        current["env"] = env
        return env

    def processors(*args, **kwargs):
        pre, post = original_processors(*args, **kwargs)
        first = True

        def wrapped_pre(frame):
            nonlocal first
            batch = pre(frame)
            if first:
                capture(current["env"], frame, batch, pre, post)
                first = False
            return batch

        return wrapped_pre, post

    v3.OffScreenRenderEnv = env_factory
    v3.make_pre_post_processors = processors
    if model_capture is not None:
        class PolicyFactory:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                policy = original_policy.from_pretrained(*args, **kwargs)
                model_capture(policy)
                return policy
        v3.SmolVLAPolicy = PolicyFactory

    def restore():
        v3.OffScreenRenderEnv = original_env
        v3.make_pre_post_processors = original_processors
        if model_capture is not None:
            v3.SmolVLAPolicy = original_policy

    return restore
