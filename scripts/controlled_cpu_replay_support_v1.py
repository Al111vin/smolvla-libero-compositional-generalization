"""Policy-free replay helpers; importing this module never creates an environment."""

ENVIRONMENT_SEED = 12351


def create_case_output(path):
    """Create missing experiment parent, but never reuse an existing case."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)


def bounded_reset(env):
    """One native reset only: propagate randomization failure, never silently retry."""
    return env.env.reset()


def initialize_replay(env, initial_state, effective_seed, np_module):
    """Mirror frozen evaluator reset/set-state/wait/reseed order without torch."""
    obs = bounded_reset(env)
    updated = env.set_init_state(initial_state)
    if updated is not None:
        obs = updated
    waits = 0
    for _ in range(10):
        obs, _, done, _ = env.step(np_module.zeros(7, dtype=np_module.float32))
        waits += 1
        if done:
            break
    np_module.random.seed(effective_seed)
    return obs, waits


def headless_kwargs(bddl_path):
    return dict(bddl_file_name=str(bddl_path), camera_heights=128,
                camera_widths=128, use_camera_obs=False,
                has_renderer=False, has_offscreen_renderer=False)
