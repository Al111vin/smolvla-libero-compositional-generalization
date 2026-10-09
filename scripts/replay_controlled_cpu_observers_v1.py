"""Bounded, policy-free stored-action replay with diagnostic object/contact observers."""
import argparse
import math
import csv
import hashlib
import json
import os
from pathlib import Path
import random
from controlled_cpu_replay_support_v1 import ENVIRONMENT_SEED, headless_kwargs, initialize_replay
from first_decision_provenance_v1 import digest_tree
from capture_reset_render_provenance_v1 import capture_model


def main():
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--init', type=int, required=True)
    args = parser.parse_args()
    assert args.init in (3,0,12,13,14,6)
    root = Path(args.output)
    source = Path(args.source)
    assert source.is_dir() and source.resolve() != root.resolve()
    root.mkdir(exist_ok=False)
    record = {'status': 'started', 'policy_calls': 0, 'gpu_rendering': False,
              'physical_stage_interpretation': False, 'automatic_retries': 0,
              'source_hashes': {}, 'verified_steps': 0}
    env = None
    try:
        files = [source/'protocol.json', source/'first_input.json'] + list(source.glob('*_actions.csv'))
        assert len(files) == 3
        record['source_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        protocol = json.loads(files[0].read_text())
        assert protocol['init'] == args.init and protocol['environment_seed'] == 12351
        assert protocol['wait'] == 10 and protocol['max_steps'] == 300
        assert protocol['effective_seed'] == 12345+2*args.init
        assert protocol['action_steps'] == 25
        assert protocol['model_sha'] in (
            '475089185e820b0cd994a5b80365ebd4ea843ac355ef76d5c81d60b01b302b1e',
            '7c480b7d34c3bc4d494538cd26a4a82560d16b8a116424f086a7b1e956ce6ddf')
        capture = json.loads(files[1].read_text())
        rows = list(csv.DictReader(files[2].open()))
        assert 0 < len(rows) <= 300
        suite = benchmark.get_benchmark('libero_spatial')()
        task = suite.get_task(0)
        bddl = Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
        record['bddl_sha256'] = hashlib.sha256(bddl.read_bytes()).hexdigest()
        random.seed(ENVIRONMENT_SEED)
        np.random.seed(ENVIRONMENT_SEED)
        env = ControlEnv(**headless_kwargs(bddl))
        obs, waits = initialize_replay(env, np.asarray(suite.get_task_init_states(0)[args.init], dtype=np.float64), 12345+2*args.init, np)
        assert waits == 10
        record['model_arrays'] = capture_model(env.sim.model._model)
        record['simulator_state'] = digest_tree(env.sim.get_state().flatten())
        assert record['model_arrays'] == capture['model_arrays'], 'initial compiled model mismatch'
        assert record['simulator_state'] == capture['simulator_state'], 'initial simulator mismatch'
        observations = []
        baseline_z = float(obs['akita_black_bowl_1_pos'][2])
        worst = 0.0
        for i, row in enumerate(rows):
            assert int(row['step']) == i
            q = np.asarray(obs['robot0_eef_quat'], dtype=np.float64)
            norm = np.linalg.norm(q)
            if norm < 1e-8:
                ori = np.zeros(3, dtype=np.float32)
            else:
                q = q/norm
                w = np.clip(q[3], -1, 1)
                angle = 2*np.arccos(w)
                scale = np.sqrt(max(1-w*w, 0))
                axis = np.array([1.,0.,0.]) if scale < 1e-8 else q[:3]/scale
                ori = (axis*angle).astype(np.float32)
            state = np.concatenate([np.asarray(obs['robot0_joint_pos'], dtype=np.float32),
                 np.asarray(obs['robot0_eef_pos'], dtype=np.float32), ori,
                 np.asarray(obs['robot0_gripper_qpos'], dtype=np.float32)]).astype(np.float32)
            assert state.shape == (15,)
            expected = np.array([float(row[f'state_{j}']) for j in range(15)])
            delta = float(np.max(np.abs(state-expected)))
            assert np.isfinite(delta) and delta <= 1e-6, f'robot state mismatch step {i}: {delta}'
            action = np.array([float(row[f'applied_action_{j}']) for j in range(7)], dtype=np.float32)
            assert np.isfinite(action).all() and np.max(np.abs(action)) <= 1
            pre_distance = float(np.linalg.norm(np.asarray(obs['robot0_eef_pos'])-obs['akita_black_bowl_1_pos']))
            obs, reward, done, _ = env.step(action)
            assert math.isfinite(float(reward)) and math.isfinite(float(row['reward']))
            assert abs(float(reward)-float(row['reward'])) <= 1e-7, f'reward mismatch step {i}'
            assert not (done or reward > 0) or i == len(rows)-1, 'early terminal mismatch'
            bowl = np.asarray(obs['akita_black_bowl_1_pos'])
            plate = np.asarray(obs['plate_1_pos'])
            contacts = []
            model = env.sim.model
            for contact in env.sim.data.contact[:env.sim.data.ncon]:
                contacts.append([model.geom_id2name(int(contact.geom1)), model.geom_id2name(int(contact.geom2))])
            observations.append(dict(step=i+1, bowl_xyz=bowl.tolist(), plate_xyz=plate.tolist(),
                eef_xyz=np.asarray(obs['robot0_eef_pos']).tolist(), contacts=contacts,
                approach=pre_distance <= .05, lift=float(bowl[2])-baseline_z >= .03,
                transport=float(np.linalg.norm(bowl[:2]-plate[:2])) <= .08,
                official_success=bool(env.check_success()), reward=float(reward)))
            worst = max(worst, delta)
            record['verified_steps'] = i+1
        assert done or reward > 0 or len(rows) == 300, 'truncated source'
        with (root/'object_contact_observations.json').open('x') as f:
            json.dump(observations, f, allow_nan=False)
        record['first_proxy_steps'] = {k: next((r['step'] for r in observations if r[k]), None)
            for k in ('approach','lift','transport','official_success')}
        record.update(status='parity_passed', max_robot_state_error=worst, final_reward=float(reward))
    except Exception as error:
        record.update(status='stopped_preserved_failure', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        with (root/'result.json').open('x') as f:
            json.dump(record, f, indent=2, allow_nan=False)
        print(json.dumps(record))
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
