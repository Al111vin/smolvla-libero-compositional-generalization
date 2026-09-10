"""Diagnostic-only bounded IK on a frozen scene; not a physical trajectory."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scripts import calibrate_libero_36_push as c
from scripts.probe_gate5_pad_entries import yaw_rotation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task-id', type=int, default=15,
                   help='Push-task id in the frozen layout specification.')
    p.add_argument('--gripper-trace', type=Path)
    p.add_argument('--state-trace', type=Path,
                   help='Diagnostic trajectory whose frozen simulator state is scanned.')
    p.add_argument('--state-index', type=int,
                   help='State index in --state-trace (required with it).')
    p.add_argument('--yaws', type=float, nargs='+', default=(-135., -90., -45., 0., 45., 90., 135., 180.),
                   help='Tool yaw offsets, in degrees, relative to the settled orientation.')
    p.add_argument('--heights', type=float, nargs='+', default=(-0.02, 0.00, 0.02, 0.04, 0.06),
                   help='Vertical offsets from the target centre for the left pad, in metres.')
    p.add_argument('--x-offsets', type=float, nargs='+', default=(0.070,),
                   help='Positive-x left-pad offsets from the target centre, in metres.')
    p.add_argument('--y-offsets', type=float, nargs='+', default=(0.0,),
                   help='Y left-pad offsets from the target centre, in metres.')
    p.add_argument('--attempts', type=int, default=2)
    p.add_argument('--pad', choices=('left', 'right'), default='left',
                   help='Finger-pad centre constrained by the IK diagnostic.')
    p.add_argument('--gripper-scale', type=float, default=1.0,
                   help='Scale applied to the traced open-jaw joint positions (0 = nearly closed).')
    args = p.parse_args()
    if (args.state_trace is None) != (args.state_index is None):
        p.error('--state-trace and --state-index must be supplied together')
    if args.output.exists():
        raise FileExistsError(args.output)
    row = next(r for r in c.reset_validator.read_layout_spec(Path('data/libero_36/layout_spec.csv'))
               if int(r['task_id']) == args.task_id and int(r['layout_id']) == 1)
    seed = c.reset_validator.seed_for(row, c.GATE3_CALIBRATION_RESET_INDEX, c.reset_validator.FULL_SEED_BASE)
    env = c.reset_validator.make_environment(Path(row['bddl_path']))
    try:
        settled = c.settle_environment(env, row, seed)
        sim = env.env.sim
        geom = c.contact_geometry(env, c.target_instance(row))
        ids = [sim.model.joint_name2id(f'robot0_joint{i}') for i in range(1,8)]
        addresses = sim.model.jnt_qposadr[ids]
        bounds = sim.model.jnt_range[ids]
        initial_state = np.asarray(env.get_sim_state()).copy()
        initial_q = sim.data.qpos[addresses].copy()
        frozen_target = None
        if args.state_trace:
            with np.load(args.state_trace, allow_pickle=False) as trace:
                if not 0 <= args.state_index < len(trace['states']):
                    raise IndexError('--state-index is outside states')
                sim.set_state_from_flattened(trace['states'][args.state_index])
            sim.forward()
            initial_state = np.asarray(env.get_sim_state()).copy()
            initial_q = sim.data.qpos[addresses].copy()
            frozen_target = sim.data.body_xpos[
                sim.model.body_name2id(f'{c.target_instance(row)}_main')
            ].copy()
        if args.gripper_trace:
            gripper_addresses = [int(sim.model.jnt_qposadr[j]) for j in range(sim.model.njnt)
                                 if (sim.model.joint_id2name(j) or '').startswith('gripper0_')]
            with np.load(args.gripper_trace, allow_pickle=False) as trace:
                sim.set_state_from_flattened(trace['states'][300])
                sim.forward()
                opening = sim.data.qpos[gripper_addresses].copy()
            sim.set_state_from_flattened(initial_state)
            sim.data.qpos[gripper_addresses] = opening * args.gripper_scale
            sim.forward()
            initial_state = np.asarray(env.get_sim_state()).copy()
        site_id = sim.model.site_name2id('gripper0_grip_site')
        # A frozen trajectory may have arrived with a materially different wrist
        # orientation from the reset pose.  Interpret the scan offsets relative
        # to that actual state, otherwise this is not a local re-contact scan.
        base_rotation = (sim.data.site_xmat[site_id].reshape(3, 3).copy()
                         if frozen_target is not None else settled['desired_rotation'])
        target = frozen_target if frozen_target is not None else c.target_xyz(settled['obs'], c.target_instance(row))
        records = []
        for yaw in args.yaws:
            desired_rotation = yaw_rotation(yaw) @ base_rotation
            for height in args.heights:
                for x_offset in args.x_offsets:
                    for y_offset in args.y_offsets:
                        desired_pad = target + np.array([x_offset, y_offset, height])
                        for attempt in range(args.attempts):
                            sim.set_state_from_flattened(initial_state)
                            sim.forward()
                            start = initial_q.copy()
                            if attempt:
                                start += np.random.default_rng(attempt).normal(0, 0.35, 7)
                            start = np.clip(start, bounds[:,0]+1e-5, bounds[:,1]-1e-5)

                            def residual(q):
                                sim.data.qpos[addresses] = q
                                sim.forward()
                                pad = c.finger_positions(env.env, geom)[args.pad]
                                current_rotation = sim.data.site_xmat[site_id].reshape(3,3)
                                rot_error = Rotation.from_matrix(desired_rotation @ current_rotation.T).as_rotvec()
                                return np.concatenate([pad-desired_pad, 0.15*rot_error])

                            def jacobian(q):
                                # A relative finite-difference step vanishes for reset
                                # joints near zero. Use a fixed angular perturbation.
                                columns = []
                                for axis in range(len(q)):
                                    plus, minus = q.copy(), q.copy()
                                    plus[axis] += 1e-5
                                    minus[axis] -= 1e-5
                                    columns.append((residual(plus)-residual(minus))/2e-5)
                                residual(q)
                                return np.column_stack(columns)

                            result = least_squares(residual, start, bounds=(bounds[:,0]+1e-5,bounds[:,1]-1e-5),
                                                   jac=jacobian, max_nfev=160,
                                                   ftol=1e-10, xtol=1e-10, gtol=1e-10)
                            error = residual(result.x)
                            contacts = []
                            for j in range(sim.data.ncon):
                                con = sim.data.contact[j]
                                names = [sim.model.geom_id2name(int(g)) for g in (con.geom1,con.geom2)]
                                if any(n and (n.startswith('robot0_') or n.startswith('gripper0_')) for n in names):
                                    contacts.append(dict(names=names, distance=float(con.dist)))
                            record = dict(pad=args.pad, gripper_scale=args.gripper_scale, yaw=yaw, height=height,
                                          x_offset=x_offset, y_offset=y_offset,
                                          attempt=attempt, q=result.x.tolist(), desired_pad=desired_pad.tolist(),
                                          position_error=float(np.linalg.norm(error[:3])),
                                          orientation_error=float(np.linalg.norm(error[3:])/0.15), contacts=contacts,
                                          evaluations=result.nfev,
                                          joint_margin=float(np.min(np.minimum(result.x-bounds[:,0], bounds[:,1]-result.x))))
                            records.append(record)
                            print(json.dumps(record), flush=True)
        args.output.write_text(json.dumps(dict(diagnostic_only=True,
                                              gripper_trace=str(args.gripper_trace), records=records), indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
