"""Inspect saved simulator states without advancing or changing evidence."""
import argparse
import json
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
from scripts import calibrate_libero_36_push as c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attempt', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.attempt, allow_pickle=False) as z:
        meta = json.loads(str(z['metadata_json'].item()))
        row = next(r for r in c.reset_validator.read_layout_spec(Path('data/libero_36/layout_spec.csv'))
                   if int(r['task_id']) == meta['task_id'] and int(r['layout_id']) == meta['layout_id'])
        env = c.reset_validator.make_environment(Path(row['bddl_path']))
        try:
            c.settle_environment(env, row, meta['seed'])
            sim = env.env.sim
            selected = meta.get('pusher_finger', 'left')
            contact_indices = np.flatnonzero(z[selected + '_contact'])
            push_indices = np.flatnonzero(z['phase'] == 'push')
            start = int(contact_indices[0] if len(contact_indices) else
                        push_indices[0] if len(push_indices) else len(z['actions'])-1)
            reports = []
            for step in sorted(set([max(0,start-1), start, min(start+1,len(z['actions'])-1), len(z['actions'])-1])):
                sim.set_state_from_flattened(z['states'][step+1])
                sim.forward()
                contacts = []
                for j in range(sim.data.ncon):
                    con = sim.data.contact[j]
                    names = [sim.model.geom_id2name(int(g)) for g in (con.geom1, con.geom2)]
                    if any(name and name.startswith('gripper') for name in names):
                        contacts.append(dict(names=names, position=con.pos.tolist(),
                                             normal=np.asarray(con.frame).reshape(3,3)[0].tolist(), distance=float(con.dist)))
                rgb = sim.render(width=640, height=640, camera_name='agentview')
                imageio.imwrite(args.output / f'step_{step}.png', rgb[::-1])
                geometry = c.contact_geometry(env, c.target_instance(row))
                pads = {k: v.tolist() for k,v in c.finger_positions(env.env, geometry).items()}
                reports.append(dict(step=step, pads=pads, target=z['target_xyz'][step].tolist(),
                                    action=z['actions'][step].tolist(), contacts=contacts))
            (args.output / 'contacts.json').write_text(json.dumps(reports, indent=2))
            print(json.dumps(reports))
        finally:
            env.close()


if __name__ == '__main__':
    main()
