"""Controller diagnostics for lateral push contact; no acceptance overrides."""
import numpy as np


def lateral_push_contacts(sim, target_names, robot_names, push_direction):
    direction = np.asarray(push_direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    records = []
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        first = sim.model.geom_id2name(int(contact.geom1))
        second = sim.model.geom_id2name(int(contact.geom2))
        normal = np.asarray(contact.frame).reshape(3, 3)[0].copy()
        # MuJoCo normal points from geom1 toward geom2. Orient it as the
        # direction of the force that the robot would apply to the target.
        if first in robot_names and second in target_names:
            pass
        elif first in target_names and second in robot_names:
            normal = -normal
        else:
            continue
        alignment = float(normal[:2] @ direction)
        records.append(dict(geometries=[first, second],
                            position=contact.pos.tolist(),
                            robot_to_target_normal=normal.tolist(),
                            alignment=alignment,
                            lateral_aligned=bool(abs(normal[2]) <= 0.35 and alignment >= 0.8)))
    return records
