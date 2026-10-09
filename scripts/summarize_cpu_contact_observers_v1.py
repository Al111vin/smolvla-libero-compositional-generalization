"""Read-only summary of completed replay observations. Never creates an env."""
import hashlib
import json
from pathlib import Path


def contact_with(contacts, other):
    return any(any((v or '').startswith('akita_black_bowl_1_g') for v in pair)
               and other in pair for pair in contacts)


def summarize(root):
    records = []
    for case in sorted(root.iterdir()):
        result = json.loads((case/'result.json').read_text())
        assert result['status'] == 'parity_passed'
        path = case/'object_contact_observations.json'
        rows = json.loads(path.read_text())
        assert len(rows) == result['verified_steps']
        left = [r['step'] for r in rows if contact_with(r['contacts'], 'gripper0_finger1_pad_collision')]
        right = [r['step'] for r in rows if contact_with(r['contacts'], 'gripper0_finger2_pad_collision')]
        both = sorted(set(left) & set(right))
        records.append(dict(case=case.name, steps=len(rows), reward=result['final_reward'],
            observation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            first_proxy_steps=result['first_proxy_steps'],
            fingerpad_contacts=dict(left_steps=len(left), right_steps=len(right),
                 simultaneous_steps=len(both), first_simultaneous=both[0] if both else None),
            final_bowl_xyz=rows[-1]['bowl_xyz'], final_plate_xyz=rows[-1]['plate_xyz']))
    assert len(records) == 12
    return dict(status='read_only_completed12_contact_summary', records=records,
        limits=['Fingerpad contact alone does not prove grasp or its absence',
                'Approach proxy is not a necessary condition: successful init3 never meets it',
                'Post-step contacts may miss transient contacts; no causal claim'],
        new_simulations=0, policy_calls=0, eight_task_progression='locked', fold02='locked')


if __name__ == '__main__':
    print(json.dumps(summarize(Path('/root/smolvla-eval-prep/recovery/controlled_cpu_object_contact_observers_v2_20261009'))))
