"""Read-only counterexamples to necessary/sufficient stage claims."""
import json
from pathlib import Path


def audit(records):
    if len({r['case'] for r in records}) != len(records):
        raise ValueError('duplicate case')
    result = {'success_without_approach': [], 'success_without_bilateral_contact': [],
              'failure_with_lift_and_transport': [], 'failure_with_bilateral_contact': []}
    for r in records:
        if r['reward'] not in (0, 1):
            raise ValueError('unknown official reward')
        p, c = r['first_proxy_steps'], r['fingerpad_contacts']
        if r['reward'] == 1:
            if p['official_success'] is None:
                raise ValueError('success record lacks success step')
            if p['approach'] is None:
                result['success_without_approach'].append(r['case'])
            if c['simultaneous_steps'] == 0:
                result['success_without_bilateral_contact'].append(r['case'])
        else:
            if p['official_success'] is not None:
                raise ValueError('reward contradicts success step')
            if p['lift'] is not None and p['transport'] is not None:
                result['failure_with_lift_and_transport'].append(r['case'])
            if c['simultaneous_steps'] > 0:
                result['failure_with_bilateral_contact'].append(r['case'])
    return result


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    source = root / 'results/teacher_native_spatial_4task_cpu_contact_review_v1_20261009.json'
    print(json.dumps(audit(json.loads(source.read_text())['records']), indent=2))
