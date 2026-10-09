"""Read-only full40 raw evidence audit. No environment or policy creation."""
import csv
import hashlib
import json
from pathlib import Path
from midpoint_eval_schedule_v1 import schedule, validate_protocol, validate_rollout, environment_signature


def audit(root, manifest):
    if (root/'exit_code').read_text().strip() != '0':
        raise ValueError('execution incomplete or failed')
    records=[]; signatures={}; loaded={}
    for model in manifest['models'].values():
        path=Path(model['path'])/'model.safetensors'
        if hashlib.sha256(path.read_bytes()).hexdigest() != model['sha']:
            raise ValueError('checkpoint model hash mismatch')
    for row in schedule():
        target=root/row['key']; model=manifest['models'][row['model']]
        protocol=json.loads((target/'protocol.json').read_text())
        validate_protocol(row,protocol)
        if protocol['checkpoint'] != model['path'] or protocol['model_sha'] != model['sha']:
            raise ValueError('protocol checkpoint mismatch')
        if protocol['evaluator_sha'] != '89fd36a89dc45a56382219d4c3f6e5d12a3b1689abb3440ec29be346815af7cb':
            raise ValueError('frozen evaluator mismatch')
        if protocol['wrapper_sha'] != manifest['code_hashes']['eval_controlled_environment_v1.py']:
            raise ValueError('wrapper hash mismatch')
        capture=json.loads((target/'first_input.json').read_text())
        for k in ('model_arrays','simulator_state','images','frame','input','processor','loaded_model','rng'):
            if not capture.get(k): raise ValueError('missing input provenance '+k)
        signature=environment_signature(capture)
        if row['init'] in signatures and signature != signatures[row['init']]:
            raise ValueError('paired starting environment mismatch')
        signatures[row['init']]=signature
        if row['model'] in loaded and capture['loaded_model'] != loaded[row['model']]:
            raise ValueError('loaded policy differs within condition')
        loaded[row['model']]=capture['loaded_model']
        summaries=list(target.glob('*_summary.csv')); actions=list(target.glob('*_actions.csv'))
        if len(summaries) != 1 or len(actions) != 1: raise ValueError('raw files incomplete')
        with summaries[0].open() as f: summary=list(csv.DictReader(f))
        with actions[0].open() as f: action_rows=list(csv.DictReader(f))
        if len(summary) != 1: raise ValueError('summary count')
        validate_rollout(row,summary[0],action_rows)
        result=dict(row,success=summary[0]['success']=='True',steps=int(summary[0]['steps']))
        if json.loads((root/(row['key']+'.verified.json')).read_text()) != result:
            raise ValueError('verified record differs from raw CSV')
        records.append(result)
    if json.loads((root/'completion.json').read_text()) != records:
        raise ValueError('completion differs from complete40 raw records')
    return dict(status='complete40_raw_audit_passed',records=records,
        successes={m:sum(r['success'] for r in records if r['model']==m) for m in manifest['models']},
        paired_initial_environments_matched=20,
        limits=['Previously used initializations','No fixed-init stability gate replacement','No causal interference claim'])
