"""Re-read complete96 raw evidence on CPU before reporting progression gates."""
import argparse
import hashlib
import json
from pathlib import Path

from run_scaling_eval_v1 import unique_csv, validate_checkpoint
from scaling_eval_schedule_v1 import schedule, validate_protocol, validate_rollout, gate
from capture_reset_render_provenance_v1 import EVALUATOR_SHA


def audit(root, manifest):
    root = Path(root)
    if (root / 'exit_code').read_text().strip() != '0':
        raise ValueError('Evaluation has not succeeded')
    validate_checkpoint(Path(manifest['checkpoint']), manifest['model_sha'])
    completion = json.loads((root / 'completion.json').read_text())
    records, starts, loaded_models, repeat_traces = [], {}, set(), {}
    expected = schedule()
    if {p.name for p in root.glob('*.verified.json')} != {
            r['key'] + '.verified.json' for r in expected}:
        raise ValueError('Missing or unexpected verified records')
    for row in expected:
        target = root / row['key']
        protocol = json.loads((target / 'protocol.json').read_text())
        validate_protocol(row, protocol)
        if (protocol['checkpoint'] != manifest['checkpoint'] or
                protocol['model_sha'] != manifest['model_sha'] or
                protocol['evaluator_sha'] != EVALUATOR_SHA or
                protocol['wrapper_sha'] != manifest['code_hashes']['eval_scaling_environment_v1.py']):
            raise ValueError('Model or code provenance mismatch')
        summaries = unique_csv(target, '*_summary.csv')
        if len(summaries) != 1:
            raise ValueError('Ambiguous summary')
        actions = unique_csv(target, '*_actions.csv')
        success = validate_rollout(row, summaries[0], actions)
        record = dict(row, success=success, steps=int(summaries[0]['steps']))
        if json.loads((root / (row['key'] + '.verified.json')).read_text()) != record:
            raise ValueError('Verified record disagrees with raw CSV')
        capture = json.loads((target / 'first_input.json').read_text())
        fields = ('model_arrays', 'simulator_state', 'images', 'frame', 'input',
                  'processor', 'loaded_model', 'rng')
        if any(not capture.get(k) for k in fields):
            raise ValueError('Incomplete first-input provenance')
        signature = {k: capture[k] for k in fields}
        key = (row['task_id'], row['init'])
        if key in starts and starts[key] != signature:
            raise ValueError('Repeated initialization provenance mismatch')
        starts[key] = signature
        loaded_models.add(json.dumps(capture['loaded_model'], sort_keys=True))
        if row['init'] == 3:
            path = next(target.glob('*_actions.csv'))
            repeat_traces.setdefault(str(row['task_id']), []).append(
                hashlib.sha256(path.read_bytes()).hexdigest())
        records.append(record)
    if len(loaded_models) != 1:
        raise ValueError('Loaded policy differs between rollouts')
    result = gate({r['key']: r['success'] for r in records})
    if completion != {'records': records, 'gate': result}:
        raise ValueError('Completion disagrees with full raw evidence')
    return dict(status='complete96_full_evidence_revalidated', records=records,
                gate=result, model_sha=manifest['model_sha'],
                same_loaded_model_all96=True,
                repeated_initialization_provenance_equal=True,
                init3_action_csv_hashes=repeat_traces,
                init3_action_repeat_identical={k: len(set(v)) == 1
                                              for k, v in repeat_traces.items()},
                scope='Used-init diagnostic; no historicalV3 parity, unused-init '
                      'generalization, post-success persistence, or causal '
                      'interference claim without matched single-task baselines')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, json.loads(Path(args.manifest).read_text())), indent=2))
