"""Prespecified20-init paired midpoint comparison; no best-checkpoint search."""
from controlled_eval_schedule_v1 import validate_protocol, environment_signature, validate_rollout


def schedule():
    rows = []
    for index in range(20):
        for model in ('single20k', 'joint80k'):
            rows.append(dict(model=model, kind='paired', init=index, repeat=0,
                cli_seed=12345+index, key=f'paired_{index}_r0_{model}'))
    assert len(rows) == len({r['key'] for r in rows}) == 40
    return rows
