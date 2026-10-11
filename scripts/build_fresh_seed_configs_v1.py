"""Produce registered replication configs without touching historical files."""
import copy
import json
from pathlib import Path

SEED = 2001
SOURCES = {
    'single': 'teacher_native_spatial_task0_single_40k_batch2_20261003.json',
    'joint': 'teacher_native_spatial_4task_homogeneous_160k_batch2_v1_20261009.json',
}
RUN_IDS = {
    'single': 'teacher_native_spatial_task0_freshseed2001_40k_batch2_v1_20261011',
    'joint': 'teacher_native_spatial_4task_freshseed2001_160k_batch2_v1_20261011',
}

def build(source, arm):
    if arm not in SOURCES:
        raise ValueError('unregistered arm')
    expected_steps = 40000 if arm == 'single' else 160000
    if (source['steps'] != expected_steps or source['seed'] != 1000
            or source['batch_size'] != 2 or source['env'] is not None
            or source['resume']):
        raise ValueError('unexpected historical recipe')
    result = copy.deepcopy(source)
    result.update(seed=SEED, job_name=RUN_IDS[arm],
                  output_dir='/root/smolvla-training-prep/results/training/' + RUN_IDS[arm])
    changed = {key for key in source if source[key] != result[key]}
    if changed != {'seed', 'job_name', 'output_dir'}:
        raise ValueError('unexpected recipe drift')
    return result

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=tuple(SOURCES), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = json.loads((root / 'configs' / SOURCES[args.arm]).read_text())
    print(json.dumps(build(source, args.arm), indent=2))

if __name__ == '__main__':
    main()
