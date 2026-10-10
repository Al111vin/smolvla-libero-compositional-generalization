"""Independent joint midpoint counterpart to the single full-entry probe."""
from build_full_entry_cpu_probe_v2 import build_source as build_single


def build_source(source):
    source = build_single(source)
    replacements = [
        ('teacher_native_spatial_task0_single_current_recipe_40k_batch2_v1/checkpoints/020000',
         'teacher_native_spatial_4task_homogeneous_160k_batch2_v1_20261009/checkpoints/080000'),
        ('cpu_probe_v2_20261010', 'joint_cpu_probe_v3_20261010'),
        ("cfg.job_name = 'full_entry_cpu_probe_v2'", "cfg.job_name = 'joint_full_entry_cpu_probe_v3'"),
        ('full_installed_train_entry_cpu_v2_passed', 'joint_full_installed_train_entry_cpu_v3_passed'),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('unexpected single counterpart site')
        source = source.replace(old, new)
    compile(source, 'joint_full_entry_cpu_v3', 'exec')
    return source
