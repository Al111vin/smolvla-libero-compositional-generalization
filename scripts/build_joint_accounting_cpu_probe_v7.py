"""Joint counterpart: observe restored scheduler, not single-only factory."""
from build_accounting_cpu_probe_v6 import build_source as build_v6


def build_source(original):
    source = build_v6(original)
    old = '''    original_factory = trainer.make_optimizer_and_scheduler
    def tracked_factory(*args, **kwargs):
        optimizer, scheduler = original_factory(*args, **kwargs)
        if schedulers:
            raise RuntimeError('repeated optimizer construction')
        schedulers.append(scheduler)
        return optimizer, scheduler
    trainer.make_optimizer_and_scheduler = tracked_factory'''
    new = '''    original_restore = trainer.load_training_state
    def tracked_restore(*args, **kwargs):
        step, optimizer, scheduler = original_restore(*args, **kwargs)
        if schedulers:
            raise RuntimeError('repeated scheduler restore')
        schedulers.append(scheduler)
        return step, optimizer, scheduler
    trainer.load_training_state = tracked_restore'''
    replacements = [
        (old, new),
        ('trainer.make_optimizer_and_scheduler = original_factory', 'trainer.load_training_state = original_restore'),
        ('teacher_native_spatial_task0_single_current_recipe_40k_batch2_v1/checkpoints/020000',
         'teacher_native_spatial_4task_homogeneous_160k_batch2_v1_20261009/checkpoints/080000'),
        ('accounting_cpu_probe_v6_20261011', 'joint_accounting_cpu_probe_v7_20261011'),
        ("cfg.job_name = 'accounting_cpu_probe_v6'", "cfg.job_name = 'joint_accounting_cpu_probe_v7'"),
        ('one_iteration_accounting_epilogue_cpu_v6_passed', 'joint_one_iteration_accounting_epilogue_cpu_v7_passed'),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('unexpected joint observer replacement site')
        source = source.replace(old, new)
    # These registered numeric boundaries occur only in the diagnostic guards.
    source = source.replace('20001', '80001').replace('20000', '80000').replace('40000', '160000')
    compile(source, 'joint_accounting_cpu_probe_v7', 'exec')
    return source
