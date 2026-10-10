"""Observe real scheduler at construction; retain executed-update guard unchanged."""
from build_accounting_cpu_probe_v5 import build_source as build_v5


def build_source(original):
    source = build_v5(original)
    replacements = [
        ('accounting_cpu_probe_v5_20261011', 'accounting_cpu_probe_v6_20261011'),
        ("cfg.job_name = 'accounting_cpu_probe_v5'", "cfg.job_name = 'accounting_cpu_probe_v6'"),
        ('    original_update = trainer.update_policy', '''    schedulers = []
    original_factory = trainer.make_optimizer_and_scheduler
    def tracked_factory(*args, **kwargs):
        optimizer, scheduler = original_factory(*args, **kwargs)
        if schedulers:
            raise RuntimeError('repeated optimizer construction')
        schedulers.append(scheduler)
        return optimizer, scheduler
    trainer.make_optimizer_and_scheduler = tracked_factory
    original_update = trainer.update_policy'''),
        ('''        scheduler = kwargs['lr_scheduler']
        if scheduler.state_dict()['last_epoch'] != 20001:''', '''        if len(schedulers) != 1:
            raise RuntimeError('real scheduler not observed')
        if schedulers[0].last_epoch != 20001:'''),
        ('one_iteration_accounting_epilogue_cpu_v5_passed', 'one_iteration_accounting_epilogue_cpu_v6_passed'),
        ('        trainer.update_policy = original_update', '''        trainer.make_optimizer_and_scheduler = original_factory
        trainer.update_policy = original_update'''),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('unexpected real scheduler observer site')
        source = source.replace(old, new)
    compile(source, 'accounting_cpu_probe_v6', 'exec')
    return source
