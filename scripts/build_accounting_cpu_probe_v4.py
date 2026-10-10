"""Independent one-iteration epilogue diagnostic; not full-budget completion."""
from build_full_entry_cpu_probe_v2 import build_source as build_v2


def build_source(original):
    source = build_v2(original)
    replacements = [
        ('cpu_probe_v2_20261010', 'accounting_cpu_probe_v4_20261011'),
        ("cfg.job_name = 'full_entry_cpu_probe_v2'", "cfg.job_name = 'accounting_cpu_probe_v4'"),
        ('    updates = []', '''    updates = []
    accounting = []
    ended = []
    range_calls = []
    original_end = accelerator.end_training
    original_range = trainer.__dict__.get('range')
    had_range = 'range' in trainer.__dict__
    def one_iteration_range(*args):
        if args != (20000, 40000) or range_calls:
            raise RuntimeError('unexpected diagnostic loop boundary')
        range_calls.append(args)
        return range(20000, 20001)
    def checked_end():
        if len(updates) != 1 or accounting != [20001]:
            raise RuntimeError('missing post-update accounting')
        ended.append(True)
        return original_end()
    accelerator.end_training = checked_end
    trainer.range = one_iteration_range'''),
        ('        metrics, _ = original_update(*args, **kwargs)', '        metrics, output_dict = original_update(*args, **kwargs)'),
        ('        raise ProbeFinished()', '''        scheduler = kwargs['lr_scheduler']
        if scheduler.state_dict()['last_epoch'] != 20001:
            raise RuntimeError('scheduler did not advance exactly once')
        original_step = metrics.step
        def counted_step():
            result = original_step()
            accounting.append(metrics.steps)
            return result
        metrics.step = counted_step
        return metrics, output_dict'''),
        ('''            try:
                trainer.train(cfg, accelerator=accelerator)
            except ProbeFinished:
                pass
            else:
                raise RuntimeError('train returned without controlled one-update stop')''', '''            trainer.train(cfg, accelerator=accelerator)
        if accounting != [20001] or ended != [True] or range_calls != [(20000, 40000)]:
            raise RuntimeError('normal diagnostic return not verified')'''),
        ('full_installed_train_entry_cpu_v2_passed', 'one_iteration_accounting_epilogue_cpu_v4_passed'),
        ("stop='controlled exception after first real update before save/eval branches'", "stop='natural return after explicitly shortened diagnostic loop', accounting=accounting, end_training_calls=len(ended), configured_total_steps=cfg.steps"),
        ('        trainer.save_checkpoint = original_save', '''        trainer.save_checkpoint = original_save
        accelerator.end_training = original_end
        if had_range:
            trainer.range = original_range
        else:
            trainer.__dict__.pop('range', None)'''),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('unexpected observer replacement site')
        source = source.replace(old, new)
    compile(source, 'accounting_cpu_probe_v4', 'exec')
    return source
