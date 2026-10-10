"""Preserve v4; adapt spy attachment to actual MetricsTracker attribute rules."""
from build_accounting_cpu_probe_v4 import build_source as build_v4


def build_source(original):
    source = build_v4(original)
    for old, new in [
        ('accounting_cpu_probe_v4_20261011', 'accounting_cpu_probe_v5_20261011'),
        ("cfg.job_name = 'accounting_cpu_probe_v4'", "cfg.job_name = 'accounting_cpu_probe_v5'"),
        ('metrics.step = counted_step', "metrics.__dict__['step'] = counted_step"),
        ('one_iteration_accounting_epilogue_cpu_v4_passed', 'one_iteration_accounting_epilogue_cpu_v5_passed'),
    ]:
        if source.count(old) != 1:
            raise ValueError('unexpected accounting observer site')
        source = source.replace(old, new)
    compile(source, 'accounting_cpu_probe_v5', 'exec')
    return source
