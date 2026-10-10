"""Preserve v1; adapt only the independently diagnosed metric observer."""
import hashlib


def build_source(source):
    if hashlib.sha256(source.encode()).hexdigest() != '61178d53edc169fc757f89f7b9aba6fbcc6c1bb6d26327a7b1c8c9347471cfd0':
        raise ValueError('full entry v1 source drift')
    replacements = [
        ('cpu_probe_v1_20261010', 'cpu_probe_v2_20261010'),
        ("cfg.job_name = 'full_entry_cpu_probe_v1'", "cfg.job_name = 'full_entry_cpu_probe_v2'"),
        ("        if not math.isfinite(metrics.loss) or not math.isfinite(metrics.grad_norm):",
         "        loss, grad_norm = metrics.loss.val, metrics.grad_norm.val\n        if not math.isfinite(loss) or not math.isfinite(grad_norm):"),
        ('dict(loss=metrics.loss, grad_norm=metrics.grad_norm, pre_update_lr=pre_lr)',
         'dict(loss=loss, grad_norm=grad_norm, pre_update_lr=pre_lr)'),
        ('full_installed_train_entry_cpu_v1_passed', 'full_installed_train_entry_cpu_v2_passed'),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('unexpected v1 observer site')
        source = source.replace(old, new)
    compile(source, 'full_entry_cpu_v2', 'exec')
    return source
