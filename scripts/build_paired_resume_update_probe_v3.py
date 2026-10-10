"""Versioned paired one-update CPU diagnostic source generator."""
from build_actual_resume_update_probe_v2 import build_source as build_v2


def build_source(source, factor):
    if factor not in (1.0, 0.5):
        raise ValueError("unregistered factor")
    source = build_v2(source)
    source = source.replace("factor=0.5)", f"factor={factor})")
    source = source.replace("assert pre_lr == 0.000025625", f"assert pre_lr == {0.00005125 * factor!r}")
    anchor = "    metrics, _ = trainer.update_policy"
    assert source.count(anchor) == 1
    capture = '''    import hashlib
    def tensor_hash(value):
        return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    pre_parameter_sha = tensor_hash(before)
    pre_rng_sha = tensor_hash(torch.get_rng_state())
    batch_tensor_shas = {k: tensor_hash(v) for k, v in batch.items() if isinstance(v, torch.Tensor)}
'''
    source = source.replace(anchor, capture + anchor)
    source = source.replace('"status": "one_actual_cpu_resume_update_v2_passed",',
        '"status": "paired_actual_cpu_resume_update_v3_passed", "factor": ' + repr(factor) + ', "pre_parameter_sha256": pre_parameter_sha, "pre_rng_sha256": pre_rng_sha, "processed_batch_tensor_shas": batch_tensor_shas,')
    compile(source, "paired_resume_update_v3", "exec")
    return source
