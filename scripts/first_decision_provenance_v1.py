"""Read-only provenance helpers; this module never launches a model or environment."""
import copy
import hashlib
import json


def digest_tree(value):
    """Hash typed nested values, including tensor dtype/shape and exact bytes."""
    h = hashlib.sha256()

    def part(data):
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)

    def add(x):
        if isinstance(x, dict):
            part(b"dict")
            part(str(len(x)).encode())
            if not all(isinstance(k, str) for k in x):
                raise TypeError("Only string mapping keys supported")
            for k in sorted(x):
                add(k)
                add(x[k])
        elif isinstance(x, (list, tuple)):
            part(type(x).__name__.encode())
            part(str(len(x)).encode())
            for v in x:
                add(v)
        elif hasattr(x, "detach"):
            t = x.detach().cpu().contiguous()
            part(b"torch_tensor")
            part(str(t.dtype).encode())
            part(json.dumps(list(t.shape)).encode())
            # Byte view supports bfloat16 without lossy conversion.
            import torch
            part(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif hasattr(x, "dtype") and hasattr(x, "tobytes"):
            if x.dtype.hasobject:
                raise TypeError("Object arrays cannot be hashed reproducibly")
            part(b"numpy_value")
            part(str(x.dtype).encode())
            part(json.dumps(list(x.shape)).encode())
            part(x.tobytes(order="C"))
        elif isinstance(x, bytes):
            part(b"bytes")
            part(x)
        elif x is None or isinstance(x, (str, bool, int, float)):
            part(type(x).__name__.encode())
            part(json.dumps(x, allow_nan=False).encode())
        else:
            raise TypeError(f"Unsupported provenance type: {type(x).__name__}")

    add(value)
    return h.hexdigest()


def paired_first_calls(policy, batch, snapshot_rng, restore_rng, infer):
    """Reset queue and RNG before each call; isolate mutable batch copies."""
    original = digest_tree(batch)
    state = snapshot_rng()
    results = []
    for _ in range(2):
        policy.reset()
        restore_rng(state)
        local = copy.deepcopy(batch)
        before_rng = snapshot_rng()
        results.append({"input_digest": digest_tree(local),
                        "rng_digest": digest_tree(before_rng),
                        "output": infer(policy, local)})
    if digest_tree(batch) != original:
        raise RuntimeError("Captured input mutated")
    if results[0]["input_digest"] != results[1]["input_digest"]:
        raise RuntimeError("Restored inputs differ")
    if results[0]["rng_digest"] != results[1]["rng_digest"]:
        raise RuntimeError("RNG restoration failed")
    return results
