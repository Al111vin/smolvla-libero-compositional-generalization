"""Independent hash-pinned reconstruction fix; preserves failed v1 hooks."""
import hashlib


def build_source(source):
    if hashlib.sha256(source.encode()).hexdigest() != "cc0cd545ca58c5c898d26963d9c933e69f99087d962c32b050272a665c55c9ee":
        raise ValueError("v1 hook source drift")
    source = source.replace("    loader_calls = []", "    loader_calls = []\n    reconstructions = []")
    anchor = '        if loader_calls or args or kwargs.get("batch_size") != 2 or kwargs.get("sampler") is not None:'
    addition = '''        if loader_calls:
            registered_dataset, registered_sampler, registered_generator = loader_calls[0]
            if (not reconstructions and not args and dataset is registered_dataset
                    and kwargs.get("batch_sampler") is registered_sampler
                    and kwargs.get("generator") is registered_generator
                    and kwargs.get("num_workers") == 4
                    and kwargs.get("prefetch_factor") == 2):
                reconstructions.append(True)
                return original_loader_init(self, dataset, *args, **kwargs)
            raise ValueError("unregistered or repeated loader reconstruction")
'''
    if source.count(anchor) != 1:
        raise ValueError("unexpected reconstruction site")
    source = source.replace(anchor, addition + anchor)
    source = source.replace("        loader_calls.append(True)",
        '        loader_calls.append((dataset, sampler, kwargs["generator"]))')
    compile(source, "resume_training_hooks_v2", "exec")
    return source
