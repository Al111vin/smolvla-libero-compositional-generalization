"""Bounded synthetic CPU probe; no model, dataset files, or training execution."""
import json
import torch
from torch.utils.data import DataLoader, TensorDataset


def main():
    torch.manual_seed(1000)
    state = torch.get_rng_state().clone()
    dataset = TensorDataset(torch.arange(200))
    default = DataLoader(dataset, batch_size=2, num_workers=0)
    iter(default)
    default_changed = not torch.equal(state, torch.get_rng_state())
    assert default_changed, "Default iterator behavior changed; review integration"
    torch.set_rng_state(state)
    results = {}
    for workers in (0, 4):
        arms = []
        for _ in range(2):
            generator = torch.Generator().manual_seed(2000)
            loader = DataLoader(dataset, batch_size=2, shuffle=True,
                                num_workers=workers, generator=generator)
            arms.append([batch[0].tolist() for batch in loader])
            assert torch.equal(state, torch.get_rng_state())
        assert arms[0] == arms[1] and len(arms[0]) == 100
        assert sorted(x for batch in arms[0] for x in batch) == list(range(200))
        results[str(workers)] = {"batches": 100, "arms_identical": True,
                                 "global_torch_rng_preserved": True}
    print(json.dumps({"status": "synthetic_cpu_only", "torch_version": torch.__version__,
                      "default_iterator_changes_rng": default_changed,
                      "worker_results": results,
                      "real_dataset_and_accelerator_tested": False}, sort_keys=True))


if __name__ == "__main__":
    main()
