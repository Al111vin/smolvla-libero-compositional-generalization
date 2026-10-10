"""CPU reconstruction identity tests; no policy or parameter updates."""
import unittest
from types import SimpleNamespace as NS
import torch
from accelerate import Accelerator
from resume_training_hooks_v2 import registered_resume_hooks


class Dataset(torch.utils.data.Dataset):
    repo_id = "identity_probe"
    hf_dataset = NS(data=NS(column=lambda key: NS(to_pylist=lambda: [0] * 5068)))
    def __len__(self):
        return 5068
    def __getitem__(self, i):
        return i


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = NS(resume=True, steps=40000, batch_size=2, num_workers=4, env=None,
            dataset=NS(repo_id=Dataset.repo_id, streaming=False, image_transforms=NS(enable=False)),
            policy=NS(push_to_hub=False))
        self.trainer = NS(make_optimizer_and_scheduler=lambda *a: None,
            load_training_state=lambda *a: None, cycle=lambda x: iter(x), update_policy=lambda *a: None)
        self.dataset = Dataset()

    def original(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=2, num_workers=4,
                                          prefetch_factor=2, sampler=None, shuffle=True)

    def reconstruct(self, loader, **overrides):
        kwargs = dict(batch_sampler=loader.batch_sampler, generator=loader.generator,
                      num_workers=4, prefetch_factor=2)
        kwargs.update(overrides)
        return torch.utils.data.DataLoader(self.dataset, **kwargs)

    def test_actual_accelerator_single_reconstruction(self):
        with registered_resume_hooks(self.trainer, self.cfg, factor=0.5):
            loader = self.original()
            prepared = Accelerator(cpu=True).prepare(loader)
            self.assertEqual(len(prepared), 20000)
            self.assertIs(prepared.batch_sampler, loader.batch_sampler)
            self.assertIs(prepared.generator, loader.generator)

    def test_wrong_generator_and_sampler_rejected(self):
        with registered_resume_hooks(self.trainer, self.cfg, factor=0.5):
            loader = self.original()
            for changes in ({"generator": torch.Generator().manual_seed(2000)},
                            {"batch_sampler": [[0, 1]]}):
                with self.assertRaisesRegex(ValueError, "unregistered or repeated"):
                    self.reconstruct(loader, **changes)
            self.reconstruct(loader)

    def test_repeat_reconstruction_rejected(self):
        with registered_resume_hooks(self.trainer, self.cfg, factor=0.5):
            loader = self.original()
            self.reconstruct(loader)
            with self.assertRaisesRegex(ValueError, "unregistered or repeated"):
                self.reconstruct(loader)

    def test_same_repo_different_dataset_rejected(self):
        with registered_resume_hooks(self.trainer, self.cfg, factor=0.5):
            loader = self.original()
            with self.assertRaisesRegex(ValueError, "unregistered or repeated"):
                torch.utils.data.DataLoader(Dataset(), batch_sampler=loader.batch_sampler,
                    generator=loader.generator, num_workers=4, prefetch_factor=2)


if __name__ == "__main__":
    unittest.main()
