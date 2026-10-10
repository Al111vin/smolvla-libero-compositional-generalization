import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from resume_entry_contract_v1 import require_resume_entry_contract


class EntryContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name).resolve()
        checkpoint = root / 'old'; checkpoint.mkdir()
        self.paths = dict(checkpoint=checkpoint, output=root/'new', log=root/'run.log',
                          pid=root/'run.pid', exit_code=root/'run.exit')
        self.cfg = NS(resume=True, steps=160000, batch_size=2, num_workers=4, env=None,
            output_dir=self.paths['output'], checkpoint_path=checkpoint,
            policy=NS(pretrained_path=checkpoint/'pretrained_model', push_to_hub=False),
            dataset=NS(streaming=False, image_transforms=NS(enable=False)), wandb=NS(enable=False))

    def tearDown(self):
        self.temp.cleanup()

    def test_one_process_passes_without_creating_paths(self):
        require_resume_entry_contract(self.cfg, NS(num_processes=1), **self.paths)
        self.assertFalse(self.paths['output'].exists())

    def test_distributed_zero_and_noninteger_rejected(self):
        for count in (0, 2, 4, True, 1.0):
            with self.assertRaises(ValueError):
                require_resume_entry_contract(self.cfg, NS(num_processes=count), **self.paths)

    def test_checkpoint_output_and_policy_aliases_rejected(self):
        for obj, key in ((self.cfg, 'output_dir'), (self.cfg, 'checkpoint_path'),
                         (self.cfg.policy, 'pretrained_path')):
            old = getattr(obj, key); setattr(obj, key, Path('/unexpected'))
            with self.assertRaises(ValueError):
                require_resume_entry_contract(self.cfg, NS(num_processes=1), **self.paths)
            setattr(obj, key, old)

    def test_existing_output_preserved(self):
        self.paths['output'].mkdir()
        with self.assertRaises(FileExistsError):
            require_resume_entry_contract(self.cfg, NS(num_processes=1), **self.paths)
        self.assertTrue(self.paths['output'].is_dir())


if __name__ == '__main__':
    unittest.main()
