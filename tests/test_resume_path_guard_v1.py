import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from resume_path_guard_v1 import require_independent_absent_paths


class PathGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        checkpoint = self.root / "old_checkpoint"
        checkpoint.mkdir()
        self.paths = dict(checkpoint=checkpoint, output=self.root / "new_output",
                          log=self.root / "new.log", pid=self.root / "new.pid",
                          exit_code=self.root / "new.exit")

    def tearDown(self):
        self.temp.cleanup()

    def test_absent_independent_paths_no_creation(self):
        require_independent_absent_paths(**self.paths)
        self.assertTrue(all(not p.exists() for k, p in self.paths.items() if k != "checkpoint"))

    def test_existing_paths_and_dangling_symlinks_rejected(self):
        for key in ("output", "log", "pid", "exit_code"):
            p = self.paths[key]
            p.touch()
            with self.assertRaises(FileExistsError):
                require_independent_absent_paths(**self.paths)
            p.unlink()
        self.paths["log"].symlink_to(self.root / "missing")
        with self.assertRaises(FileExistsError):
            require_independent_absent_paths(**self.paths)

    def test_overlap_alias_and_output_internal_log_rejected(self):
        for changes in ({"log": self.paths["pid"]},
                        {"output": self.paths["checkpoint"] / "child"},
                        {"log": self.paths["output"] / "train.log"},
                        {"output": Path("relative")}):
            with self.assertRaises(ValueError):
                require_independent_absent_paths(**{**self.paths, **changes})

    def test_symlink_parent_rejected(self):
        link = self.root / "redirect"
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            require_independent_absent_paths(**{**self.paths, "log": link / "unused.log"})


if __name__ == "__main__":
    unittest.main()
