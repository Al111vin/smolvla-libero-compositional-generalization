from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_fresh_seed_eval_v1 as runner


class FreshSeedEvalPreflightTests(unittest.TestCase):
    def test_conflicting_process_rejected_before_results_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            lock = root / "gpu.lock"
            lock.touch()
            results = root / "new-results"
            manifest = {
                "root": str(results),
                "scripts_root": str(root),
                "code_hashes": {},
                "gpu_lock": str(lock),
                "pinned_files": {},
                "models": {},
                "conflict_tokens": ["existing-policy-evaluation"],
                "minimum_free_bytes": 1,
                "cwd": str(root),
            }
            with patch.object(
                runner.subprocess,
                "check_output",
                side_effect=[
                    "",  # no NVIDIA compute process
                    "PID ARGS\n123 existing-policy-evaluation --held-gpu",
                ],
            ):
                with self.assertRaisesRegex(RuntimeError, "conflicting process"):
                    runner.main(manifest)
            self.assertFalse(results.exists())


if __name__ == "__main__":
    unittest.main()
