import copy
import importlib.util
import unittest


SPEC = importlib.util.spec_from_file_location(
    "task1_seed2001_builder",
    "scripts/build_teacher_native_spatial_task1_freshseed2001_40k_v1.py",
)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class Task1Seed2001BuilderTests(unittest.TestCase):
    def test_config_only_changes_registered_task_slice_and_paths(self):
        source_bytes = BUILDER.SOURCE.read_bytes()
        self.assertEqual(BUILDER.sha256(source_bytes), BUILDER.EXPECTED_SOURCE_SHA256)
        source = __import__("json").loads(source_bytes)
        result = BUILDER.load_and_build()

        self.assertEqual(
            BUILDER.changed_paths(source, result),
            {"dataset.episodes", "job_name", "output_dir"},
        )
        self.assertEqual(result["dataset"]["episodes"], list(range(50, 100)))
        self.assertEqual(result["seed"], 2001)
        self.assertEqual(result["steps"], 40000)
        self.assertEqual(result["batch_size"], 2)
        self.assertEqual(result["policy"]["pretrained_path"], source["policy"]["pretrained_path"])
        self.assertEqual(result["optimizer"], source["optimizer"])
        self.assertEqual(result["scheduler"], source["scheduler"])
        self.assertFalse(result["resume"])
        self.assertIsNone(result["checkpoint_path"])

    def test_does_not_mutate_input(self):
        source = __import__("json").loads(BUILDER.SOURCE.read_text())
        before = copy.deepcopy(source)
        BUILDER.build_config(source)
        self.assertEqual(source, before)

    def test_refuses_wrong_episode_slice(self):
        source = __import__("json").loads(BUILDER.SOURCE.read_text())
        source["dataset"]["episodes"] = list(range(1, 51))
        with self.assertRaises(ValueError):
            BUILDER.build_config(source)


if __name__ == "__main__":
    unittest.main()
