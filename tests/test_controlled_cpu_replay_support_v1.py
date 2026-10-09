import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("support", Path(__file__).resolve().parents[1] / "scripts/controlled_cpu_replay_support_v1.py")
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


class ReplaySupportTests(unittest.TestCase):
    def test_reset_failure_never_retries(self):
        calls = []
        def fail():
            calls.append(1)
            raise RuntimeError("randomization failed")
        with self.assertRaisesRegex(RuntimeError, "randomization failed"):
            support.bounded_reset(SimpleNamespace(env=SimpleNamespace(reset=fail)))
        self.assertEqual(len(calls), 1)

    def test_order_and_early_done(self):
        events = []
        env = SimpleNamespace(env=SimpleNamespace(reset=lambda: events.append("reset") or "reset_obs"),
                              set_init_state=lambda state: events.append(("init", state)) or "init_obs",
                              step=lambda action: events.append("wait") or ("wait_obs", 0, True, {}))
        np = SimpleNamespace(float32="float32", zeros=lambda n, dtype: [0] * n,
                             random=SimpleNamespace(seed=lambda s: events.append(("seed", s))))
        self.assertEqual(support.initialize_replay(env, "state", 12351, np), ("wait_obs", 1))
        self.assertEqual(events, ["reset", ("init", "state"), "wait", ("seed", 12351)])

    def test_no_renderer(self):
        kwargs = support.headless_kwargs("task.bddl")
        self.assertFalse(kwargs["use_camera_obs"])
        self.assertFalse(kwargs["has_renderer"])
        self.assertFalse(kwargs["has_offscreen_renderer"])


if __name__ == "__main__":
    unittest.main()
