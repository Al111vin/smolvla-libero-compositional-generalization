import random
import unittest
from scripts.first_decision_provenance_v1 import digest_tree, paired_first_calls


class ProvenanceTests(unittest.TestCase):
    def test_mapping_order_and_type(self):
        self.assertEqual(digest_tree({"a": 1, "b": 2}), digest_tree({"b": 2, "a": 1}))
        self.assertNotEqual(digest_tree([1]), digest_tree((1,)))
        self.assertNotEqual(digest_tree(1), digest_tree("1"))

    def test_unknown_fails_closed(self):
        with self.assertRaises(TypeError):
            digest_tree(object())

    def test_structure_and_bytes(self):
        self.assertNotEqual(digest_tree({"a": {"b": 1}, "c": 2}),
                            digest_tree({"a": {"b": 1, "c": 2}}))
        self.assertNotEqual(digest_tree([b"ab", b"c"]), digest_tree([b"a", b"bc"]))

    def test_bad_rng_restore_fails(self):
        class Policy:
            def reset(self):
                pass
        counter = [0]
        def snapshot():
            counter[0] += 1
            return counter[0]
        with self.assertRaisesRegex(RuntimeError, "RNG restoration failed"):
            paired_first_calls(Policy(), {}, snapshot, lambda state: None,
                               lambda policy, batch: 0)

    def test_restores_rng_queue_and_batch(self):
        class Policy:
            def reset(self):
                self.queue = []
        p = Policy()
        batch = {"x": [1]}

        def infer(policy, local):
            local["x"].append(2)
            policy.queue.append(random.random())
            return policy.queue[0]

        result = paired_first_calls(p, batch, random.getstate, random.setstate, infer)
        self.assertEqual(result[0]["output"], result[1]["output"])
        self.assertEqual(batch, {"x": [1]})


if __name__ == "__main__":
    unittest.main()
