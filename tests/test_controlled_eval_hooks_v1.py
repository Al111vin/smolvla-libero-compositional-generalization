import unittest
from types import SimpleNamespace
from scripts.controlled_eval_hooks_v1 import install_hooks


class HookTests(unittest.TestCase):
    def test_model_factory_capture_and_restore(self):
        policy=object()
        class OriginalPolicy:
            @staticmethod
            def from_pretrained(path):
                return policy
        v3=SimpleNamespace(SmolVLAPolicy=OriginalPolicy, OffScreenRenderEnv=lambda:None,
            make_pre_post_processors=lambda:None)
        seen=[]
        restore=install_hooks(v3,lambda *a:None,model_capture=seen.append)
        self.assertIs(v3.SmolVLAPolicy.from_pretrained("checkpoint"),policy)
        self.assertEqual(seen,[policy])
        restore()
        self.assertIs(v3.SmolVLAPolicy,OriginalPolicy)

    def test_seed_capture_order_and_restore(self):
        events=[]
        env=object()
        def factory():
            events.append("env")
            return env
        def pre(frame):
            events.append("pre")
            return frame
        def processors():
            return pre, "post"
        v3=SimpleNamespace(np=SimpleNamespace(random=SimpleNamespace(seed=lambda s:events.append(("seed",s)))),
                           OffScreenRenderEnv=factory,make_pre_post_processors=processors)
        def capture(e,f,b,p,q):
            self.assertIs(e,env)
            events.append("capture")
        restore=install_hooks(v3,capture)
        wrapped,post=v3.make_pre_post_processors()
        v3.OffScreenRenderEnv()
        self.assertEqual(wrapped("input"),"input")
        wrapped("next")
        self.assertEqual(events,[("seed",12351),"env","pre","capture","pre"])
        self.assertEqual(post,"post")
        restore()
        self.assertIs(v3.OffScreenRenderEnv,factory)
        self.assertIs(v3.make_pre_post_processors,processors)


if __name__ == "__main__":
    unittest.main()
