import numpy as np

from audit_task1_pose_target_windows_v1 import first_sustained


def test_first_sustained_finds_first_three_frame_close():
    values = np.array([-1, -1, 0.2, 0.6, 0.8, 0.9, -1], dtype=np.float32)
    assert first_sustained(values, lambda x: x > 0.5, 3) == 3


def test_first_sustained_returns_none_when_not_long_enough():
    values = np.array([-1, 0.6, -1, 0.7], dtype=np.float32)
    assert first_sustained(values, lambda x: x > 0.5, 3) is None


def test_release_search_can_start_after_close():
    values = np.array([-1, -1, 0.7, 0.8, 0.9, -1, -1, -1], dtype=np.float32)
    close = first_sustained(values, lambda x: x > 0.5, 3)
    release = first_sustained(values[close + 1 :], lambda x: x < -0.5, 3)
    assert close == 2
    assert release == 2


if __name__ == "__main__":
    for name in list(globals()):
        if name.startswith("test_"):
            globals()[name]()
    print("ALL_POSE_TARGET_WINDOW_TESTS_OK")
