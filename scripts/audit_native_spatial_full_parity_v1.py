"""All-frame CPU parity audit; prints lightweight summary, writes no data."""
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    import h5py
    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    manifest_path = Path("/root/smolvla-training-prep/recovery/task_scaling_native_spatial_20261003/conversion_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete"
    root = Path("/root/smolvla-training-prep/datasets/lerobot/libero_spatial_tasks0_3_native_20261003_v1")
    dataset = LeRobotDataset("local/libero_spatial_tasks0_3_native_v1", root=root, download_videos=False)
    assert dataset.num_episodes == 200 and len(dataset) == 22709
    assert manifest["task_index_by_libero_id"] == {str(t): t for t in range(4)}
    row = 0
    episode_global = 0
    counts = {}
    sources = {}
    for task_id, task in enumerate(manifest["tasks"]):
        path = Path(task["path"])
        sources[str(task_id)] = digest(path)
        assert sources[str(task_id)] == task["sha256"]
        assert len(task["episodes_detail"]) == 50
        checked = 0
        with h5py.File(path, "r") as source:
            for episode in task["episodes_detail"]:
                demo = source["data"][episode["demo_key"]]
                frames = episode["frames"]
                assert len(demo["actions"]) == frames
                for f in range(frames):
                    item = dataset[row]
                    context = (task_id, episode["demo_key"], f, row)
                    expected_state = np.concatenate([np.asarray(demo[k][f], dtype=np.float32) for k in
                        ("obs/joint_states", "obs/ee_pos", "obs/ee_ori", "obs/gripper_states")])
                    for feature, expected in (("observation.state", expected_state),
                                              ("action", np.asarray(demo["actions"][f], dtype=np.float32))):
                        actual = np.asarray(item[feature], dtype=np.float32)
                        assert np.isfinite(actual).all() and np.isfinite(expected).all(), context
                        assert np.array_equal(actual, expected), (context, feature)
                    assert expected_state.shape == (15,) and item["action"].shape == (7,), context
                    assert int(item["task_index"]) == task_id, context
                    assert int(item["episode_index"]) == episode_global, context
                    assert int(item["frame_index"]) == f, context
                    for feature, key in (("observation.images.agentview", "obs/agentview_rgb"),
                                         ("observation.images.wrist", "obs/eye_in_hand_rgb")):
                        image = item[feature].detach().cpu()
                        assert torch.isfinite(image).all(), context
                        assert tuple(image.shape) == (3, 128, 128), context
                        actual = (image * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
                        expected = np.asarray(demo[key][f], dtype=np.uint8)
                        assert np.array_equal(actual, expected), (context, feature)
                    row += 1
                    checked += 1
                episode_global += 1
        assert checked == task["frames"]
        counts[str(task_id)] = checked
    assert row == len(dataset) and episode_global == dataset.num_episodes
    print(json.dumps(dict(status="passed", frames=row, images=2*row, episodes=episode_global,
                         frames_by_task=counts, source_sha256=sources,
                         manifest_sha256=digest(manifest_path), all_state_action_image_checks_exact=True,
                         all_episode_frame_task_indices_exact=True, cpu_only=True), indent=2))


if __name__ == "__main__":
    main()
