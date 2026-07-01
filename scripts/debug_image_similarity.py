from pathlib import Path
import imageio.v2 as imageio
import numpy as np

img_dir = Path("results/debug_images")

pairs = [
    ("agentview", "dataset_agentview_rgb.png", "env_agentview_image.png"),
    ("wrist", "dataset_wrist_rgb.png", "env_wrist_image.png"),
]

def mse(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(((a - b) ** 2).mean())

for name, dataset_file, env_file in pairs:
    d = imageio.imread(img_dir / dataset_file)
    e = imageio.imread(img_dir / env_file)

    print("=" * 80)
    print(name)
    print("dataset shape:", d.shape, "env shape:", e.shape)

    candidates = {
        "original": e,
        "vertical_flip": e[::-1, :, :],
        "horizontal_flip": e[:, ::-1, :],
        "vertical_horizontal_flip": e[::-1, ::-1, :],
        "bgr_original": e[:, :, ::-1],
        "bgr_vertical_flip": e[::-1, :, ::-1],
        "bgr_horizontal_flip": e[:, ::-1, ::-1],
        "bgr_vertical_horizontal_flip": e[::-1, ::-1, ::-1],
    }

    scores = []
    for k, v in candidates.items():
        scores.append((k, mse(d, v)))

    scores = sorted(scores, key=lambda x: x[1])
    for k, s in scores:
        print(f"{k:32s} MSE = {s:.2f}")

    print("best:", scores[0])
