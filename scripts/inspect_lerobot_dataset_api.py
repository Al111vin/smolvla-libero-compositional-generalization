import inspect

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    print("=" * 80)
    print("LeRobotDataset class:")
    print(LeRobotDataset)

    print("=" * 80)
    print("LeRobotDataset __init__ signature:")
    print(inspect.signature(LeRobotDataset.__init__))

    print("=" * 80)
    print("LeRobotDataset methods:")
    methods = [
        name
        for name in dir(LeRobotDataset)
        if not name.startswith("_")
    ]
    for name in methods:
        attr = getattr(LeRobotDataset, name)
        if callable(attr):
            try:
                sig = inspect.signature(attr)
                print(f"{name}{sig}")
            except Exception:
                print(name)

    print("=" * 80)
    print("LeRobotDataset API inspection passed.")


if __name__ == "__main__":
    main()