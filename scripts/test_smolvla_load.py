import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("SmolVLAConfig import OK:", SmolVLAConfig)
    print("SmolVLAPolicy import OK:", SmolVLAPolicy)

    print("SmolVLA basic import test passed.")


if __name__ == "__main__":
    main()