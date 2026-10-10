"""Exercise the real joint A/B path with every sample forced to depth 24."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train
import torch

original_losses = train.joint_sample_losses


def full_depth_losses(*args, **kwargs):
    a1, a2, selected, p, entropy = original_losses(*args, **kwargs)
    return a1, a2, torch.full_like(selected, args[0].shape[1]), p, entropy


train.joint_sample_losses = full_depth_losses
if __name__ == "__main__":
    train.main()
