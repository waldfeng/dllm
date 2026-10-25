import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

if not torch.cuda.is_available():
    raise RuntimeError("tests require CUDA: GPU-only policy for training and inference")

torch.set_default_device("cuda")
