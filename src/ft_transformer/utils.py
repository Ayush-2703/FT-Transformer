"""Reproducibility and device-selection helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fix every relevant RNG (Python, NumPy, PyTorch CPU/CUDA) for reproducibility.

    Also disables CuDNN auto-tuning and forces deterministic CuDNN kernels, matching
    the exact configuration used in the original research notebook (Cell 1).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA if available, else CPU.

    The reference experiments were run on a Colab T4 GPU; this repository's test
    suite and CI pipeline are designed to run identically (just slower) on CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
