"""Seeding for reproducible runs (TRD: "fixed random seeds")."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and (if installed) PyTorch.

    `deterministic=True` also forces deterministic CUDA kernels. That makes GPU runs
    bit-reproducible but slower, so it is off for training (seed-to-seed variance is
    reported instead) and on for evaluation, where it is cheap.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # local CPU environment has no torch
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def rng(seed: int) -> random.Random:
    """An isolated RNG, so e.g. candidate shuffling doesn't consume the global stream."""
    return random.Random(seed)
