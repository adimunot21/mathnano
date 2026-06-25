"""Reproducibility: a single place to seed every RNG we touch.

WHY: ML results must be reproducible (global project rule). Seeding Python, NumPy and
PyTorch (CPU + CUDA) from one call means every script can `set_seed(CONFIG.seed)` at
startup and get deterministic data shuffles / init. nanochat seeds its own training
internally; this util is for OUR scripts (data prep, eval, Track B).
"""
from __future__ import annotations

import os
import random


def set_seed(seed: int = 1337, *, deterministic_torch: bool = False) -> int:
    """Seed Python, NumPy and PyTorch. Returns the seed for logging alongside outputs.

    deterministic_torch=True trades speed for bitwise determinism (cuDNN deterministic
    algorithms). Leave False for training (the throughput hit is not worth it); set True
    for small eval/debug runs where you want identical results every time.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


if __name__ == "__main__":
    # Sanity check: two calls with the same seed produce the same stream.
    import numpy as np

    set_seed(42)
    a = np.random.rand(3)
    set_seed(42)
    b = np.random.rand(3)
    assert np.allclose(a, b), "set_seed is not deterministic!"
    print(f"set_seed OK — reproducible stream: {a}")
