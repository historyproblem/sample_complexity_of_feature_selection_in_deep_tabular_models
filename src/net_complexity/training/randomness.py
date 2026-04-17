from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def set_random_seed(seed: Any) -> int | None:
    if seed is None:
        return None

    resolved_seed = int(seed)
    random.seed(resolved_seed)
    np.random.seed(resolved_seed)
    torch.manual_seed(resolved_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(resolved_seed)
        torch.cuda.manual_seed_all(resolved_seed)
    return resolved_seed
