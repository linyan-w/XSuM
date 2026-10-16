"""Train / validation / test splits by input file, never by row."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd


def split_by_level(index: pd.MultiIndex, level: str = "file_idx", fractions: Sequence[float] = (0.8, 0.1, 0.1),
                   seed: int = 20261016) -> Dict[str, np.ndarray]:
    """Assign whole groups (default: files) to train/val/test; returns boolean masks over `index`."""
    assert abs(sum(fractions) - 1.0) < 1e-9 and len(fractions) == 3
    groups = np.asarray(sorted(set(index.get_level_values(level))))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    parts = {"train": groups[:n_train], "val": groups[n_train:n_train + n_val], "test": groups[n_train + n_val:]}
    values = index.get_level_values(level)
    return {k: np.isin(values, v) for k, v in parts.items()}
