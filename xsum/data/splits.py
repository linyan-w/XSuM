"""Train / validation / test splits by input file, never by row."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd


def split_by_level(index: pd.MultiIndex, level: str = "file_idx", fractions: Sequence[float] = (0.8, 0.1, 0.1),
                   seed: int = 20261016) -> Dict[str, np.ndarray]:
    """Assign whole groups (default: files) to train/val/test; returns boolean masks over `index`.

    The assignment is a seeded shuffle of the SORTED list of groups, so it depends on the whole list: two tables whose
    file lists differ by even one file get unrelated assignments (10% overlap of their test sets). Every table that
    must share the models' split (file_splits.parquet of a dataset, the test-file loader) has to be derived from the
    models' training table with `file_split_table`, never from another file list."""
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


def file_split_table(model_features: str, all_features: str | None = None, seed: int = 0) -> pd.DataFrame:
    """The (chunk, __ntuple, split) table of a dataset, from the file list of the table the generators were trained on
    (`model_features`: its features.parquet; seed 0 is the training seed of run_m2 / run_m3 / run_m4). Files present in
    `all_features` (the full neutrino table) but absent from the model table (no matched slice) were never trained on
    and are labelled test."""
    def fids(path):
        idx = pd.read_parquet(path, columns=[]).index
        return np.unique(idx.get_level_values("chunk").to_numpy().astype(np.int64) * 100000 + idx.get_level_values("__ntuple").to_numpy().astype(np.int64))
    f = fids(model_features)
    sp = split_by_level(pd.MultiIndex.from_arrays([f], names=["file"]), level="file", seed=seed)
    lab = np.where(sp["train"], "train", np.where(sp["val"], "val", "test"))
    if all_features is not None:
        extra = np.setdiff1d(fids(all_features), f); f = np.concatenate([f, extra]); lab = np.concatenate([lab, np.full(len(extra), "test")])
    return pd.DataFrame({"chunk": (f // 100000).astype(np.int64), "__ntuple": (f % 100000).astype(np.int64), "split": lab}).sort_values(["chunk", "__ntuple"]).reset_index(drop=True)
