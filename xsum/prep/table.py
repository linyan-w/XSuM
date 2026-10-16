"""Build the per-neutrino training table: truth, the Tier 0 `selected` flag and the Tier 1 reco block.

One row per true neutrino in `mcnu`. `selected` is 1 when a slice in `evt` is truth-matched to it through
(file_idx, __ntuple, entry, tmatch_idx). Reco columns are NaN for unselected neutrinos.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from xsum.constants import ENTRY_INDEX, MCNU_INDEX, TRUTH_ALL

# reco columns that are copies of truth or bookkeeping, never targets
_NOT_RECO = set(TRUTH_ALL) | {"rec.mc.nu..index", "tmatch_idx", "tmatch_eff", "tmatch_purity",
                              "long_dirz_pass", "short_dirz_pass"}


def reco_columns(evt: pd.DataFrame) -> List[str]:
    return [c for c in evt.columns if c not in _NOT_RECO]


def match_key(evt: pd.DataFrame) -> pd.MultiIndex:
    """(file_idx, __ntuple, entry, tmatch_idx) of each slice, named like the mcnu index."""
    return pd.MultiIndex.from_arrays(
        [evt.index.get_level_values(n) for n in ENTRY_INDEX] + [evt["tmatch_idx"].astype(int).values],
        names=MCNU_INDEX,
    )


def build_table(mcnu: pd.DataFrame, evt: pd.DataFrame, reco_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Join reco onto truth. Raises if a neutrino has more than one matched slice."""
    reco_cols = reco_cols or reco_columns(evt)
    key = match_key(evt)
    matched = evt.loc[key.isin(mcnu.index), reco_cols].copy()
    matched.index = key[key.isin(mcnu.index)]
    if matched.index.duplicated().any():
        n = int(matched.index.duplicated().sum())
        raise ValueError(f"{n} neutrinos have more than one matched slice; decide a rule before building the table")
    table = mcnu[[c for c in TRUTH_ALL if c in mcnu.columns]].copy()
    table["selected"] = table.index.isin(matched.index).astype(np.int8)
    table = table.join(matched, how="left")
    return table


def summary(table: pd.DataFrame) -> dict:
    sel = table[table.selected == 1]
    out = {
        "n_truth": int(len(table)),
        "n_selected": int(table.selected.sum()),
        "efficiency": float(table.selected.mean()),
    }
    if {"range_p_mu", "true_p_mu"}.issubset(sel.columns):
        r = sel.range_p_mu / sel.true_p_mu
        out["range_p_mu_over_true_q16_50_84"] = [float(x) for x in r.quantile([0.16, 0.5, 0.84])]
    if {"range_p_cpi", "true_p_pi"}.issubset(sel.columns):
        r = sel.range_p_cpi / sel.true_p_pi
        out["range_p_cpi_over_true_q16_50_84"] = [float(x) for x in r.quantile([0.16, 0.5, 0.84])]
    return out
