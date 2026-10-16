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


# ------------------------------------------------------------------------------------- production format
SLC_RECO_COLS = ["vtx_x", "vtx_y", "vtx_z", "nu_score", "bcfm_score", "total_pe", "n_pfp", "n_prong", "n_trk", "n_proton", "n_shower",
                 "tmatch_eff", "tmatch_purity",
                 "long_dirx", "long_diry", "long_dirz", "long_trk_len", "long_trk_pid", "long_trk_chi2_mu", "long_trk_chi2_pro",
                 "long_trk_chi2_mu_new", "long_trk_chi2_pro_new",
                 "short_dirx", "short_diry", "short_dirz", "short_trk_len", "short_trk_pid", "short_trk_chi2_mu", "short_trk_chi2_pro",
                 "short_trk_chi2_mu_new", "short_trk_chi2_pro_new",
                 "reco_t", "opening_angle", "is_muon_contained", "is_cpi_contained", "range_p_mu", "range_p_cpi", "range_p_cpi_pion",
                 "muon_start_x", "muon_start_y", "muon_start_z", "muon_end_x", "muon_end_y", "muon_end_z",
                 "cpi_start_x", "cpi_start_y", "cpi_start_z", "cpi_end_x", "cpi_end_y", "cpi_end_z",
                 "sum_integ_1cm", "sum_integ_2cm", "sum_integ_3cm", "sum_integ_4cm", "sphere_3_over_4m3",
                 "pass_fv_cosmic", "pass_pe_bcfm", "pass_2trk", "pass_dir", "pass_chi2", "pass_sphere", "pass_cathode", "pass_cohpi", "in_sr"]


def production_table(mcnu: pd.DataFrame, slc: pd.DataFrame) -> pd.DataFrame:
    """One row per true neutrino of the XSuM production: Tier 0 flags and the reco block of the matched slice.

    Index levels (chunk, __ntuple, entry, rec.mc.nu..index). A neutrino can be truth-matched by several
    slices; the one with the highest tmatch_eff is kept (`n_matched_slices` records how many there were).
    `matched` = at least one slice matched; `selected` = the kept slice passes the cohpi selection.
    """
    nu_levels = list(mcnu.index.names)
    entry_levels = nu_levels[:-1]
    key = pd.MultiIndex.from_arrays([slc.index.get_level_values(n) for n in entry_levels] + [slc["tmatch_idx"].astype(int).values], names=nu_levels)
    s = slc.copy()
    s.index = key
    s = s[s.index.isin(mcnu.index)]
    n_matched = s.groupby(level=nu_levels).size()
    s = s.sort_values("tmatch_eff", ascending=False)
    s = s[~s.index.duplicated(keep="first")]
    cols = [c for c in SLC_RECO_COLS if c in s.columns]
    table = pd.DataFrame(index=mcnu.index)
    table["matched"] = table.index.isin(s.index).astype(np.int8)
    table["n_matched_slices"] = n_matched.reindex(table.index).fillna(0).astype(np.int8)
    table = table.join(s[cols], how="left")
    for c in cols:
        if table[c].dtype == object or table[c].dtype == bool:
            table[c] = table[c].astype(float)
    table["selected"] = table["pass_cohpi"].fillna(0).astype(np.int8)
    return table
