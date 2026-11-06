"""PFP sets per matched slice: the padded per-PFP record the M4 set model generates.

For every neutrino's kept slice (highest tmatch_eff, as in production_table) the PFPs that have a track or a
shower fit are kept (PFPs with neither, trackScore == -5, carry no information; their number is recorded), sorted
by distance to the vertex and padded to N_MAX. Raw values are stored (NaN where a field does not exist); the
model-space transform lives in xsum.data.pfpset_tf.
The track end point is carried as bend_{x,y,z} = end - (start + len * dir), so that generated ends keep the
scattering offset of real tracks (straight-line ends alone lose about 60% of truth-contained CC1p0pi signal
containment near the boundaries, reports/cc1p_containment/diag_summary.txt).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_MAX = 12
# (name, source column, transform used later)  -- see pfpset_tf
FIELDS = [
    ("has_trk", None, "flag"), ("has_shw", None, "flag"), ("has_mcs", None, "flag"), ("is_primary", "parent_is_primary", "flag"),
    ("dist_to_vertex", "dist_to_vertex", "log1p"), ("trackScore", "trackScore", "unit"),
    ("start_x", None, "lin"), ("start_y", None, "lin"), ("start_z", None, "lin"),
    ("dir_x", "trk_dir_x", "lin"), ("dir_y", "trk_dir_y", "lin"), ("dir_z", "trk_dir_z", "lin"),
    ("trk_len", "trk_len", "log"), ("rangeP_mu", "trk_rangeP_mu", "log"), ("mcsP_mu", "trk_mcsP_mu", "log"),
    ("chi2_mu", "chi2_mu", "log1p"), ("chi2_pro", "chi2_pro", "log1p"), ("chi2_mu_new", "chi2_mu_new_I2", "log1p"), ("chi2_pro_new", "chi2_pro_new_I2", "log1p"),
    ("chi2_kaon", "chi2_kaon_I2", "log1p"), ("chi2_pion", "chi2_pion_I2", "log1p"), ("pida", "pida", "log1p"), ("chi2_ndof", "chi2_ndof", "log1p"),
    ("shw_energy", "shw_bestplane_energy", "log"), ("shw_len", "shw_len", "log"), ("shw_open_angle", "shw_open_angle", "lin"),
    ("shw_density", "shw_density", "log1p"), ("shw_conversion_gap", "shw_conversion_gap", "log1p"), ("shw_dEdx", "shw_bestplane_dEdx", "log"),
    # track end point as the offset from the straight line start + len * dir (multiple-scattering bend), cm
    ("bend_x", None, "asinh5"), ("bend_y", None, "asinh5"), ("bend_z", None, "asinh5"),
]
NAMES = [f[0] for f in FIELDS]


def kept_slices(slc: pd.DataFrame, mcnu_index: pd.MultiIndex):
    """(slice key -> neutrino key) for the kept slice of every matched neutrino, as production_table does."""
    nu_levels = list(mcnu_index.names); ent = nu_levels[:-1]
    key = pd.MultiIndex.from_arrays([slc.index.get_level_values(n) for n in ent] + [slc["tmatch_idx"].astype(int).values], names=nu_levels)
    ok = key.isin(mcnu_index)
    s = pd.DataFrame({"eff": slc["tmatch_eff"].to_numpy(), "slc_index": slc.index.get_level_values(-1).to_numpy()}, index=key)[ok]
    s = s.sort_values("eff", ascending=False); s = s[~s.index.duplicated(keep="first")]
    return s


def build_sets(mcnu: pd.DataFrame, slc: pd.DataFrame, pfp: pd.DataFrame, n_max: int = N_MAX):
    """Returns dict of arrays: key [n,4] int64 (neutrino index), X [n,n_max,D] float32 (NaN = absent),
    n_fit [n] (fitted PFPs before truncation), n_empty [n], truth_pdg [n,n_max] int32 (0 = pad)."""
    ks = kept_slices(slc, mcnu.index)
    if ks.empty:
        return None
    ent = list(mcnu.index.names)[:-1]
    # select the PFPs of the kept slices
    skey = pd.MultiIndex.from_arrays([ks.index.get_level_values(n) for n in ent] + [ks["slc_index"].to_numpy()])
    pkey = pfp.index.droplevel(-1)
    p = pfp[pkey.isin(skey)].copy()
    p_slc = p.index.droplevel(-1)
    nu_of_slice = pd.Series(np.arange(len(ks)), index=skey)           # slice -> row number
    row = nu_of_slice.reindex(p_slc).to_numpy()
    has_trk = p["trk_len"].notna().to_numpy(); has_shw = (p["shw_bestplane_energy"].to_numpy() > -4) & (p["shw_len"].to_numpy() > 0)
    fitted = has_trk | has_shw
    n_empty = np.bincount(row[~fitted], minlength=len(ks))
    p = p[fitted]; row = row[fitted]; has_trk = has_trk[fitted]; has_shw = has_shw[fitted]
    order = np.lexsort((p["dist_to_vertex"].to_numpy(), row)); p = p.iloc[order]; row = row[order]; has_trk = has_trk[order]; has_shw = has_shw[order]
    pos = np.zeros(len(row), dtype=int)
    if len(row):
        first = np.r_[True, row[1:] != row[:-1]]; idx = np.arange(len(row)); starts = np.maximum.accumulate(np.where(first, idx, 0)); pos = idx - starts
    n_fit = np.bincount(row, minlength=len(ks))
    keepm = pos < n_max
    D = len(FIELDS); X = np.full((len(ks), n_max, D), np.nan, dtype=np.float32); tp = np.zeros((len(ks), n_max), dtype=np.int32)
    r, c = row[keepm], pos[keepm]; q = p[keepm]; ht, hs = has_trk[keepm], has_shw[keepm]
    for j, (name, col, tf) in enumerate(FIELDS):
        if name == "has_trk": v = ht.astype(np.float32)
        elif name == "has_shw": v = hs.astype(np.float32)
        elif name == "has_mcs": v = (q["trk_mcsP_mu"].notna().to_numpy() & ht).astype(np.float32)
        elif name.startswith("bend_"):
            a = name[-1]; v = q[f"trk_end_{a}"].to_numpy() - (q[f"trk_start_{a}"].to_numpy() + q["trk_len"].to_numpy() * q[f"trk_dir_{a}"].to_numpy())
            v = np.where(ht, v, np.nan).astype(np.float32)
        elif name.startswith("start_"):
            a = name[-1]; v = np.where(ht, q[f"trk_start_{a}"].to_numpy(), q[f"shw_start_{a}"].to_numpy())
        else:
            v = q[col].to_numpy().astype(np.float32)
            if name.startswith("shw_"): v = np.where(hs, v, np.nan)
            elif name in ("trackScore", "is_primary", "dist_to_vertex"): pass
            else: v = np.where(ht, v, np.nan)
        X[r, c, j] = v
    tp[r, c] = q["truth_pdg"].to_numpy()
    key = np.stack([ks.index.get_level_values(i).to_numpy() for i in range(ks.index.nlevels)], 1).astype(np.int64)
    return {"key": key, "X": X, "n_fit": n_fit.astype(np.int16), "n_empty": n_empty.astype(np.int16), "truth_pdg": tp}
