"""PFP sets per matched slice: the padded per-PFP record the M4 set model generates.

For every neutrino's kept slice (highest tmatch_eff, as in production_table) the PFPs that have a track or a
shower fit are kept (PFPs with neither, trackScore == -5, carry no information; their number is recorded), sorted
by distance to the vertex and padded to N_MAX. Raw values are stored (NaN where a field does not exist); the
model-space transform lives in xsum.data.pfpset_tf.
Layout v3: the track end point is a direct target, disp_{x,y,z} = end - start, and the path length is carried as
len_excess = trk_len - |disp| (cm of path beyond the chord), so trk_len = |disp| + len_excess and end = start + disp
exactly.  (v2 carried trk_len and bend = end - (start + len*dir); building end from four separately generated pieces
summed their errors and cost ~6% of truth-contained CC1p0pi signal containment, reports/cc1p_containment/.)
Group files written by older code (no `layout` entry) are v2 and are converted on load by `v2_to_v3`.
Layout v4 adds the TPC wall: real reco tracks that leave the active volume end in a sharp pile-up on its boundary
(x = +-201.3 cm and z = 0 / 500.5 cm within ~0.5 cm; y in a band ~199-204 cm), which a continuous flow smears over
the 1-30 cm inside it (exiting muons then pass the fiducial containment cut). Per axis a, wall_a flags a track
end in the wall band of that axis and wall_off_a is its outward distance from the reference plane WALL_REF; the
generator snaps the end onto the plane of the side it is on (`snap_walls`). v3 records are extended on load
(`v3_to_v4`).
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
    ("len_excess", None, "logp01"), ("rangeP_mu", "trk_rangeP_mu", "log"), ("mcsP_mu", "trk_mcsP_mu", "log"),
    ("chi2_mu", "chi2_mu", "log1p"), ("chi2_pro", "chi2_pro", "log1p"), ("chi2_mu_new", "chi2_mu_new_I2", "log1p"), ("chi2_pro_new", "chi2_pro_new_I2", "log1p"),
    ("chi2_kaon", "chi2_kaon_I2", "log1p"), ("chi2_pion", "chi2_pion_I2", "log1p"), ("pida", "pida", "log1p"), ("chi2_ndof", "chi2_ndof", "log1p"),
    ("shw_energy", "shw_bestplane_energy", "log"), ("shw_len", "shw_len", "log"), ("shw_open_angle", "shw_open_angle", "lin"),
    ("shw_density", "shw_density", "log1p"), ("shw_conversion_gap", "shw_conversion_gap", "log1p"), ("shw_dEdx", "shw_bestplane_dEdx", "log"),
    # track end point relative to its start, cm (the quantity containment cuts use)
    ("disp_x", None, "asinh10"), ("disp_y", None, "asinh10"), ("disp_z", None, "asinh10"),
    # track end on the TPC wall (layout v4): per-axis flag and outward offset from the reference plane, cm
    ("wall_x", None, "flag"), ("wall_y", None, "flag"), ("wall_z", None, "flag"),
    ("wall_off_x", None, "lin"), ("wall_off_y", None, "lin"), ("wall_off_z", None, "lin"),
]
NAMES = [f[0] for f in FIELDS]
LAYOUT = "v4"
N_V3 = NAMES.index("wall_x")          # v2 / v3 records have the first N_V3 fields
# reference planes (low side, high side) and the band (in cm beyond the plane, outward positive) counted as "on the wall",
# read off the real reco track-end distributions (pfpsets_v2 group 0)
WALL_REF = {"x": (-201.3, 201.3), "y": (-201.0, 201.0), "z": (0.0, 500.5)}
WALL_BAND = {"x": (-0.8, 1.7), "y": (-2.0, 5.0), "z": (-0.75, 3.0)}


def v2_to_v3(X):
    """In place: v2 record [.., D] (trk_len, bend) -> v3 (len_excess, disp). Same shape, same field positions."""
    il = NAMES.index("len_excess"); idisp = [NAMES.index(f"disp_{a}") for a in "xyz"]; idir = [NAMES.index(f"dir_{a}") for a in "xyz"]
    L = X[..., il].astype(np.float64)
    d = np.stack([L * X[..., idir[k]] + X[..., idisp[k]] for k in range(3)], -1)        # len*dir + bend = end - start
    for k in range(3): X[..., idisp[k]] = d[..., k]
    X[..., il] = np.maximum(L - np.sqrt((d ** 2).sum(-1)), 0.0)
    return X


def wall_fields(X):
    """From a record with start and disp: [..., 6] float32 (wall_x/y/z flags, wall_off_x/y/z; NaN where no track /
    not on the wall)."""
    ht = X[..., NAMES.index("has_trk")] > 0.5; out = np.full(X.shape[:-1] + (6,), np.nan, np.float32)
    for k, a in enumerate("xyz"):
        e = X[..., NAMES.index(f"start_{a}")] + X[..., NAMES.index(f"disp_{a}")]
        lo, hi = WALL_REF[a]; mid = 0.5 * (lo + hi)
        off = np.where(e > mid, e - hi, lo - e)                      # outward distance from the plane of the nearer side
        on = ht & (off >= WALL_BAND[a][0]) & (off <= WALL_BAND[a][1])
        out[..., k] = np.where(ht, on, np.nan); out[..., 3 + k] = np.where(on, off, np.nan)
    return out


def v3_to_v4(X3):
    """v3 record [..., N_V3] -> v4 [..., D] (new array)."""
    X = np.empty(X3.shape[:-1] + (len(FIELDS),), np.float32); X[..., :N_V3] = X3; X[..., N_V3:] = wall_fields(X3)
    return X


def snap_walls(X):
    """In place on a generated raw record: for every track end flagged on the wall of axis a, put the end coordinate at
    the reference plane of the side it is on plus the generated outward offset (disp_a adjusted; start unchanged)."""
    for a in "xyz":
        on = X[..., NAMES.index(f"wall_{a}")] > 0.5
        if not on.any(): continue
        js, jd = NAMES.index(f"start_{a}"), NAMES.index(f"disp_{a}")
        e = X[..., js] + X[..., jd]; lo, hi = WALL_REF[a]; off = np.nan_to_num(X[..., NAMES.index(f"wall_off_{a}")], nan=0.0)
        snapped = np.where(e > 0.5 * (lo + hi), hi + off, lo - off)
        X[..., jd] = np.where(on & np.isfinite(e), snapped - X[..., js], X[..., jd])
    return X


def track_len(X):
    """trk_len from a v3 record (NaN where there is no track)."""
    d = np.stack([X[..., NAMES.index(f"disp_{a}")] for a in "xyz"], -1)
    return np.sqrt((d ** 2).sum(-1)) + X[..., NAMES.index("len_excess")]


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
    for j, (name, col, tf) in enumerate(FIELDS[:N_V3]):
        if name == "has_trk": v = ht.astype(np.float32)
        elif name == "has_shw": v = hs.astype(np.float32)
        elif name == "has_mcs": v = (q["trk_mcsP_mu"].notna().to_numpy() & ht).astype(np.float32)
        elif name.startswith("disp_"):
            a = name[-1]; v = np.where(ht, q[f"trk_end_{a}"].to_numpy() - q[f"trk_start_{a}"].to_numpy(), np.nan).astype(np.float32)
        elif name == "len_excess":
            d = np.sqrt(sum((q[f"trk_end_{a}"].to_numpy() - q[f"trk_start_{a}"].to_numpy()) ** 2 for a in "xyz"))
            v = np.where(ht, np.maximum(q["trk_len"].to_numpy() - d, 0.0), np.nan).astype(np.float32)
        elif name.startswith("start_"):
            a = name[-1]; v = np.where(ht, q[f"trk_start_{a}"].to_numpy(), q[f"shw_start_{a}"].to_numpy())
        else:
            v = q[col].to_numpy().astype(np.float32)
            if name.startswith("shw_"): v = np.where(hs, v, np.nan)
            elif name in ("trackScore", "is_primary", "dist_to_vertex"): pass
            else: v = np.where(ht, v, np.nan)
        X[r, c, j] = v
    tp[r, c] = q["truth_pdg"].to_numpy()
    X[..., N_V3:] = wall_fields(X)
    key = np.stack([ks.index.get_level_values(i).to_numpy() for i in range(ks.index.nlevels)], 1).astype(np.int64)
    return {"layout": np.array(LAYOUT), "key": key, "X": X, "n_fit": n_fit.astype(np.int16), "n_empty": n_empty.astype(np.int16), "truth_pdg": tp}
