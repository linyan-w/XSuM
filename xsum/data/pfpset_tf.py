"""PFP-set records (xsum.prep.pfpset FIELDS, raw) <-> model space.

Flags are dequantised to +-[0.2, 1]; `trackScore` (in [0,1] for fitted PFPs) goes through artanh(2x-1); positive
quantities through log / log1p; everything is robustly standardised. Fields that do not exist for a token (NaN:
e.g. MCS momentum without a fit) are replaced by standard-normal noise so that the flow learns "flag low => rest is
noise"; the decoder restores NaN from the flags. Padding tokens are masked out of the loss entirely.
"""
from __future__ import annotations

import numpy as np

from ..prep.pfpset import FIELDS, NAMES

FLAG = [i for i, f in enumerate(FIELDS) if f[2] == "flag"]
EDGE = 1 - 1e-5
# which flag governs which fields (NaN pattern); None = always present for a fitted PFP
GOVERNOR = {"mcsP_mu": "has_mcs", "wall_off_x": "wall_x", "wall_off_y": "wall_y", "wall_off_z": "wall_z"}
GOVERNOR.update({n: "has_shw" for n in NAMES if n.startswith("shw_")})
GOVERNOR.update({n: "has_trk" for n in ("disp_x", "disp_y", "disp_z", "dir_x", "dir_y", "dir_z", "len_excess", "rangeP_mu", "chi2_mu", "chi2_pro", "chi2_mu_new", "chi2_pro_new", "chi2_kaon", "chi2_pion", "pida", "chi2_ndof")})


def _dequant(flag, rng):
    u = 0.2 + 0.8 * rng.random(flag.shape)
    return np.where(flag > 0.5, u, -u)


def fwd(x, tf):
    x = np.asarray(x, dtype=np.float64)
    if tf == "log": return np.log(np.where(x > 0, x, np.nan))
    if tf == "log1p": return np.log1p(np.where(x >= 0, x, np.nan))
    if tf == "unit": return np.arctanh(np.clip(2 * x - 1, -EDGE, EDGE))
    if tf == "logp01": return np.log(np.where(x >= 0, x, np.nan) + 0.1)
    if tf.startswith("asinh"): return np.arcsinh(x / float(tf[5:]))
    return x


def inv(y, tf):
    if tf == "log": return np.exp(y)
    if tf == "log1p": return np.maximum(np.expm1(y), 0.0)
    if tf == "unit": return 0.5 * (np.tanh(y) + 1)
    if tf == "logp01": return np.maximum(np.exp(y) - 0.1, 0.0)
    if tf.startswith("asinh"): return float(tf[5:]) * np.sinh(y)
    return y


class PfpTransform:
    def __init__(self):
        self.loc = np.zeros(len(FIELDS)); self.scale = np.ones(len(FIELDS))

    def _unscaled(self, X, rng):
        """X [n, N, D] raw -> same shape, transformed, NaN where absent; flags dequantised."""
        U = np.empty(X.shape, dtype=np.float64)
        for j, (name, _, tf) in enumerate(FIELDS):
            U[..., j] = _dequant(np.nan_to_num(X[..., j], nan=0.0), rng) if tf == "flag" else fwd(X[..., j], tf)
        return U

    def fit(self, X, seed=0, max_rows=500_000):
        """Robust location/scale per field from (a subsample of) the valid tokens."""
        rng = np.random.default_rng(seed)
        if len(X) > max_rows: X = X[np.sort(rng.choice(len(X), max_rows, replace=False))]
        valid = ~np.isnan(X[..., 0])
        U = self._unscaled(X, rng)
        for j, (name, _, tf) in enumerate(FIELDS):
            if tf == "flag": continue
            v = U[..., j][valid]; v = v[np.isfinite(v)]
            if len(v) == 0: continue
            self.loc[j] = np.median(v); self.scale[j] = 1.4826 * np.median(np.abs(v - self.loc[j])) + 1e-6
        return self

    def forward(self, X, rng, chunk=200_000, inplace=False):
        """-> Z [n, N, D] float32 (noise where absent or padded), valid [n, N] bool. Chunked: float64 temporaries
        exist only for `chunk` slices at a time. inplace=True overwrites X (float32) with Z to save memory."""
        valid = ~np.isnan(X[..., 0])
        Z = X if (inplace and X.dtype == np.float32) else np.empty(X.shape, dtype=np.float32)
        for i in range(0, len(X), chunk):
            U = (self._unscaled(X[i:i + chunk], rng) - self.loc) / self.scale
            z = np.where(np.isfinite(U), U, rng.standard_normal(U.shape)); del U
            v = valid[i:i + chunk]; z[~v] = rng.standard_normal((int((~v).sum()), z.shape[-1]))
            Z[i:i + chunk] = np.clip(z, -15, 15); del z   # wide enough for the vertex-distance tail (robust MAD is tiny)
        return Z, valid

    def inverse(self, Z, valid):
        """Model space -> raw record [n, N, D] with NaN where the governing flag is off or the token is padding."""
        U = Z.astype(np.float64) * self.scale + self.loc
        X = np.empty(U.shape); flags = {}
        for j, (name, _, tf) in enumerate(FIELDS):
            if tf == "flag": X[..., j] = (U[..., j] > 0).astype(float); flags[name] = X[..., j] > 0.5
        for j, (name, _, tf) in enumerate(FIELDS):
            if tf == "flag": continue
            v = inv(U[..., j], tf); g = GOVERNOR.get(name)
            if g is not None: v = np.where(flags[g], v, np.nan)
            X[..., j] = v
        # unit direction
        n = np.sqrt(np.nansum(np.stack([X[..., NAMES.index(f"dir_{a}")] ** 2 for a in "xyz"]), 0))
        for a in "xyz":
            j = NAMES.index(f"dir_{a}"); X[..., j] = np.where(n > 0, X[..., j] / np.where(n > 0, n, 1), X[..., j])
        X[~valid] = np.nan
        return X
