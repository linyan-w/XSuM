"""Particle sets per true neutrino: every post-FSI GENIE primary as a token (the full truth input of M3).

Kept: all primaries except outgoing neutrinos, nuclear remnants (|pdg| > 1e9) and the GENIE binding-energy
pseudo-particle (2000000101). Per token: species class, (px, py, pz) in detector coordinates [GeV], kinetic energy
[GeV]. Tokens sorted by kinetic energy (descending), padded / truncated to N_MAX (the dropped tail is <0.5% of
particles, mostly low-energy neutrons; their number and summed KE are recorded).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_MAX = 32
CLASSES = {"mu": (13,), "e": (11,), "p": (2212,), "n": (2112,), "pi": (211,), "pi0": (111,), "g": (22,),
           "k": (321,), "k0": (311, 310, 130), "lam": (3122,), "sig": (3112, 3222, 3212), "other": ()}
CLASS_NAMES = list(CLASSES)
MASS = {13: 0.1056584, 11: 0.000511, 2212: 0.938272, 2112: 0.9395654, 211: 0.13957039, 111: 0.1349768, 22: 0.0,
        321: 0.493677, 311: 0.497611, 310: 0.497611, 130: 0.497611, 3122: 1.115683, 3112: 1.197449, 3222: 1.18937, 3212: 1.192642}
FIELDS = ["px", "py", "pz", "ke"]


def build_primsets(mcnu: pd.DataFrame, mcprim: pd.DataFrame, n_max: int = N_MAX):
    """-> dict: key [n,4] int64 (neutrino index, same order as mcnu), cls [n,N] int8 (-1 = pad),
    X [n,N,4] float32, n_prim [n], n_dropped [n], ke_dropped [n]."""
    apdg = mcprim["pdg"].abs().to_numpy()
    keep = ~(np.isin(apdg, (12, 14, 16)) | (apdg > 1_000_000_000) | (apdg == 2000000101))
    p = mcprim[keep]
    apdg = apdg[keep]
    cls = np.full(len(p), CLASS_NAMES.index("other"), dtype=np.int8)
    for i, (name, codes) in enumerate(CLASSES.items()):
        if codes: cls[np.isin(apdg, codes)] = i
    mass = np.array([MASS.get(int(a), np.nan) for a in apdg])
    e = p["genE"].to_numpy()
    ke = np.where(np.isfinite(mass), e - mass, e)                        # "other": total energy
    px, py, pz = (p[c].to_numpy() for c in ("genp_x", "genp_y", "genp_z"))
    # row of each primary in mcnu order
    nu_levels = list(mcnu.index.names)
    pkey = p.index.droplevel(-1)
    row = pd.Series(np.arange(len(mcnu)), index=mcnu.index).reindex(pkey).to_numpy()
    ok = ~np.isnan(row); row = row[ok].astype(np.int64)
    cls, ke, px, py, pz = cls[ok], ke[ok], px[ok], py[ok], pz[ok]
    order = np.lexsort((-ke, row)); row, cls, ke, px, py, pz = row[order], cls[order], ke[order], px[order], py[order], pz[order]
    idx = np.arange(len(row)); first = np.r_[True, row[1:] != row[:-1]] if len(row) else np.array([], bool)
    pos = idx - np.maximum.accumulate(np.where(first, idx, 0)) if len(row) else idx
    n = len(mcnu); n_prim = np.bincount(row, minlength=n)
    drop = pos >= n_max
    n_dropped = np.bincount(row[drop], minlength=n); ke_dropped = np.bincount(row[drop], weights=ke[drop], minlength=n)
    m = ~drop
    C = np.full((n, n_max), -1, dtype=np.int8); X = np.zeros((n, n_max, len(FIELDS)), dtype=np.float32)
    C[row[m], pos[m]] = cls[m]
    for j, v in enumerate((px, py, pz, ke)): X[row[m], pos[m], j] = v[m]
    key = np.stack([mcnu.index.get_level_values(l).to_numpy() for l in nu_levels], 1).astype(np.int64)
    return {"key": key, "cls": C, "X": X, "n_prim": n_prim.astype(np.int16), "n_dropped": n_dropped.astype(np.int16), "ke_dropped": ke_dropped.astype(np.float32)}
