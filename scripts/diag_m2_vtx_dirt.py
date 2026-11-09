#!/usr/bin/env python3
"""The reco vertex of slices from neutrinos OUTSIDE the active volume (dirt), truth CC + pion, n_pfp = 3, nu_score > 0.45:
where the real reconstruction puts it relative to the true muon line, and how that ties to the longest track exiting
(diag_m2_vtx_dir.py: the out-of-AV zone carries the exit <-> vertex-in-FV link, real +0.108, generated +0.02..0.06).
Muon direction from the primary set (truth, diagnostic only).
usage: diag_m2_vtx_dirt.py <M2 report dir> <table dir> <primsets dir>"""
import sys

import numpy as np
import pandas as pd

from xsum.train.m3 import pack

D, TAB, PS = sys.argv[1], sys.argv[2], sys.argv[3]
cols = ["vtx_x", "vtx_y", "vtx_z", "long_dirx", "long_diry", "long_dirz", "long_trk_len", "n_pfp", "nu_score"]
R = pd.read_parquet(f"{D}/real_test.parquet", columns=cols); G = pd.read_parquet(f"{D}/generated_test.parquet", columns=cols)
if "draw" in G.index.names: G = G.droplevel("draw")
G = G.loc[R.index]
F = pd.read_parquet(f"{TAB}/features.parquet", columns=["vtx_x", "vtx_y", "vtx_z", "n_mu", "n_pi"]).loc[R.index]
LO, HI = np.array([-190., -190., 10.]), np.array([190., 190., 450.]); AVLO, AVHI = np.array([-200., -200., 0.]), np.array([200., 200., 500.])
TV = F[["vtx_x", "vtx_y", "vtx_z"]].to_numpy(float); out_av = ~((TV > AVLO) & (TV < AVHI)).all(1)
# leading true muon direction from the primary set
keys = np.load(f"{PS}/mmap/keys.npy"); cls = np.load(f"{PS}/mmap/cls.npy", mmap_mode="r"); X = np.load(f"{PS}/mmap/X.npy", mmap_mode="r")
k = pack(R.index.get_level_values("chunk"), R.index.get_level_values("__ntuple"), R.index.get_level_values("entry"), R.index.get_level_values("rec.mc.nu..index"))
pos = np.searchsorted(keys, k); ok = (pos < len(keys)) & (keys[np.clip(pos, 0, len(keys) - 1)] == k)
mu_dir = np.full((len(R), 3), np.nan)
for i in np.flatnonzero(ok):
    c = cls[pos[i]]; j = np.flatnonzero(c == 0)
    if len(j):
        p = X[pos[i], j, :3].astype(float); jj = j[np.argmax((p ** 2).sum(1))]; q = X[pos[i], jj, :3].astype(float); mu_dir[i] = q / max(np.linalg.norm(q), 1e-9)
has_mu = ~np.isnan(mu_dir[:, 0])
base = ((F.n_mu >= 1) & (F.n_pi >= 1)).to_numpy() & out_av & has_mu & R.long_trk_len.notna().to_numpy()


def entry_point(tv, u):
    """first point of the line tv + t u (t > 0) inside the AV, nan if it misses"""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (AVLO - tv) / u; t2 = (AVHI - tv) / u
    tmin = np.maximum.reduce([np.minimum(t1, t2)[:, i] for i in range(3)]); tmax = np.minimum.reduce([np.maximum(t1, t2)[:, i] for i in range(3)])
    hit = (tmax > tmin) & (tmin > 0)
    e = tv + tmin[:, None] * u; e[~hit] = np.nan
    return e, tmin, tmax


E, tin, tout = entry_point(TV, mu_dir)
hits = ~np.isnan(E[:, 0])
print(f"out-of-AV CC + pion slices with a longest track: {base.sum()}; the true muon line enters the AV for {hits[base].mean():.3f} of them; chord length median {np.median((tout - tin)[base & hits]):.0f} cm")

for lab, S in (("real", R), ("gen", G)):
    V = S[["vtx_x", "vtx_y", "vtx_z"]].to_numpy(float); u = S[["long_dirx", "long_diry", "long_dirz"]].to_numpy(float)
    u = u / np.clip(np.linalg.norm(u, axis=1, keepdims=True), 1e-6, None)
    end = V + S.long_trk_len.to_numpy(float)[:, None] * u; exit_ = ~((end > LO) & (end < HI)).all(1)
    sel = base & hits & (S.nu_score > 0.45).to_numpy() & (S.n_pfp == 3).to_numpy()
    fvp = ((V > LO) & (V < HI)).all(1)
    # reco vertex relative to the true muon line: along (from the entry point, in units of the chord) and perpendicular distance
    dv = V - E; along = (dv * mu_dir).sum(1); perp = np.linalg.norm(dv - along[:, None] * mu_dir, axis=1); frac = along / (tout - tin)
    cosmu = np.abs((u * mu_dir).sum(1))
    # the straight-line end of the longest track against the AV wall along the reco direction
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (AVLO - V) / u; t2 = (AVHI - V) / u
    path = np.minimum.reduce([np.maximum(t1, t2)[:, i] for i in range(3)]); ratio = S.long_trk_len.to_numpy(float) / path
    print(f"\n{lab}: n_pfp = 3, nu_score > 0.45, out-of-AV, muon line enters the AV: n {sel.sum()}  P(exit) {exit_[sel].mean():.3f}  P(reco vertex in FV) {fvp[sel].mean():.3f}  P(FV | exit) {fvp[sel & exit_].mean():.3f}  P(FV | cont) {fvp[sel & ~exit_].mean():.3f}  P(exit | FV) {exit_[sel & fvp].mean():.3f}  P(exit | not FV) {exit_[sel & ~fvp].mean():.3f}")
    for name, m in (("exit", sel & exit_), ("cont", sel & ~exit_), ("in FV", sel & fvp), ("not FV", sel & ~fvp)):
        print(f"   {name:7s} n {m.sum():5d} | perp to the muon line: median {np.median(perp[m]):6.1f} cm  P(< 5) {(perp[m] < 5).mean():.3f} | along from the entry point / chord: q25 {np.quantile(frac[m], .25):+.2f} med {np.median(frac[m]):+.2f} q75 {np.quantile(frac[m], .75):+.2f}  P(< 0.05) {(frac[m] < .05).mean():.3f}"
              f" | |cos(reco long dir, true mu)| > 0.95: {(cosmu[m] > .95).mean():.3f} | len / path to wall in 0.95-1.05: {((ratio[m] > .95) & (ratio[m] < 1.05)).mean():.3f}  P(< 0.8) {(ratio[m] < .8).mean():.3f}")
    # the joint that makes the link: vertex deep inside (frac > 0.05, perp < 5) and the track collinear with the muon -> exit
    deep = sel & (perp < 5) & (frac > .05) & (frac < .95); print(f"   on the muon line, inside the chord: n {deep.sum():5d}  P(FV) {fvp[deep].mean():.3f}  P(exit) {exit_[deep].mean():.3f}  P(exit | collinear) {exit_[deep & (cosmu > .95)].mean():.3f}  P(collinear) {(cosmu[deep] > .95).mean():.3f}  len/path in 0.95-1.05 {((ratio[deep] > .95) & (ratio[deep] < 1.05)).mean():.3f}")
    atwall = sel & (perp < 5) & (frac <= .05); print(f"   on the muon line, at the entry wall:    n {atwall.sum():5d}  P(FV) {fvp[atwall].mean():.3f}  P(exit) {exit_[atwall].mean():.3f}  P(collinear) {(cosmu[atwall] > .95).mean():.3f}")
    off = sel & (perp >= 5); print(f"   off the muon line (> 5 cm):             n {off.sum():5d}  P(FV) {fvp[off].mean():.3f}  P(exit) {exit_[off].mean():.3f}")
