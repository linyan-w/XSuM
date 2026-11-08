#!/usr/bin/env python3
"""Which M2 slice-block outputs drive the CC1p0pi background excess at 2prong-contained?
Joins, per test neutrino of jobs 0-199, the real slice table (best-matched slice), the M2-generated slice (draw 0) and
the containment diagnostic of the full chain (reports/cc1p_containment/<diag dir>). For background neutrinos it compares
the real and the XSuM chain variable by variable: the fraction of neutrinos in each bin of a slice variable, and the
containment probability given the bin. If the conditionals agree and the fractions differ, the excess comes from M2
populating that bin too often; the 'excess explained' column is sum_bins (n_XSuM - n_real) * P_real(contained | bin).
usage: diag_m2_purity.py <diag dir> [generated slc parquet]"""
import glob
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
diag = sys.argv[1]; gen = sys.argv[2] if len(sys.argv) > 2 else "data/generated/slc_official_m3_all.parquet"
TAB = "data/prod_v10_14_02_03_official/table_m3/targets.parquet"
V = ["vtx_x", "vtx_y", "vtx_z", "nu_score", "n_pfp", "n_prong", "n_trk", "n_proton", "n_shower", "is_muon_contained", "is_cpi_contained",
     "muon_end_x", "muon_end_y", "muon_end_z", "cpi_end_x", "cpi_end_y", "cpi_end_z", "long_trk_len", "short_trk_len", "long_dirz"]
K = ["chunk", "__ntuple", "entry", "rec.mc.nu..index"]

R = pd.read_parquet(TAB, columns=V + ["matched"]); R = R[(R.matched == 1) & (R.index.get_level_values("chunk") < 200)].drop(columns="matched")
G = pd.read_parquet(gen, columns=V); G = G[(G.index.get_level_values("chunk") < 200) & (G.index.get_level_values("draw") == 0)].droplevel("draw")
D = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{diag}/g*.parquet")]).rename(columns={"job": "chunk", "nu": "rec.mc.nu..index"}).set_index(K)
for c in D.columns:
    if c[:2] in ("r_", "s_") and D[c].dtype == object: D[c] = D[c].fillna(False).astype(bool)
D = D[~D.signal]
fv = lambda x, y, z: (np.abs(x) < 190) & (np.abs(y) < 190) & (z > 10) & (z < 450)


def derive(T):
    T = T.copy()
    T["muon_end_fv"] = np.where(np.isfinite(T.muon_end_x), fv(T.muon_end_x, T.muon_end_y, T.muon_end_z), np.nan)
    T["cpi_end_fv"] = np.where(np.isfinite(T.cpi_end_x), fv(T.cpi_end_x, T.cpi_end_y, T.cpi_end_z), np.nan)
    T["topo"] = T[["n_prong", "n_trk", "n_proton", "n_shower"]].fillna(-1).astype(int).astype(str).agg("/".join, axis=1)
    return T


R, G = derive(R), derive(G)
r = D[["cat", "r_two", "r_contained"]].join(R.add_prefix("v_"), how="inner"); r = r[r.r_two]
s = D[["cat", "s_two", "s_contained"]].join(G.add_prefix("v_"), how="inner"); s = s[s.s_two]
print(f"background neutrinos at 2prong: real {len(r)} (contained {int(r.r_contained.sum())}), XSuM {len(s)} (contained {int(s.s_contained.sum())}); "
      f"excess contained {int(s.s_contained.sum() - r.r_contained.sum())}\n")


def table(var, bins=None, top=None):
    a, b = r[f"v_{var}"], s[f"v_{var}"]
    if bins is not None: a, b = pd.cut(a, bins).astype(str), pd.cut(b, bins).astype(str)
    else: a, b = a.astype(str), b.astype(str)
    keys = pd.concat([a, b]).value_counts().index[:top] if top else sorted(set(a) | set(b))
    print(f"{var}:   bin  |  real n  P(cont)  |  XSuM n  P(cont)")
    expl = 0.0
    for k in keys:
        ma, mb = a == k, b == k; na, nb = int(ma.sum()), int(mb.sum())
        pa = r.r_contained[ma].mean() if na else np.nan; pb = s.s_contained[mb].mean() if nb else np.nan
        if na: expl += (nb - na) * pa
        print(f"   {k:>18s} | {na:7d} {pa:7.3f} | {nb:7d} {pb:7.3f}")
    print(f"   excess explained by the bin populations: {expl:+.0f}\n")


table("is_muon_contained"); table("is_cpi_contained"); table("muon_end_fv"); table("cpi_end_fv")
table("topo", top=8)
table("long_trk_len", bins=[0, 25, 50, 100, 150, 200, 300, 1000])
table("long_dirz", bins=[-1, -0.5, 0, 0.5, 0.8, 0.95, 1])
table("vtx_z", bins=[0, 50, 100, 200, 300, 400, 450, 510])
for cat in sorted(r.cat.unique()):
    print(f"  {cat:28s} real 2prong {int((r.cat == cat).sum()):5d} cont {int(r.r_contained[r.cat == cat].sum()):5d} | XSuM {int((s.cat == cat).sum()):5d} cont {int(s.s_contained[s.cat == cat].sum()):5d}")

# wall consistency of the slice block itself: an exiting longest track has length = path from the vertex to the TPC wall
# along its direction, and an exiting cohpi muon ends on the wall
print("\nwall consistency of the slice block (all matched slices of jobs 0-199)")
W = ["vtx_x", "vtx_y", "vtx_z", "long_dirx", "long_diry", "long_dirz", "long_trk_len", "muon_end_x", "muon_end_y", "muon_end_z"]
R2 = pd.read_parquet(TAB, columns=W + ["matched"]); R2 = R2[(R2.matched == 1) & (R2.index.get_level_values("chunk") < 200)]
G2 = pd.read_parquet(gen, columns=W); G2 = G2[G2.index.get_level_values("chunk") < 200]
lo, hi = np.array([-201.3, -203.7, 0.0]), np.array([201.3, 203.7, 500.8]); bins = [0, 0.5, 0.8, 0.9, 0.95, 0.98, 1.02, 1.05, 1.2, 10]
for lab, T in (("real", R2), ("M2", G2)):
    T = T[np.isfinite(T.long_trk_len) & (T.long_trk_len > 0)]
    s0 = T[["vtx_x", "vtx_y", "vtx_z"]].to_numpy(); d = T[["long_dirx", "long_diry", "long_dirz"]].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        path = np.where(d > 0, (hi - s0) / d, np.where(d < 0, (lo - s0) / d, np.inf)).min(1)
    h = np.histogram(T.long_trk_len.to_numpy() / path, bins)[0] / len(T)
    print(f"  {lab:4s} long_trk_len / path to wall: " + " ".join(f"{a}-{b}:{v:.3f}" for a, b, v in zip(bins[:-1], bins[1:], h)))
    e = T[np.isfinite(T.muon_end_x)]; dw = np.minimum.reduce([201.3 - e.muon_end_x.abs(), 203.7 - e.muon_end_y.abs(), e.muon_end_z, 500.8 - e.muon_end_z])
    print(f"       cohpi muon end to wall: <-3 {np.mean(dw < -3):.3f}  +-3 {np.mean(np.abs(dw) < 3):.3f}  3-10 {np.mean((dw >= 3) & (dw < 10)):.3f}  "
          f"10-20 {np.mean((dw >= 10) & (dw < 20)):.3f}  >20 {np.mean(dw >= 20):.3f}")
