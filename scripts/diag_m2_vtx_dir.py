#!/usr/bin/env python3
"""Direction of the reco - true vertex displacement, real vs generated, for truth CC + pion slices whose true vertex
is in the FV (follow-up of diag_m2_link.py section 8 once the displacement magnitude is right, M2v): the component
along the longest track, the component towards the nearest FV wall, and the FV pass rate against the true distance to
that wall, by reco exit.
usage: diag_m2_vtx_dir.py <M2 report dir> <table dir>"""
import sys

import numpy as np
import pandas as pd

D, TAB = sys.argv[1], sys.argv[2]
cols = ["vtx_x", "vtx_y", "vtx_z", "long_dirx", "long_diry", "long_dirz", "long_trk_len", "n_pfp", "nu_score"]
R = pd.read_parquet(f"{D}/real_test.parquet", columns=cols); G = pd.read_parquet(f"{D}/generated_test.parquet", columns=cols)
if "draw" in G.index.names: G = G.droplevel("draw")
G = G.loc[R.index]
F = pd.read_parquet(f"{TAB}/features.parquet", columns=["vtx_x", "vtx_y", "vtx_z", "n_mu", "n_pi"]).loc[R.index]
LO, HI = np.array([-190., -190., 10.]), np.array([190., 190., 450.])
TV = F[["vtx_x", "vtx_y", "vtx_z"]].to_numpy(float)
# FV in x is two-sided around the cathode: wall distance in x is min(190 - |x|); treat |x| as the coordinate
tx = np.abs(TV[:, 0]); dwall = np.stack([190 - tx, 190 - np.abs(TV[:, 1]), TV[:, 2] - 10, 450 - TV[:, 2]], 1)
base = ((F.n_mu >= 1) & (F.n_pi >= 1)).to_numpy() & (dwall.min(1) > 0) & R.long_trk_len.notna().to_numpy()


def fv(S):
    V = S[["vtx_x", "vtx_y", "vtx_z"]].to_numpy(float)
    return ((V > LO) & (V < HI)).all(1)


def rows(S, lab, n3only):
    V = S[["vtx_x", "vtx_y", "vtx_z"]].to_numpy(float); d = V - TV
    u = S[["long_dirx", "long_diry", "long_dirz"]].to_numpy(float); u = u / np.clip(np.linalg.norm(u, axis=1, keepdims=True), 1e-6, None)
    end = V + S.long_trk_len.to_numpy(float)[:, None] * u
    exit_ = ~((end > LO) & (end < HI)).all(1)
    sel = base & (S.nu_score > 0.45).to_numpy() & ((S.n_pfp == 3).to_numpy() if n3only else True)
    along = (d * u).sum(1)
    # component towards the nearest wall: +1 means moving out
    j = dwall.argmin(1); sgn = np.where(TV[:, 0] < 0, -1, 1)
    out_x = sgn * d[:, 0]; comp = np.select([j == 0, j == 1, j == 2, j == 3], [out_x, np.sign(TV[:, 1]) * d[:, 1], -d[:, 2], d[:, 2]])
    passed = fv(S)
    for e, name in ((True, "exit"), (False, "cont")):
        m = sel & (exit_ == e); near = m & (dwall.min(1) < 10)
        print(f"   {lab:4s} {name}  n {m.sum():5d}  FV pass {passed[m].mean():.3f} | along track: median {np.median(along[m]):+.2f} cm  P(along > 2) {(along[m] > 2).mean():.3f}  P(along < -2) {(along[m] < -2).mean():.3f}"
              f" | towards nearest wall: median {np.median(comp[m]):+.2f}  P(> 2) {(comp[m] > 2).mean():.3f}  P(< -2) {(comp[m] < -2).mean():.3f}"
              f" | true vertex < 10 cm from a wall: frac {near.sum() / max(m.sum(), 1):.3f}  FV pass {passed[near].mean() if near.sum() else np.nan:.3f}  P(|d| > 5) {(np.linalg.norm(d[near], axis=1) > 5).mean() if near.sum() else np.nan:.3f}")


for n3only, head in ((True, "n_pfp = 3"), (False, "all n_pfp")):
    print(f"\nCC + pion, true vertex in FV, nu_score > 0.45, {head}: displacement components and FV pass by reco exit   [real | gen]")
    rows(R, "real", n3only); rows(G, "gen", n3only)

print("\nFV pass rate vs the true distance to the nearest wall (n_pfp = 3, nu_score > 0.45), by reco exit   [real exit / cont | gen exit / cont]")
edges = [0, 2, 5, 10, 20, 50, 200]
out = {}
for lab, S in (("real", R), ("gen", G)):
    V = S[["vtx_x", "vtx_y", "vtx_z"]].to_numpy(float); u = S[["long_dirx", "long_diry", "long_dirz"]].to_numpy(float)
    end = V + S.long_trk_len.to_numpy(float)[:, None] * u; exit_ = ~((end > LO) & (end < HI)).all(1)
    sel = base & (S.nu_score > 0.45).to_numpy() & (S.n_pfp == 3).to_numpy(); p = fv(S); dm = dwall.min(1)
    out[lab] = [(sel & exit_ & (dm >= a) & (dm < b), sel & ~exit_ & (dm >= a) & (dm < b), p) for a, b in zip(edges[:-1], edges[1:])]
for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
    cells = []
    for lab in ("real", "gen"):
        me, mc, p = out[lab][i]
        cells.append(f"{p[me].mean() if me.sum() else np.nan:.3f} ({me.sum():4d}) / {p[mc].mean() if mc.sum() else np.nan:.3f} ({mc.sum():4d})")
    print(f"   d_wall {a:3d}-{b:3d} cm | {cells[0]} | {cells[1]}")

print("\nP(reco vertex in FV) for n_pfp = 3 CC + pion slices with a longest track and nu_score > 0.45, by where the TRUE vertex is and by reco exit   [real exit / cont | gen exit / cont]")
AVLO, AVHI = np.array([-200., -200., 0.]), np.array([200., 200., 500.])
in_fv_t = ((TV > LO) & (TV < HI)).all(1); in_av_t = ((TV > AVLO) & (TV < AVHI)).all(1)
zones = [("true vertex in FV", in_fv_t), ("in AV, outside FV", in_av_t & ~in_fv_t), ("outside AV", ~in_av_t)]
base2 = ((F.n_mu >= 1) & (F.n_pi >= 1)).to_numpy() & R.long_trk_len.notna().to_numpy()
tot = {}
for lab, S in (("real", R), ("gen", G)):
    V = S[["vtx_x", "vtx_y", "vtx_z"]].to_numpy(float); u = S[["long_dirx", "long_diry", "long_dirz"]].to_numpy(float)
    end = V + S.long_trk_len.to_numpy(float)[:, None] * u; exit_ = ~((end > LO) & (end < HI)).all(1)
    sel = base2 & (S.nu_score > 0.45).to_numpy() & (S.n_pfp == 3).to_numpy(); p = fv(S)
    tot[lab] = (sel, exit_, p)
for zname, zm in zones:
    cells = []
    for lab in ("real", "gen"):
        sel, exit_, p = tot[lab]; me, mc = sel & zm & exit_, sel & zm & ~exit_
        cells.append(f"{p[me].mean() if me.sum() else np.nan:.3f} ({me.sum():5d}) / {p[mc].mean() if mc.sum() else np.nan:.3f} ({mc.sum():5d})")
    print(f"   {zname:22s} | {cells[0]} | {cells[1]}")
cells = []
for lab in ("real", "gen"):
    sel, exit_, p = tot[lab]; me, mc = sel & exit_, sel & ~exit_
    cells.append(f"{p[me].mean():.3f} ({me.sum():5d}) / {p[mc].mean():.3f} ({mc.sum():5d})")
print(f"   {'all':22s} | {cells[0]} | {cells[1]}")
print("   contribution of each zone to P(FV | exit) - P(FV | cont): zone share x pass-rate difference   [real | gen]")
for zname, zm in zones:
    cells = []
    for lab in ("real", "gen"):
        sel, exit_, p = tot[lab]; me, mc = sel & zm & exit_, sel & zm & ~exit_; ne, nc = (sel & exit_).sum(), (sel & ~exit_).sum()
        cells.append(f"{(p[me].sum() / ne - p[mc].sum() / nc):+.4f}  (share exit {me.sum() / ne:.3f} cont {mc.sum() / nc:.3f})")
    print(f"   {zname:22s} | {cells[0]} | {cells[1]}")
