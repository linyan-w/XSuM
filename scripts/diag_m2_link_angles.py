#!/usr/bin/env python3
"""Is the missing P(n_pfp=3 | truth) dependence (forward, exiting muons of CC + pion events) carried by truth variables
that the 80 summary features do not contain, e.g. the angles between the muon and the other charged primaries?
GBDT P(n_pfp=3 | 80 features) vs P(n_pfp=3 | 80 features + pairwise angles from the particle set), on the shipped M2
test slices (truth CC + pion, vertex in FV, nu_score > 0.45), half for training; the cells of link_head_m3.txt.
usage: diag_m2_link_angles.py <shipped M2 dir> <table_matched dir> <primsets mmap dir>"""
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from xsum.train.baselines import gbdt

D, TAB, MM = sys.argv[1], sys.argv[2], sys.argv[3]
R = pd.read_parquet(f"{D}/real_test.parquet", columns=["n_pfp", "nu_score", "long_trk_len", "vtx_x", "vtx_y", "vtx_z", "long_dirx", "long_diry", "long_dirz"])
FEAT = [c for c in pq.read_schema(f"{TAB}/features.parquet").names if not c.startswith("z_") and c not in ("chunk", "__ntuple", "entry", "rec.mc.nu..index")]
F = pd.read_parquet(f"{TAB}/features.parquet", columns=FEAT).loc[R.index]
keys = np.load(f"{MM}/keys.npy", mmap_mode="r"); X = np.load(f"{MM}/X.npy", mmap_mode="r"); cls = np.load(f"{MM}/cls.npy", mmap_mode="r")
from xsum.train.m3 import pack
keys = np.asarray(keys)
want = pack(*[R.index.get_level_values(n).to_numpy() for n in ("chunk", "__ntuple", "entry", "rec.mc.nu..index")])
pos = np.searchsorted(keys, want); assert (keys[np.minimum(pos, len(keys) - 1)] == want).all(), "particle sets missing for some test rows"
order = np.argsort(pos); Xs = np.asarray(X[np.sort(pos)])[np.argsort(order)]; Cs = np.asarray(cls[np.sort(pos)])[np.argsort(order)]
print("particle sets", Xs.shape, flush=True)
MU, P, PI = 0, 2, 4
def lead(c):
    m = Cs == c; ke = np.where(m, Xs[..., 3], -1); j = ke.argmax(1); has = m.any(1)
    v = Xs[np.arange(len(Xs)), j, :3]; n = np.linalg.norm(v, axis=1); u = v / np.where(n > 0, n, 1)[:, None]
    return has, u, n
hmu, umu, pmu = lead(MU); hpi, upi, ppi = lead(PI); hp, up, pp = lead(P)
cos = lambda a, b, ha, hb: np.where(ha & hb, (a * b).sum(1), -2.0)
charged = np.isin(Cs, [1, 2, 4, 7, 10, 11]) & (Xs[..., 3] > 0.02)      # e, p, pi, K, Sigma, other (KE > 20 MeV)
ucharged = Xs[..., :3] / np.where(np.linalg.norm(Xs[..., :3], axis=-1, keepdims=True) > 0, np.linalg.norm(Xs[..., :3], axis=-1, keepdims=True), 1)
cos_mu_all = np.where(charged, (ucharged * umu[:, None, :]).sum(-1), -2.0)
A = pd.DataFrame({"cos_mu_pi": cos(umu, upi, hmu, hpi), "cos_mu_p": cos(umu, up, hmu, hp), "cos_pi_p": cos(upi, up, hpi, hp),
                  "max_cos_mu_charged": cos_mu_all.max(1), "n_charged_within15": (cos_mu_all > np.cos(np.radians(15))).sum(1),
                  "n_charged_within30": (cos_mu_all > np.cos(np.radians(30))).sum(1), "p_pi_over_p_mu": np.where(hmu & hpi, ppi / np.maximum(pmu, 1e-3), -1.0)}, index=R.index)
def fv(x, y, z): return (x > -190) & (x < 190) & (y > -190) & (y < 190) & (z > 10) & (z < 450)
sel = ((F.n_mu >= 1) & (F.n_pi >= 1) & fv(R.vtx_x, R.vtx_y, R.vtx_z) & (R.nu_score > 0.45) & R.long_trk_len.notna()).to_numpy()
n3 = (R.n_pfp == 3).to_numpy(); exr = ~fv(*[R[f"vtx_{c}"] + R.long_trk_len * R[f"long_dir{c}"] for c in "xyz"]).to_numpy()
rng = np.random.default_rng(0); ms = np.flatnonzero(sel); rng.shuffle(ms); h1, h2 = ms[: len(ms) // 2], ms[len(ms) // 2:]
X80 = F.to_numpy(np.float32); XA = np.concatenate([X80, A.to_numpy(np.float32)], 1)
pe = gbdt(X80[h1], exr[h1].astype(int), max_iter=300).predict_proba(X80)[:, 1]
ll = lambda p, y: float(-np.mean(y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1))))
P0 = gbdt(X80[h1], n3[h1].astype(int), max_iter=400).predict_proba(X80)[:, 1]
PA = gbdt(XA[h1], n3[h1].astype(int), max_iter=400).predict_proba(XA)[:, 1]
link = lambda v, e, m: float(v[m & e].mean() - v[m & ~e].mean())
s2 = np.zeros(len(R), bool); s2[h2] = True
print(f"held-out half {s2.sum()}: logloss 80 features {ll(P0[s2], n3[s2]):.4f}  + angles {ll(PA[s2], n3[s2]):.4f}  marginal {ll(np.full(s2.sum(), n3[s2].mean()), n3[s2]):.4f}")
print(f"link with the truth-predicted exit: real {link(n3, pe > 0.5, s2):+.3f}  GBDT 80 {link(P0, pe > 0.5, s2):+.3f}  GBDT 80 + angles {link(PA, pe > 0.5, s2):+.3f}")
print("P(n_pfp=3 | truth-predicted exit bin): real / GBDT 80 / GBDT 80 + angles")
q = np.quantile(pe[s2], np.linspace(0, 1, 6))
for i in range(5):
    m = s2 & (pe >= q[i]) & (pe <= q[i + 1]); print(f"   P(exit) {q[i]:.2f}-{q[i+1]:.2f}  {n3[m].mean():.3f} / {P0[m].mean():.3f} / {PA[m].mean():.3f}   n {m.sum()}")
pm, cz = F.lead_p_mu.to_numpy(), F.lead_cosz_mu.to_numpy(); qp = np.quantile(pm[s2], [0, .33, .66, 1]); qc = np.quantile(cz[s2], [0, .33, .66, 1])
print("P(n_pfp=3 | p_mu, cos theta_mu terciles): real / GBDT 80 / GBDT 80 + angles")
for i in range(3):
    row = []
    for j in range(3):
        m = s2 & (pm >= qp[i]) & (pm <= qp[i + 1]) & (cz >= qc[j]) & (cz <= qc[j + 1]); row.append(f"{n3[m].mean():.3f}/{P0[m].mean():.3f}/{PA[m].mean():.3f}")
    print(f"   p_mu {qp[i]:.2f}-{qp[i+1]:.2f} | " + "  ".join(row))
print("\nreal P(n_pfp=3) by cos(mu, pi) in the selection, and by n charged primaries within 15 deg of the muon")
for lo, hi in ((-2.5, -1.5), (-1, 0), (0, 0.7), (0.7, 0.9), (0.9, 0.97), (0.97, 1.001)):
    m = sel & (A.cos_mu_pi.to_numpy() >= lo) & (A.cos_mu_pi.to_numpy() < hi); print(f"   cos(mu,pi) {lo:5.2f}..{hi:5.2f}  P(n3) {n3[m].mean() if m.any() else np.nan:.3f}  P(exit) {exr[m].mean() if m.any() else np.nan:.3f}  n {m.sum()}")
for k in range(4):
    m = sel & (np.clip(A.n_charged_within15.to_numpy(), 0, 3) == k); print(f"   within 15 deg {k if k < 3 else '>=3'}  P(n3) {n3[m].mean():.3f}  P(exit) {exr[m].mean():.3f}  n {m.sum()}")
