#!/usr/bin/env python3
"""Swap test for a topo_last M2: feed the topology head the REAL slice block with one group of variables replaced by
the generated values (and the reverse), on the M2 test slices of truth CC + charged pion events (vertex in FV,
nu_score > 0.45), and measure the head's link P(n_pfp=3 | longest-track end outside FV) - P(n_pfp=3 | end in FV).
The group whose swap kills (restores) the link carries the joint correlation the generated block misses.
usage: diag_m2_swap.py <M2 report dir> <table dir with features.parquet>"""
import sys

import numpy as np
import pandas as pd

from xsum.gen.slc import Surrogate
from xsum.train.m2 import topo_input

D, TAB = sys.argv[1], sys.argv[2]
sur = Surrogate(D)
R = pd.read_parquet(f"{D}/real_test.parquet"); G = pd.read_parquet(f"{D}/generated_test.parquet")
F = pd.read_parquet(f"{TAB}/features.parquet", columns=sur.features).loc[R.index]
C = sur.standardise(F.to_numpy(np.float32)); cc = ((F.n_mu >= 1) & (F.n_pi >= 1)).to_numpy()
n3 = np.array([c[0] == 3 for c in sur.topo.classes] + [False])


def fv(x, y, z): return (x > -190) & (x < 190) & (y > -190) & (y < 190) & (z > 10) & (z < 450)


def link(H, base=None):
    """base: frame that defines the event selection and the end-in-FV split (default H itself), so that a swap only
    changes the head's input, not which events are compared."""
    B = H if base is None else base
    Z, M = sur.bt.standardise(sur.bt.matrix(H)); p3 = sur.topo_head.predict_proba(topo_input(C, Z, M))[:, n3].sum(1)
    m = cc & fv(B.vtx_x, B.vtx_y, B.vtx_z).to_numpy() & (B.nu_score > 0.45).to_numpy()
    e = fv(*[B[f"vtx_{a}"] + B.long_trk_len * B[f"long_dir{a}"] for a in "xyz"]).to_numpy()
    return p3[m & e].mean(), p3[m & ~e].mean()


cols = sur.bt.cols
GROUPS = {
    "vertex": ["vtx_x", "vtx_y", "vtx_z"],
    "scores+pe": ["nu_score", "bcfm_score", "total_pe"],
    "long dir": ["long_dirx", "long_diry", "long_dirz"],
    "long length+wall": ["long_trk_len", "long_wall", "long_woff"],
    "long pid/chi2": ["long_trk_pid", "long_trk_chi2_mu", "long_trk_chi2_pro", "long_trk_chi2_mu_new", "long_trk_chi2_pro_new"],
    "short group": [c for c in cols if c.startswith("short_")],
    "two group": [c for c, _, g in sur.bt.spec if g == "two"],
    "charge spheres": [c for c in cols if c.startswith("sum_integ")],
}
nr = R.n_pfp.to_numpy(); m = cc & fv(R.vtx_x, R.vtx_y, R.vtx_z).to_numpy() & (R.nu_score > 0.45).to_numpy()
e = fv(*[R[f"vtx_{a}"] + R.long_trk_len * R[f"long_dir{a}"] for a in "xyz"]).to_numpy()
print(f"events: {int(m.sum())};  real P(n_pfp=3 | end in FV / outside) {np.mean(nr[m & e] == 3):.3f} / {np.mean(nr[m & ~e] == 3):.3f}")
a, b = link(R); print(f"{'head on real block':34s} {a:.3f} / {b:.3f}  link {b - a:+.4f}")
a, b = link(G); print(f"{'head on generated block':34s} {a:.3f} / {b:.3f}  link {b - a:+.4f}")
print("\n(event split fixed by the unswapped block)\nreal block, one group replaced by generated  |  generated block, one group replaced by real")
for g, cs in GROUPS.items():
    H = R.copy(); H[cs] = G[cs].to_numpy(); a, b = link(H, R)
    K = G.copy(); K[cs] = R[cs].to_numpy(); c, d = link(K, G)
    print(f"  {g:18s} {a:.3f} / {b:.3f}  link {b - a:+.4f}   |   {c:.3f} / {d:.3f}  link {d - c:+.4f}")
