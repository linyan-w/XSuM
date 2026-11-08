#!/usr/bin/env python3
"""Is the 2prong <-> containment correlation carried by the slice topology counts (M2's topology class in the surrogate)
or lost between the slice counts and the PFP-level 2prong of the CC1p0pi selection?
Per truth category, at the nu_score stage: cells of (slice n_trk, PFP 2prong), population share and the fraction of
events whose longest good track ends in the FV, real vs surrogate.
usage: diag_cc1p_topo.py <dir with g*.parquet from diag_cc1p_containment.py (with slice counts)>"""
import glob
import sys

import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)
D = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{sys.argv[1]}/g*.parquet"))], ignore_index=True)
for c in [c for c in D.columns if c[2:4] in ("pr", "tw", "c_", "co")]:
    D[c] = D[c].fillna(False).astype(bool)
CATS = [("1mu1p0pi", D.signal | (D.cat == "numu CC 1p0pi (any cont.)")), ("CC Npi", (D.cat == "numu CC Npi") & ~D.signal),
        ("NC", (D.cat == "NC") & ~D.signal), ("CC other 0pi", (D.cat == "numu CC other 0pi") & ~D.signal)]


def cell(a, p):
    return f"{len(a):5d} {a[p + '_c_t1_end'].mean() if len(a) else np.nan:6.3f}"


for lab, m in CATS:
    print(f"\n{lab}: cells (slice n_trk, PFP 2prong) -> n, longest-track end in FV   [real | surrogate]")
    A = {p: D[m & D[p + "_pre"] & D[p + "_t1_trk_len"].notna()] for p in "rs"}
    for p in "rs":
        A[p] = A[p].assign(k=np.clip(A[p][p + "_n_trk"].fillna(-1), -1, 3).astype(int))
    for k in (1, 2, 3):
        for two in (True, False):
            r = A["r"][(A["r"].k == k) & (A["r"].r_two == two)]; s = A["s"][(A["s"].k == k) & (A["s"].s_two == two)]
            print(f"   n_trk {'>=3' if k == 3 else k:>3} 2prong {str(two):5s} | {cell(r, 'r')} | {cell(s, 's')}")
    for p in "rs":
        a = A[p]; t = a[p + "_two"]
        print(f"   {p}: P(slice n_trk=2 | 2prong) {(a.k[t] == 2).mean():.3f}   P(2prong | slice n_trk=2) {t[a.k == 2].mean():.3f}"
              f"   n_good vs n_trk equal {(a[p + '_n_good'] == a[p + '_n_trk']).mean():.3f}")

print("\nby slice n_pfp (2prong is ~ n_pfp = 3): n, longest-track end in FV, P(2prong)   [real | surrogate]")
for lab, m in CATS:
    print(f"  {lab}")
    A = {p: D[m & D[p + "_pre"] & D[p + "_t1_trk_len"].notna()] for p in "rs"}
    for n in range(2, 7):
        cells = []
        for p in "rs":
            k = np.clip(A[p][p + "_n_pfp"].fillna(-1), 0, 6).astype(int) == n; a = A[p][k]
            cells.append(f"{len(a):5d} {a[p + '_c_t1_end'].mean() if len(a) else np.nan:6.3f} {a[p + '_two'].mean() if len(a) else np.nan:6.3f}")
        print(f"    n_pfp {n if n < 6 else '>=6':>3} | {cells[0]} | {cells[1]}")
