#!/usr/bin/env python3
"""Summarise diag_cc1p_containment.py output: which containment condition and which track quantity differ between
real and surrogate, for signal and background.  usage: diag_cc1p_summary.py <dir with g*.parquet>"""
import glob
import sys

import numpy as np
import pandas as pd

D = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{sys.argv[1]}/g*.parquet"))], ignore_index=True)
for c in [c for c in D.columns if c[2:4] in ("pr", "tw", "c_", "co")]:  # noqa
    D[c] = D[c].fillna(False).astype(bool)
D["bkg"] = ~D.signal
print(f"neutrinos at nu_score in either sample: {len(D)}; true signal {int(D.signal.sum())}\n")

print("A. 2prong -> contained, each sample on its own 2prong population, by truth category")
print(f"{'category':28s} {'real 2pr':>8s} {'cont':>6s} {'frac':>6s} | {'XSuM 2pr':>8s} {'cont':>6s} {'frac':>6s}")
for cat in ["signal"] + sorted(D.cat.unique()):
    m = D.signal if cat == "signal" else (D.cat == cat) & ~D.signal
    r2, rc, s2, sc = D.r_two[m].sum(), D.r_contained[m].sum(), D.s_two[m].sum(), D.s_contained[m].sum()
    print(f"{('signal (truth-contained)' if cat == 'signal' else cat + ' (bkg)'):28s} {r2:8d} {rc:6d} {rc / max(r2, 1):6.3f} | {s2:8d} {sc:6d} {sc / max(s2, 1):6.3f}")

B = D[D.r_two & D.s_two]
print(f"\nB. neutrinos at 2prong in BOTH samples: {len(B)} (signal {int(B.signal.sum())}); pass rate of each condition and per-event agreement")
conds = ["c_vtx", "c_t1_start", "c_t1_end", "c_t2_start", "c_t2_end", "c_side", "contained"]
for lab, m in (("signal", B.signal), ("background", B.bkg)):
    print(f"  {lab}")
    print(f"    {'condition':12s} {'real':>6s} {'XSuM':>6s} {'agree':>6s}  {'real only':>9s} {'XSuM only':>9s}")
    for c in conds:
        r, s = B[f"r_{c}"][m], B[f"s_{c}"][m]
        print(f"    {c:12s} {r.mean():6.3f} {s.mean():6.3f} {(r == s).mean():6.3f}  {(r & ~s).mean():9.3f} {(~r & s).mean():9.3f}")


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); k = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[k], b[k])[0, 1] if k.sum() > 10 else np.nan


print("\nC. per-event real-vs-XSuM agreement of the two longest tracks (neutrinos at 2prong in both)")
print(f"    {'quantity':16s} {'t1 corr':>8s} {'t2 corr':>8s}")
for q in ["trk_len", "trk_dir_x", "trk_dir_y", "trk_dir_z", "trk_start_x", "trk_start_y", "trk_start_z", "trk_end_x", "trk_end_y", "trk_end_z"]:
    f = (lambda v: np.log(np.clip(v, 1, None))) if q == "trk_len" else (lambda v: v)
    print(f"    {q:16s} {corr(f(B[f'r_t1_{q}']), f(B[f's_t1_{q}'])):8.3f} {corr(f(B[f'r_t2_{q}']), f(B[f's_t2_{q}'])):8.3f}")
for i in (1, 2):
    sx = (np.sign(B[f"r_t{i}_trk_dir_x"]) == np.sign(B[f"s_t{i}_trk_dir_x"])).mean()
    sy = (np.sign(B[f"r_t{i}_trk_dir_y"]) == np.sign(B[f"s_t{i}_trk_dir_y"])).mean()
    print(f"    t{i}: same sign of dir_x {sx:.3f}, of dir_y {sy:.3f}")
# straight-line consistency of real tracks (is end = start + len * dir?)
for p in ("r", "s"):
    dx = B[f"{p}_t1_trk_end_x"] - (B[f"{p}_t1_trk_start_x"] + B[f"{p}_t1_trk_len"] * B[f"{p}_t1_trk_dir_x"])
    print(f"    {p}: |end_x - (start_x + len*dir_x)| median {np.nanmedian(np.abs(dx)):.1f} cm, 90% {np.nanpercentile(np.abs(dx), 90):.1f} cm")

print("\nD. length distributions of the two tracks at 2prong (each sample's own population), quantiles 25/50/75/90 [cm]")
for lab, m in (("signal", D.signal), ("background", D.bkg)):
    for i in (1, 2):
        r = D[f"r_t{i}_trk_len"][m & D.r_two]; s = D[f"s_t{i}_trk_len"][m & D.s_two]
        print(f"    {lab:10s} t{i}: real {np.round(np.nanpercentile(r, [25, 50, 75, 90]), 1)}  XSuM {np.round(np.nanpercentile(s, [25, 50, 75, 90]), 1)}")

print("\nE. truth-contained signal at 2prong in both: which condition fails (fraction of these neutrinos)")
S = B[B.signal]
for c in conds[:-1]:
    print(f"    fail {c:12s} real {(~S[f'r_{c}']).mean():.3f}  XSuM {(~S[f's_{c}']).mean():.3f}")
