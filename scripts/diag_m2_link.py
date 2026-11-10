#!/usr/bin/env python3
"""Where does the real exit <-> n_pfp link of CC + pion events come from, and why does the slice generator miss it?

On the M2 test slices (real and generated for the same neutrinos), truth CC with a charged pion, vertex in FV,
nu_score > 0.45; exit = longest track ends outside the FV.
 1. the link P(n_pfp=3 | exit) - P(n_pfp=3 | contained), real and generated
 2. the real link inside bins of single truth variables: what survives is not explained by that variable
 3. GBDT of n_pfp=3 on the 80 truth features with and without the reco exit flag: the part of the link that is
    beyond the truth (reco-internal), real vs generated
 4. is the generated exit decision right per event: agreement with the real exit, with the Geant4 muon containment
    (diagnostic reference only), and P(exit | true muon p, cos theta) real vs generated
 5. the link with the exit predicted from truth alone: is the real link reachable from the truth features
usage: diag_m2_link.py <M2 report dir> <table dir> <primgeo dir>"""
import glob
import sys

import numpy as np
import pandas as pd

from xsum.train.baselines import gbdt

D, TAB, GEO = sys.argv[1], sys.argv[2], sys.argv[3]
R = pd.read_parquet(f"{D}/real_test.parquet"); G = pd.read_parquet(f"{D}/generated_test.parquet")
if "draw" in G.index.names: G = G.droplevel("draw")
G = G.loc[R.index]
import pyarrow.parquet as pq
FEAT = [c for c in pq.read_schema(f"{TAB}/features.parquet").names if not c.startswith("z_") and c not in ("chunk", "__ntuple", "entry", "rec.mc.nu..index")]
F = pd.read_parquet(f"{TAB}/features.parquet", columns=FEAT).loc[R.index]
cc = ((F.n_mu >= 1) & (F.n_pi >= 1)).to_numpy()


def fv(x, y, z): return (x > -190) & (x < 190) & (y > -190) & (y < 190) & (z > 10) & (z < 450)


def exit_of(d): return ~fv(*[d[f"vtx_{a}"] + d.long_trk_len * d[f"long_dir{a}"] for a in "xyz"]).to_numpy()


def sel_of(d): return cc & fv(d.vtx_x, d.vtx_y, d.vtx_z).to_numpy() & (d.nu_score > 0.45).to_numpy() & d.long_trk_len.notna().to_numpy()


def link(n3, ex, m):
    return n3[m & ex].mean() - n3[m & ~ex].mean(), n3[m & ex].mean(), n3[m & ~ex].mean(), int((m & ex).sum()), int((m & ~ex).sum())


S = {p: sel_of(d) for p, d in (("real", R), ("gen", G))}
EX = {p: exit_of(d) for p, d in (("real", R), ("gen", G))}
N3 = {p: (d.n_pfp == 3).to_numpy() for p, d in (("real", R), ("gen", G))}

print("1. link P(n_pfp=3 | exit) - P(n_pfp=3 | contained), truth CC + pion, vertex in FV, nu_score > 0.45")
for p in ("real", "gen"):
    l, a, b, ne, nc = link(N3[p], EX[p], S[p])
    print(f"   {p:5s} link {l:+.3f}   P(n3|exit) {a:.3f} ({ne})   P(n3|cont) {b:.3f} ({nc})   P(exit) {EX[p][S[p]].mean():.3f}   P(n3) {N3[p][S[p]].mean():.3f}")

print("\n2. real link inside bins of one truth variable (weighted mean of the within-bin links): the part not explained by it   [real | gen]")
for v in ("lead_p_mu", "lead_cosz_mu", "lead_pt_mu", "vtx_z", "dist_back", "dist_wall_x", "dist_wall_y", "n_charged", "sumke_visible", "lead_p_pi", "n_p", "n_pi", "sumke_pi", "sum_pz"):
    out = []
    for p in ("real", "gen"):
        m = S[p]; x = F[v].to_numpy()
        q = np.unique(np.quantile(x[m], np.linspace(0, 1, 9)))
        b = np.clip(np.searchsorted(q, x, side="right") - 1, 0, len(q) - 2)
        num = den = 0.0
        for k in range(len(q) - 1):
            mk = m & (b == k)
            if (mk & EX[p]).sum() > 20 and (mk & ~EX[p]).sum() > 20:
                w = mk.sum(); num += w * (N3[p][mk & EX[p]].mean() - N3[p][mk & ~EX[p]].mean()); den += w
        out.append(num / den if den else np.nan)
    print(f"   {v:14s} within-bin link real {out[0]:+.3f} | gen {out[1]:+.3f}")

print("\n3. GBDT P(n_pfp=3 | truth [+ reco exit]) on the selection, half for training; the exit effect = mean P(.., exit=1) - P(.., exit=0)")
rng = np.random.default_rng(0)
for p in ("real", "gen"):
    m = np.flatnonzero(S[p]); rng.shuffle(m); tr, te = m[: len(m) // 2], m[len(m) // 2:]
    X = F.to_numpy(np.float32); y = N3[p].astype(int)
    c0 = gbdt(X[tr], y[tr], max_iter=300)
    Xe = np.concatenate([X, EX[p][:, None].astype(np.float32)], 1); c1 = gbdt(Xe[tr], y[tr], max_iter=300)
    ll = lambda P, yy: float(-np.mean(yy * np.log(np.clip(P, 1e-9, 1)) + (1 - yy) * np.log(np.clip(1 - P, 1e-9, 1))))
    P0 = c0.predict_proba(X[te])[:, 1]; P1 = c1.predict_proba(Xe[te])[:, 1]
    X1 = Xe[te].copy(); X1[:, -1] = 1; X0 = Xe[te].copy(); X0[:, -1] = 0
    eff = float((c1.predict_proba(X1)[:, 1] - c1.predict_proba(X0)[:, 1]).mean())
    print(f"   {p:5s} logloss truth {ll(P0, y[te]):.4f}  truth+exit {ll(P1, y[te]):.4f}   exit effect given truth {eff:+.3f}   raw link {link(N3[p], EX[p], S[p])[0]:+.3f}")

print("\n4. per-event exit decision (selection by the real frame)")
m = S["real"] & S["gen"]
er, eg = EX["real"][m], EX["gen"][m]
print(f"   P(gen exit | real exit) {eg[er].mean():.3f}   P(gen exit | real contained) {eg[~er].mean():.3f}   agreement {(er == eg).mean():.3f}   P(real exit) {er.mean():.3f}  P(gen exit) {eg.mean():.3f}")
# Geant4 muon containment of the leading muon (diagnostic reference only)
keys = pd.MultiIndex.from_frame(pd.DataFrame({"chunk": R.index.get_level_values("chunk"), "__ntuple": R.index.get_level_values("__ntuple"),
                                              "entry": R.index.get_level_values("entry"), "nu": R.index.get_level_values("rec.mc.nu..index")}))
mu_cont = pd.Series(np.nan, index=R.index)
for f in sorted(glob.glob(f"{GEO}/group_*.npz")):
    d = np.load(f); key = pd.MultiIndex.from_arrays([d["key"][:, i] for i in range(4)], names=["chunk", "__ntuple", "entry", "nu"])
    pos = key.get_indexer(keys); ok = pos >= 0
    if not ok.any(): continue
    cls = d["cls"][pos[ok]]; Gg = d["G"][pos[ok]]
    first_mu = (cls == 0).argmax(1); has = (cls == 0).any(1)
    c = Gg[np.arange(ok.sum()), first_mu, 7].astype(float); c[~has] = np.nan; c[c < -1] = np.nan
    mu_cont.iloc[np.flatnonzero(ok)] = c
tm = mu_cont.notna().to_numpy() & m; true_exit = (mu_cont.to_numpy() == 0)
print(f"   Geant4 leading muon known for {tm.sum()} of {m.sum()}: P(true exit) {true_exit[tm].mean():.3f}")
for p in ("real", "gen"):
    e = EX[p]
    print(f"   {p:5s} P(reco exit | true exit) {e[tm & true_exit].mean():.3f}   P(reco exit | true contained) {e[tm & ~true_exit].mean():.3f}   agreement with Geant4 {(e[tm] == true_exit[tm]).mean():.3f}")
print("   real link split by the Geant4 muon:  true exit  / true contained")
for p in ("real", "gen"):
    a = link(N3[p], EX[p], tm & true_exit); b = link(N3[p], EX[p], tm & ~true_exit)
    print(f"   {p:5s} link | true exit {a[0]:+.3f} (n {a[3]}/{a[4]})   link | true contained {b[0]:+.3f} (n {b[3]}/{b[4]})   P(n3 | true exit) {N3[p][tm & true_exit].mean():.3f}  P(n3 | true cont) {N3[p][tm & ~true_exit].mean():.3f}")
print("   P(exit | lead_p_mu bin) and P(n_pfp=3 | lead_p_mu bin)   [real | gen]")
x = F.lead_p_mu.to_numpy(); q = np.quantile(x[m], np.linspace(0, 1, 7))
for k in range(6):
    mk = m & (x >= q[k]) & (x < q[k + 1] if k < 5 else x <= q[k + 1])
    print(f"      p_mu {q[k]:.2f}-{q[k+1]:.2f}  exit {EX['real'][mk].mean():.3f} | {EX['gen'][mk].mean():.3f}   n3 {N3['real'][mk].mean():.3f} | {N3['gen'][mk].mean():.3f}   n {mk.sum()}")

print("\n5. link with the exit predicted from truth alone (GBDT on the real exit, half for training), applied to real and generated n_pfp")
mm = np.flatnonzero(S["real"]); rng.shuffle(mm); tr, te = mm[: len(mm) // 2], mm[len(mm) // 2:]
X = F.to_numpy(np.float32); ce = gbdt(X[tr], EX["real"][tr].astype(int), max_iter=300)
pe = ce.predict_proba(X)[:, 1]; pred = pe > 0.5
from sklearn.metrics import roc_auc_score
print(f"   AUC of the truth-predicted exit vs real exit {roc_auc_score(EX['real'][te], pe[te]):.3f}, vs gen exit {roc_auc_score(EX['gen'][te], pe[te]):.3f}; accuracy at 0.5: real {(pred[te] == EX['real'][te]).mean():.3f} gen {(pred[te] == EX['gen'][te]).mean():.3f}")
sel = np.zeros(len(R), bool); sel[te] = True
for p in ("real", "gen"):
    l = link(N3[p], pred, sel & S[p]); print(f"   {p:5s} link with the truth-predicted exit {l[0]:+.3f}   with its own reco exit {link(N3[p], EX[p], sel & S[p])[0]:+.3f}")

print("\n6. inside the bins of the truth-predicted exit probability: P(n_pfp=3 | bin, reco exit / contained) and by the Geant4 muon   [real | gen]")
q = np.quantile(pe[sel & S["real"]], np.linspace(0, 1, 6))
for k in range(5):
    b = sel & (pe >= q[k]) & (pe <= q[k + 1]); row = []
    for p in ("real", "gen"):
        mb = b & S[p]; row.append(f"{N3[p][mb & EX[p]].mean():.3f}/{N3[p][mb & ~EX[p]].mean():.3f} (link {N3[p][mb & EX[p]].mean() - N3[p][mb & ~EX[p]].mean():+.3f})")
    mb = b & S["real"] & tm
    row.append(f"Geant4 exit/cont real {N3['real'][mb & true_exit].mean():.3f}/{N3['real'][mb & ~true_exit].mean():.3f} gen {N3['gen'][mb & true_exit].mean():.3f}/{N3['gen'][mb & ~true_exit].mean():.3f}")
    print(f"   P(exit) {q[k]:.2f}-{q[k+1]:.2f}  real {row[0]} | gen {row[1]} | {row[2]}")
print("   check: P(n_pfp=3) real over the half-sample selection, with the gen selection too, with the Geant4 muon known:",
      f"{N3['real'][sel & S['real']].mean():.3f}  {N3['real'][sel & S['real'] & S['gen']].mean():.3f}  {N3['real'][sel & S['real'] & S['gen'] & mu_cont.notna().to_numpy()].mean():.3f}",
      f"| rows {int((sel & S['real']).sum())} {int((sel & S['real'] & S['gen']).sum())} {int((sel & S['real'] & S['gen'] & mu_cont.notna().to_numpy()).sum())}",
      f"| P(n3) of gen-selected-only rows {N3['real'][sel & S['real'] & ~S['gen']].mean():.3f}")

print("\n7. the link with weaker selections (each sample its own population): is the deficit in the counts or in the nu_score / vertex selection?")
for name, cut in (("CC + pion, longest track present", lambda d: cc & d.long_trk_len.notna().to_numpy()),
                  ("+ vertex in FV", lambda d: cc & d.long_trk_len.notna().to_numpy() & fv(d.vtx_x, d.vtx_y, d.vtx_z).to_numpy()),
                  ("+ nu_score > 0.45", lambda d: cc & d.long_trk_len.notna().to_numpy() & fv(d.vtx_x, d.vtx_y, d.vtx_z).to_numpy() & (d.nu_score > 0.45).to_numpy()),
                  ("+ nu_score > 0.45, n_pfp <= 4", lambda d: cc & d.long_trk_len.notna().to_numpy() & fv(d.vtx_x, d.vtx_y, d.vtx_z).to_numpy() & (d.nu_score > 0.45).to_numpy() & (d.n_pfp <= 4).to_numpy())):
    row = []
    for p, d in (("real", R), ("gen", G)):
        mm_ = cut(d); l = link(N3[p], EX[p], mm_)
        row.append(f"{p} n {mm_.sum()} P(n3) {N3[p][mm_].mean():.3f} P(exit) {EX[p][mm_].mean():.3f} link {l[0]:+.3f} [P(n3|exit) {l[1]:.3f} P(n3|cont) {l[2]:.3f}]")
    print(f"   {name:34s} " + " | ".join(row))
print("   P(nu_score > 0.45 | n_pfp=3, exit / contained) and P(vertex in FV | n_pfp=3, exit / contained), CC + pion with a longest track   [real | gen]")
for p, d in (("real", R), ("gen", G)):
    b = cc & d.long_trk_len.notna().to_numpy() & N3[p]; ns = (d.nu_score > 0.45).to_numpy(); f_ = fv(d.vtx_x, d.vtx_y, d.vtx_z).to_numpy()
    print(f"   {p:5s} nu_score pass | exit {ns[b & EX[p]].mean():.3f}  cont {ns[b & ~EX[p]].mean():.3f}    FV | exit {f_[b & EX[p]].mean():.3f}  cont {f_[b & ~EX[p]].mean():.3f}    n3 events {b.sum()}")

print("\n8. the reco vertex of n_pfp=3 CC + pion slices whose TRUE vertex is in the FV, by reco exit: pass rates of each FV boundary and the reco-true vertex displacement   [real | gen]")
tfv = fv(F.vtx_x, F.vtx_y, F.vtx_z).to_numpy()
for p, d in (("real", R), ("gen", G)):
    b = cc & d.long_trk_len.notna().to_numpy() & N3[p] & tfv
    dx, dy, dz = (d[f"vtx_{c}"].to_numpy() - F[f"vtx_{c}"].to_numpy() for c in "xyz"); dr = np.sqrt(dx**2 + dy**2 + dz**2)
    for lab, e in (("exit", EX[p]), ("cont", ~EX[p])):
        m_ = b & e
        print(f"   {p:5s} {lab}  n {m_.sum():5d}  reco FV {fv(d.vtx_x, d.vtx_y, d.vtx_z).to_numpy()[m_].mean():.3f}  |x|<190 {(d.vtx_x.abs() < 190).to_numpy()[m_].mean():.3f}  |y|<190 {(d.vtx_y.abs() < 190).to_numpy()[m_].mean():.3f}  z>10 {(d.vtx_z > 10).to_numpy()[m_].mean():.3f}  z<450 {(d.vtx_z < 450).to_numpy()[m_].mean():.3f}"
              f"  | displacement median {np.median(dr[m_]):.1f} cm  P(>5 cm) {(dr[m_] > 5).mean():.3f}  P(>20 cm) {(dr[m_] > 20).mean():.3f}  P(>50 cm) {(dr[m_] > 50).mean():.3f}")
print("   same for ALL CC + pion slices with a longest track and the true vertex in the FV (not only n_pfp=3)")
for p, d in (("real", R), ("gen", G)):
    b = cc & d.long_trk_len.notna().to_numpy() & tfv
    dx, dy, dz = (d[f"vtx_{c}"].to_numpy() - F[f"vtx_{c}"].to_numpy() for c in "xyz"); dr = np.sqrt(dx**2 + dy**2 + dz**2)
    for lab, e in (("exit", EX[p]), ("cont", ~EX[p])):
        m_ = b & e
        print(f"   {p:5s} {lab}  n {m_.sum():5d}  reco FV {fv(d.vtx_x, d.vtx_y, d.vtx_z).to_numpy()[m_].mean():.3f}  | displacement P(>5 cm) {(dr[m_] > 5).mean():.3f}  P(>20 cm) {(dr[m_] > 20).mean():.3f}  P(>50 cm) {(dr[m_] > 50).mean():.3f}   P(n3) {N3[p][m_].mean():.3f}")
print("   P(n_pfp=3 | displacement > 20 cm / <= 20 cm), CC + pion, true vertex in FV, longest track present   [real | gen]")
for p, d in (("real", R), ("gen", G)):
    b = cc & d.long_trk_len.notna().to_numpy() & tfv
    dr = np.sqrt(sum((d[f"vtx_{c}"].to_numpy() - F[f"vtx_{c}"].to_numpy())**2 for c in "xyz"))
    print(f"   {p:5s} P(n3 | far) {N3[p][b & (dr > 20)].mean():.3f} (n {(b & (dr > 20)).sum()})  P(n3 | near) {N3[p][b & (dr <= 20)].mean():.3f}   P(exit | far) {EX[p][b & (dr > 20)].mean():.3f}  P(exit | near) {EX[p][b & (dr <= 20)].mean():.3f}")
