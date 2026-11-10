#!/usr/bin/env python3
"""Retrain only the topology head of a shipped M2 (class sampled first, from truth features + z) with more capacity /
epochs, on the same inputs, classes and file split; the flows are untouched (they are conditioned on the class index).
Writes <out>/topology_head.pt and links model.pt / heads.pkl from the shipped directory, so <out> is a drop-in M2
for generate_slc.py. Reports the multi-class logloss and, for truth CC + pion test slices, P(n_pfp=3) real / shipped
head / new head in (p_mu, cos theta_mu) cells and against the truth-predicted exit (the region the shipped head
underfits, reports/cc1p_containment/link_head_m3.txt).
usage: retrain_topo_head.py <table_matched dir> <shipped M2 dir> <out dir> [--hidden 1024 --layers 4 --epochs 40 --lr 1e-3 --bs 8192]"""
import argparse
import json
import os
import pathlib
import time

import numpy as np
import pandas as pd
import torch

from xsum.train.baselines import gbdt
from xsum.train.m2 import TOPO_COLS, MLPClassifier, Topology, split_by_level

ap = argparse.ArgumentParser()
ap.add_argument("table_dir"); ap.add_argument("shipped"); ap.add_argument("out")
ap.add_argument("--hidden", type=int, default=1024); ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--bs", type=int, default=8192)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
t0 = time.time(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
ck = torch.load(f"{a.shipped}/model.pt", map_location="cpu", weights_only=False)
names, std_m, std_s = ck["features"], ck["std_m"], ck["std_s"]

F = pd.read_parquet(f"{a.table_dir}/features.parquet", columns=names)
X = np.ascontiguousarray(F.to_numpy(np.float32))
# the shipped model's test files (its split was drawn on a 3M-row subsample, so it is not reproducible from the full
# table): test = the files of its real_test.parquet; val = 10% of the remaining files; train = the rest
file_id = F.index.get_level_values("chunk").to_numpy() * 100000 + F.index.get_level_values("__ntuple").to_numpy()
Ri = pd.read_parquet(f"{a.shipped}/real_test.parquet", columns=["n_pfp"]).index
test_files = np.unique(Ri.get_level_values("chunk").to_numpy() * 100000 + Ri.get_level_values("__ntuple").to_numpy())
te = np.isin(file_id, test_files)
rest = np.unique(file_id[~te]); vf = np.random.default_rng(a.seed).choice(rest, len(rest) // 10, replace=False)
va = np.isin(file_id, vf); tr = ~te & ~va
T = pd.read_parquet(f"{a.table_dir}/targets.parquet", columns=TOPO_COLS + ["nu_score", "long_trk_len", "vtx_x", "vtx_y", "vtx_z", "long_dirx", "long_diry", "long_dirz"])
assert T.index.equals(F.index)
C = ((np.arcsinh(X) - std_m) / std_s).astype(np.float32); del X
topo = Topology.__new__(Topology)
topo.classes = [tuple(c) for c in ck["topology_classes"]]; topo.lookup = {t: i for i, t in enumerate(topo.classes)}; topo.other = len(topo.classes)
k = topo.encode(T); n_cls = topo.other + 1
print(f"rows {len(F)}  train {tr.sum()}  val {va.sum()}  test {te.sum()}  classes {n_cls}  ({time.time()-t0:.0f} s)", flush=True)

rng = np.random.default_rng(a.seed)
iva = np.flatnonzero(va); iva = np.sort(rng.choice(iva, min(len(iva), 500_000), replace=False))
head = MLPClassifier(C.shape[1], n_cls, hidden=a.hidden, n_layers=a.layers, device="cuda", epochs=a.epochs, bs=a.bs, lr=a.lr, seed=a.seed)
head.fit(C[tr], k[tr], C[iva], k[iva])
print(f"trained ({time.time()-t0:.0f} s)", flush=True)
th = torch.load(f"{a.shipped}/topology_head.pt", map_location="cuda", weights_only=False)
old = MLPClassifier(th["n_in"], th["n_classes"], device="cuda"); old.net.load_state_dict(th["state"])

ite = np.flatnonzero(te); Pn, Po = head.predict_proba(C[ite]), old.predict_proba(C[ite])
ll = lambda P: float(-np.log(np.clip(P[np.arange(len(P)), k[ite]], 1e-9, None)).mean())
res = {"config": vars(a), "logloss_new": ll(Pn), "logloss_shipped": ll(Po), "accuracy_new": float((Pn.argmax(1) == k[ite]).mean()), "accuracy_shipped": float((Po.argmax(1) == k[ite]).mean())}
print("test logloss new %.4f shipped %.4f | accuracy new %.4f shipped %.4f" % (res["logloss_new"], res["logloss_shipped"], res["accuracy_new"], res["accuracy_shipped"]), flush=True)

# the CC + pion region
Ft = F.iloc[ite]; Tt = T.iloc[ite]
def fv(x, y, z): return (x > -190) & (x < 190) & (y > -190) & (y < 190) & (z > 10) & (z < 450)
sel = ((Ft.n_mu >= 1) & (Ft.n_pi >= 1) & fv(Tt.vtx_x, Tt.vtx_y, Tt.vtx_z) & (Tt.nu_score > 0.45) & Tt.long_trk_len.notna()).to_numpy()
n3c = np.array([c[0] == 3 for c in topo.classes] + [False])
n3r = (Tt.n_pfp == 3).to_numpy(); pn, po = Pn[:, n3c].sum(1), Po[:, n3c].sum(1)
samp = lambda P: (P.cumsum(1) > rng.random(len(P))[:, None]).argmax(1)
sn, so = n3c[samp(Pn)], n3c[samp(Po)]
# truth-predicted exit: GBDT on the real reco exit from the 80 non-z features, half of the selection
TT = Tt
exr = ~fv(*[TT[f"vtx_{c}"] + TT.long_trk_len * TT[f"long_dir{c}"] for c in "xyz"]).to_numpy()
feat80 = [c for c in names if not c.startswith("z_")]; X80 = Ft[feat80].to_numpy(np.float32)
ms = np.flatnonzero(sel); rng.shuffle(ms); h1, h2 = ms[: len(ms) // 2], ms[len(ms) // 2:]
pe = gbdt(X80[h1], exr[h1].astype(int), max_iter=300).predict_proba(X80)[:, 1]
s2 = np.zeros(len(ite), bool); s2[h2] = True
link = lambda v, e, m: float(v[m & e].mean() - v[m & ~e].mean())
print(f"\nCC + pion test slices {sel.sum()}, P(n_pfp=3) real {n3r[sel].mean():.3f} shipped head {po[sel].mean():.3f} new head {pn[sel].mean():.3f}; link with the truth-predicted exit (> 0.5), half-sample: real {link(n3r, pe > 0.5, s2):+.3f}  shipped head {link(po, pe > 0.5, s2):+.3f}  new head {link(pn, pe > 0.5, s2):+.3f}  sampled shipped {link(so, pe > 0.5, s2):+.3f}  sampled new {link(sn, pe > 0.5, s2):+.3f}")
print("P(n_pfp=3 | truth-predicted exit probability bin): real / shipped head / new head")
q = np.quantile(pe[sel], np.linspace(0, 1, 6)); cells = {}
for i in range(5):
    m = sel & (pe >= q[i]) & (pe <= q[i + 1]); cells[f"pexit {q[i]:.2f}-{q[i+1]:.2f}"] = (float(n3r[m].mean()), float(po[m].mean()), float(pn[m].mean()))
    print(f"   P(exit) {q[i]:.2f}-{q[i+1]:.2f}  {n3r[m].mean():.3f} / {po[m].mean():.3f} / {pn[m].mean():.3f}   n {m.sum()}")
pm, cz = Ft.lead_p_mu.to_numpy(), Ft.lead_cosz_mu.to_numpy(); qp = np.quantile(pm[sel], [0, .33, .66, 1]); qc = np.quantile(cz[sel], [0, .33, .66, 1])
print("P(n_pfp=3 | p_mu tercile, cos theta_mu tercile): real / shipped head / new head   (rows p_mu low->high, cols cos theta low->high)")
for i in range(3):
    row = []
    for j in range(3):
        m = sel & (pm >= qp[i]) & (pm <= qp[i + 1]) & (cz >= qc[j]) & (cz <= qc[j + 1])
        row.append(f"{n3r[m].mean():.3f}/{po[m].mean():.3f}/{pn[m].mean():.3f}"); cells[f"pmu{i} cosz{j}"] = (float(n3r[m].mean()), float(po[m].mean()), float(pn[m].mean()))
    print(f"   p_mu {qp[i]:.2f}-{qp[i+1]:.2f} | " + "  ".join(row))
res["cells_real_shipped_new"] = cells

torch.save({"state": head.net.state_dict(), "n_in": C.shape[1], "n_classes": n_cls}, out / "topology_head.pt")
for f in ("model.pt", "heads.pkl"):
    if not (out / f).exists(): os.symlink(os.path.abspath(f"{a.shipped}/{f}"), out / f)
(out / "head_metrics.json").write_text(json.dumps(res, indent=1, default=float))
print(f"done in {time.time()-t0:.0f} s -> {out}", flush=True)
