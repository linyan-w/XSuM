"""M4: the PFP set of a slice. Conditioned on the truth features, the topology class and the slice block, a
classifier head gives the number of fitted PFPs (and of empty ones), a masked set flow generates their records.
Closure: per-field marginals and the cohpi multiplicities recomputed from generated PFPs against the real ones.
"""
from __future__ import annotations

import glob
import json
import pathlib
import pickle
import time

import numpy as np
import pandas as pd
import torch

from ..data.pfpset_tf import PfpTransform
from ..data.splits import split_by_level
from ..eval import plots
from ..eval.metrics import sample_vs_real_1d
from ..models.pfpflow import PfpFlow
from ..prep.pfpset import FIELDS, LAYOUT, N_MAX, NAMES, snap_walls, track_len, v2_to_v3, v3_to_v4
from .baselines import Standardizer
from .m2 import SPEC, TOPO_COLS, BlockTransform, MLPClassifier, Topology, mask_from_presence, presence

N_EMPTY_MAX = 3


def load_sets(pfpsets_dir, max_slices=None, seed=0):
    """Concatenate the group files; with max_slices, keep a random fraction of every group (memory: the subsample only).
    Older layouts are converted group by group into one preallocated v4 array (no concatenation copy)."""
    files = sorted(glob.glob(f"{pfpsets_dir}/group_*.npz")); rng = np.random.default_rng(seed)
    sizes = [len(np.load(f)["n_fit"]) for f in files]; tot = sum(sizes); frac = 1.0 if not max_slices or tot <= max_slices else max_slices / tot
    keeps = [np.sort(rng.choice(n, int(round(frac * n)), replace=False)) if frac < 1 else np.arange(n) for n in sizes]
    ntot = sum(len(k) for k in keeps)
    out = {"X": np.empty((ntot, N_MAX, len(FIELDS)), np.float32)}; o = 0
    for f, keep in zip(files, keeps):
        d = np.load(f); n = len(keep)
        x = d["X"][keep]; layout = str(d["layout"]) if "layout" in d.files else "v2"
        if layout == "v2": v2_to_v3(x); layout = "v3"
        out["X"][o:o + n] = v3_to_v4(x) if layout == "v3" else x; del x
        for k in ("key", "n_fit", "n_empty", "truth_pdg"):
            v = d[k][keep]
            if k not in out: out[k] = np.empty((ntot,) + v.shape[1:], v.dtype)
            out[k][o:o + n] = v
        o += n
    return out


def longest_track_end(X, valid):
    """Per slice, the longest track: (end from the generated end point [n,3], end from start + length * direction [n,3],
    start [n,3]); NaN where the slice has no track."""
    g = lambda n: X[..., NAMES.index(n)]
    L = np.where((g("has_trk") > 0.5) & valid, track_len(X), -1.0); L = np.nan_to_num(L, nan=-1.0)
    i = L.argmax(1); r = np.arange(len(X)); ok = L[r, i] > 0
    st = np.stack([g(f"start_{a}")[r, i] for a in "xyz"], 1); dsp = np.stack([g(f"disp_{a}")[r, i] for a in "xyz"], 1)
    dr = np.stack([g(f"dir_{a}")[r, i] for a in "xyz"], 1); Lk = L[r, i][:, None]
    nan = lambda v: np.where(ok[:, None], v, np.nan)
    return nan(st + dsp), nan(st + Lk * dr), nan(st)


TRUTH_CLASSES = ("CC Npi", "CC 0pi 1p", "CC 0pi other", "NC")


def truth_class(F, cols):
    """Generator-truth class per slice from the M3 feature table (n_mu, n_pi, n_p, n_e counts): CC Npi, CC 0pi 1p,
    CC 0pi other, NC (no lepton); -1 = other (nue CC)."""
    g = lambda n: F[:, cols.index(n)]
    mu, pi, pr, e = g("n_mu") > 0, g("n_pi") > 0, g("n_p") == 1, g("n_e") > 0
    c = np.full(len(F), -1, np.int8)
    c[mu & pi] = 0; c[mu & ~pi & pr] = 1; c[mu & ~pi & ~pr] = 2; c[~mu & ~e] = 3
    return c


def end_point_metrics(Xr, vr, Xg, vg, cls=None):
    """Same test slices, real vs generated longest track: per-event end-point error (generated end point and the
    straight-line end start + len * dir), distribution-level FV containment of the end, lateral deviation from the
    straight line."""
    er, er_line, sr = longest_track_end(Xr, vr); eg, eg_line, sg = longest_track_end(Xg, vg)
    both = np.isfinite(er[:, 0]) & np.isfinite(eg[:, 0])
    fv = lambda e: (np.abs(e[:, 0]) < 190) & (np.abs(e[:, 1]) < 190) & (e[:, 2] > 10) & (e[:, 2] < 450)
    q = lambda v: {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "frac_gt10": float(np.mean(v > 10))}
    out = {"n": int(both.sum()),
           "err_end": q(np.linalg.norm(eg[both] - er[both], axis=1)),
           "err_line_end": q(np.linalg.norm(eg_line[both] - er[both], axis=1)),
           "lateral_real": q(np.linalg.norm(er[both] - er_line[both], axis=1)), "lateral_gen": q(np.linalg.norm(eg[both] - eg_line[both], axis=1)),
           "end_in_fv_real": float(fv(er[np.isfinite(er[:, 0])]).mean()), "end_in_fv_gen": float(fv(eg[np.isfinite(eg[:, 0])]).mean()),
           "line_end_in_fv_gen": float(fv(eg_line[np.isfinite(eg_line[:, 0])]).mean())}
    # distance of the end to the TPC wall: outside (< -3 cm), on the wall (+-3), 3-10, 10-20 (FV band), > 20 cm
    wall = lambda e: np.minimum.reduce([201.3 - np.abs(e[:, 0]), 203.7 - np.abs(e[:, 1]), e[:, 2], 500.8 - e[:, 2]])
    bins = [-1e9, -3, 3, 10, 20, 1e9]; h = lambda e: (np.histogram(wall(e[np.isfinite(e[:, 0])]), bins)[0] / max(np.isfinite(e[:, 0]).sum(), 1)).tolist()
    out["wall_hist_bins"] = ["<-3", "-3..3", "3..10", "10..20", ">20"]
    out["wall_hist_real"], out["wall_hist_gen"], out["wall_hist_line_gen"] = h(er), h(eg), h(eg_line)
    if cls is not None:   # the same per generator-truth class: does the end distribution follow the event, not only the average?
        out["by_class"] = {}
        for k, name in enumerate(TRUTH_CLASSES):
            m = cls == k; fr, fg = m & np.isfinite(er[:, 0]), m & np.isfinite(eg[:, 0])
            if fr.sum() < 100: continue
            out["by_class"][name] = {"n": int(m.sum()), "end_in_fv_real": float(fv(er[fr]).mean()), "end_in_fv_gen": float(fv(eg[fg]).mean()),
                                     "on_wall_real": float((np.abs(wall(er[fr])) < 3).mean()), "on_wall_gen": float((np.abs(wall(eg[fg])) < 3).mean())}
    return out


def cc1p_selection(X, valid, vtx):
    """The CC1p0pi 2prong and 2prong-contained cuts on a raw PFP-set array with the slice vertex vtx [n,3]:
    exactly two tracks with length > 0 within 100 cm of the vertex; vertex, both starts and both ends in the FV and on
    one side of the cathode (10 cm margin) -> (two [n], contained [n])."""
    g = lambda n: X[..., NAMES.index(n)]
    L = np.nan_to_num(track_len(X), nan=-1.0)
    good = (g("has_trk") > 0.5) & valid & (g("dist_to_vertex") < 100) & (L > 0)
    fv = lambda x, y, z: (np.abs(x) < 190) & (np.abs(y) < 190) & (z > 10) & (z < 450)
    two = (good.sum(1) == 2) & fv(*vtx.T)
    Ls = np.where(good, L, -1.0); o = np.argsort(-Ls, 1)[:, :2]; r = np.arange(len(X))[:, None]
    st = np.stack([g(f"start_{a}")[r, o] for a in "xyz"], -1); en = st + np.stack([g(f"disp_{a}")[r, o] for a in "xyz"], -1)
    P = np.concatenate([vtx[:, None], st, en], 1)                      # [n, 5, 3]
    ok = fv(P[..., 0], P[..., 1], P[..., 2]).all(1) & ((P[..., 0] < -10).all(1) | (P[..., 0] > 10).all(1))
    return two, two & ok


def cohpi_counts(X, valid):
    """n_prong, n_trk, n_proton, n_shower from a raw PFP-set array (cohpi definitions on PFPs within 50 cm)."""
    g = lambda n: X[..., NAMES.index(n)]
    has_trk = (g("has_trk") > 0.5) & valid; near50 = (g("dist_to_vertex") < 50) & valid
    near = has_trk & (g("dist_to_vertex") < 6) & (track_len(X) > 4)
    trk = near & (g("trackScore") > 0.5)
    return {"n_prong": near.sum(1), "n_trk": trk.sum(1), "n_proton": (trk & (g("chi2_mu_new") > 25)).sum(1), "n_shower": (near50 & (g("trackScore") < 0.5)).sum(1)}


def run(table_dir, pfpsets_dir, out_dir, device="cuda", epochs=20, seed=0, max_slices=3_000_000, d_model=192, n_layers=4, bs=1024, lr=5e-4,
        primsets_dir=None, n_part_layers=2, geo=False):
    """primsets_dir: true particle tokens (xsum.prep.primset, mmap layout) for PFP-to-particle cross-attention; None = off."""
    out = pathlib.Path(out_dir); (out / "figures").mkdir(parents=True, exist_ok=True); t0 = time.time()
    S = load_sets(pfpsets_dir, max_slices, seed)
    T = pd.read_parquet(f"{table_dir}/targets.parquet", columns=[c for c, _, _ in SPEC] + TOPO_COLS)
    F = pd.read_parquet(f"{table_dir}/features.parquet")
    # align sets with the table rows by neutrino key
    kidx = pd.MultiIndex.from_arrays([S["key"][:, i] for i in range(4)], names=T.index.names)
    pos = T.index.get_indexer(kidx); ok = pos >= 0
    print(f"sets {len(kidx)}, table {len(T)}, aligned {ok.sum()}  ({time.time()-t0:.0f} s)", flush=True)
    rng = np.random.default_rng(seed)
    sel = np.flatnonzero(ok)
    if max_slices and len(sel) > max_slices: sel = np.sort(rng.choice(sel, max_slices, replace=False))
    X = S["X"]
    if len(sel) < len(X):   # in-place compaction (sel sorted: every read index >= its write index), no second copy
        for i in range(0, len(sel), 200_000): c = sel[i:i + 200_000]; X[i:i + len(c)] = X[c]
        X = X[:len(sel)]
    n_fit = np.clip(S["n_fit"][sel], 0, N_MAX).astype(int); n_empty = np.clip(S["n_empty"][sel], 0, N_EMPTY_MAX).astype(int); tpdg = S["truth_pdg"][sel]
    T = T.iloc[pos[sel]]; Fi = F.index[pos[sel]]; fcols = list(F.columns); F = F.to_numpy(np.float32)[pos[sel]]; del S
    tcls = truth_class(F, fcols)
    if primsets_dir:   # keys now; the particle arrays themselves are loaded after the conditioning (memory peak)
        from .m3 import pack
        g = lambda n: Fi.get_level_values(n).to_numpy()
        pkeys = pack(g("chunk"), g("__ntuple"), g("entry"), g("rec.mc.nu..index"))
    else:
        n_part_layers = 0; geo = False
    file_id = Fi.get_level_values("chunk").to_numpy() * 100000 + Fi.get_level_values("__ntuple").to_numpy(); del Fi
    sp = split_by_level(pd.MultiIndex.from_arrays([file_id], names=["file"]), level="file", seed=seed); tr, va, te = sp["train"], sp["val"], sp["test"]
    # conditioning: features, topology one-hot, slice block
    std = Standardizer(F[tr]); C = std(F); del F
    topo = Topology(T, tr); k = topo.encode(T); H = topo.onehot(k)
    bt = BlockTransform(); Y = bt.matrix(T).astype(np.float32); bt.fit(Y[tr]); Z, Mfin = bt.standardise(Y); del Y
    Mg = mask_from_presence(presence(T, bt), bt) * Mfin; Zc = Z * Mg
    cond = np.concatenate([C, H, Zc], 1).astype(np.float32); del C, H, Zc, Z, Mg, Mfin
    print(f"cond dim {cond.shape[1]}, train {tr.sum()} val {va.sum()} test {te.sum()}  ({time.time()-t0:.0f} s)", flush=True)
    metrics = {"n": int(len(cond)), "cond_dim": int(cond.shape[1]), "epochs": epochs, "d_model": d_model, "n_layers": n_layers, "n_part_layers": n_part_layers, "geo": geo, "layout": LAYOUT}

    # heads: number of fitted PFPs and of empty PFPs
    heads = {}
    for name, y, K in (("n_fit", n_fit, N_MAX + 1), ("n_empty", n_empty, N_EMPTY_MAX + 1)):
        heads[name] = MLPClassifier(cond.shape[1], K, device=device, seed=seed).fit(cond[tr], y[tr], cond[va], y[va])
        P = heads[name].predict_proba(cond[te])
        metrics[name] = {"logloss": float(-np.log(np.clip(P[np.arange(len(P)), y[te]], 1e-9, None)).mean()),
                         "logloss_marginal": float(-np.log(np.clip((np.bincount(y[tr], minlength=K) / tr.sum())[y[te]], 1e-9, None)).mean()),
                         "accuracy": float((P.argmax(1) == y[te]).mean()), "marginal_true": np.bincount(y[te], minlength=K).tolist(), "mean_prob": P.mean(0).tolist()}
        print(name, {kk: round(v, 4) for kk, v in metrics[name].items() if isinstance(v, float)}, flush=True)

    # set flow
    itr_all = np.flatnonzero(tr); sub = itr_all if len(itr_all) <= 500_000 else np.sort(np.random.default_rng(seed).choice(itr_all, 500_000, replace=False))
    pt = PfpTransform().fit(X[sub], seed)
    Xr = X[np.flatnonzero(te)]              # only the test records are needed later (closure)
    Zs, valid = pt.forward(X, rng, inplace=True); del X   # model space overwrites the raw records (saves one 3M x 12 x 32 array)
    if n_part_layers:
        from .m3 import PrimStore
        store = PrimStore(primsets_dir, geo=geo); pcls, pX = store.take(pkeys, slice(0, 3))
        pG = store.take_geo(pkeys) if geo else None; del store, pkeys
        print(f"particle tokens {pcls.shape}{' + Geant4 geometry' if geo else ''}  ({time.time()-t0:.0f} s)", flush=True)
    model = PfpFlow(Zs.shape[-1], cond.shape[1], d_model, n_layers, n_part_layers=n_part_layers, geo=geo).to(device)
    def partb(idx):
        if not n_part_layers: return None
        out = (torch.tensor(pcls[idx], device=device).long(), torch.tensor(pX[idx], device=device))
        return out + (torch.tensor(pG[idx], device=device),) if geo else out
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    itr, iva = np.flatnonzero(tr), np.flatnonzero(va)
    def batch(idx):
        return (torch.tensor(Zs[idx], device=device), torch.tensor(cond[idx], device=device), torch.tensor(valid[idx], device=device), partb(idx))
    hist, best, best_state = [], np.inf, None
    for ep in range(epochs):
        model.train(); perm = rng.permutation(itr); tl = 0.0; nb = 0
        for i in range(0, len(perm), bs):
            z, c, m, pp = batch(perm[i:i + bs]); loss = model.loss(z, c, m, pp)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl += loss.item(); nb += 1
        sched.step(); model.eval()
        with torch.no_grad():
            vl = float(np.mean([model.loss(*batch(iva[i:i + 8192])).item() for i in range(0, len(iva), 8192)]))
        hist.append({"epoch": ep, "train": tl / max(nb, 1), "val": vl}); print(f"epoch {ep:3d} train {tl / max(nb, 1):.4f} val {vl:.4f}", flush=True)
        if vl < best: best, best_state = vl, {kk: v.clone() for kk, v in model.state_dict().items()}
    model.load_state_dict(best_state); metrics["flow_val_curve"] = hist

    # closure on the test split
    ite = np.flatnonzero(te)
    Pn = heads["n_fit"].predict_proba(cond[ite]); nf_gen = (Pn.cumsum(1) > rng.random(len(Pn))[:, None]).argmax(1)
    pm = np.arange(N_MAX)[None, :] < nf_gen[:, None]
    Zg = np.concatenate([model.sample(torch.tensor(cond[ite[i:i + 8192]], device=device), torch.tensor(pm[i:i + 8192], device=device), part=partb(ite[i:i + 8192])).cpu().numpy() for i in range(0, len(ite), 8192)])
    Xg = snap_walls(pt.inverse(Zg, pm)); vr = valid[ite]
    percol = {}
    for j, (name, _, tf) in enumerate(FIELDS):
        r = Xr[..., j][vr]; f = Xg[..., j][pm]; r, f = r[np.isfinite(r)], f[np.isfinite(f)]
        if tf == "flag": percol[name] = {"mean_real": float(r.mean()), "mean_gen": float(f.mean())}
        else:
            from ..data.pfpset_tf import fwd
            rr, ff = fwd(r, tf), fwd(f, tf); rr, ff = rr[np.isfinite(rr)], ff[np.isfinite(ff)]
            percol[name] = sample_vs_real_1d(rr, ff) if len(rr) > 10 and len(ff) > 10 else {"w1": float("nan")}
    metrics["per_field"] = percol
    metrics["end_point"] = end_point_metrics(Xr, vr, Xg, pm, tcls[ite])
    ep = metrics["end_point"]
    print(f"end point (longest track, same test slices): |gen-real| median {ep['err_end']['median']:.1f} cm (straight line {ep['err_line_end']['median']:.1f}), "
          f">10 cm {ep['err_end']['frac_gt10']:.2f}; lateral real {ep['lateral_real']['median']:.1f} gen {ep['lateral_gen']['median']:.1f}; "
          f"end in FV real {ep['end_in_fv_real']:.4f} gen {ep['end_in_fv_gen']:.4f} (straight line {ep['line_end_in_fv_gen']:.4f}); "
          f"wall distance <-3/+-3/3-10/10-20/>20: real {np.round(ep['wall_hist_real'], 3).tolist()} gen {np.round(ep['wall_hist_gen'], 3).tolist()} line {np.round(ep['wall_hist_line_gen'], 3).tolist()}", flush=True)
    vtx = T.iloc[ite][["vtx_x", "vtx_y", "vtx_z"]].to_numpy(np.float64)
    (tw_r, ct_r), (tw_g, ct_g) = cc1p_selection(Xr, vr, vtx), cc1p_selection(Xg, pm, vtx); cl = tcls[ite]; sel = {}
    for k, name in enumerate(TRUTH_CLASSES):
        m = cl == k
        sel[name] = {"two_real": float(tw_r[m].mean()), "two_gen": float(tw_g[m].mean()),
                     "cont_given_two_real": float(ct_r[m].sum() / max(tw_r[m].sum(), 1)), "cont_given_two_gen": float(ct_g[m].sum() / max(tw_g[m].sum(), 1)),
                     "cont_given_two_both": [float(ct_r[m & tw_r & tw_g].mean()), float(ct_g[m & tw_r & tw_g].mean())]}
        b = sel[name]; print(f"  {name:13s} 2prong real {b['two_real']:.4f} gen {b['two_gen']:.4f}; contained/2prong real {b['cont_given_two_real']:.3f} gen {b['cont_given_two_gen']:.3f}"
                             f" (2prong in both: {b['cont_given_two_both'][0]:.3f} / {b['cont_given_two_both'][1]:.3f})", flush=True)
    metrics["cc1p_selection_by_class"] = sel
    for name, b in ep.get("by_class", {}).items():
        print(f"  {name:13s} n={b['n']:6d} end in FV real {b['end_in_fv_real']:.4f} gen {b['end_in_fv_gen']:.4f}; on wall real {b['on_wall_real']:.4f} gen {b['on_wall_gen']:.4f}", flush=True)
    cr, cg = cohpi_counts(Xr, vr), cohpi_counts(Xg, pm)
    two_r = (cr["n_prong"] == 2) & (cr["n_trk"] == 2) & (cr["n_proton"] == 0) & (cr["n_shower"] == 0)
    two_g = (cg["n_prong"] == 2) & (cg["n_trk"] == 2) & (cg["n_proton"] == 0) & (cg["n_shower"] == 0)
    metrics["cohpi_counts_test"] = {k: {"mean_real": float(cr[k].mean()), "mean_gen": float(cg[k].mean()),
                                        "dist_real": np.bincount(np.clip(cr[k], 0, 6), minlength=7).tolist(), "dist_gen": np.bincount(np.clip(cg[k], 0, 6), minlength=7).tolist()} for k in cr}
    metrics["cohpi_counts_test"]["two_track_rate"] = {"real": float(two_r.mean()), "gen": float(two_g.mean()), "n_test": int(len(ite))}
    # the slice table's own counts (what the topology head produces) for comparison
    tt = T.iloc[ite]; metrics["cohpi_counts_test"]["two_track_rate"]["slc_table"] = float(((tt.n_prong == 2) & (tt.n_trk == 2) & (tt.n_proton == 0) & (tt.n_shower == 0)).mean())
    print("n_fit real/gen mean:", n_fit[ite].mean().round(3), nf_gen.mean().round(3), " two-track rate real/gen/slc:", round(two_r.mean(), 4), round(two_g.mean(), 4), round(metrics["cohpi_counts_test"]["two_track_rate"]["slc_table"], 4), flush=True)
    print("W1:", {k: round(v["w1"], 3) for k, v in percol.items() if "w1" in v}, flush=True)
    show = ["dist_to_vertex", "trackScore", "len_excess", "rangeP_mu", "chi2_mu", "chi2_pro", "chi2_kaon", "shw_energy", "shw_dEdx", "dir_z", "start_z", "pida"]
    fig, axs = plots.plt.subplots(3, 4, figsize=(14, 9)); axs = axs.ravel()
    for ax, name in zip(axs, show):
        j = NAMES.index(name); tf = FIELDS[j][2]
        from ..data.pfpset_tf import fwd
        r, f = fwd(Xr[..., j][vr], tf), fwd(Xg[..., j][pm], tf); r, f = r[np.isfinite(r)], f[np.isfinite(f)]
        lo, hi = np.percentile(np.concatenate([r, f]), [0.5, 99.5]); plots.hist_compare(ax, r, f, np.linspace(lo, hi, 70), f"{tf}({name})" if tf != "lin" else name, (plots.REAL_LABEL, "set flow"))
    plots.save(fig, out / "figures" / "m4_marginals.png")
    fig, axs = plots.plt.subplots(1, 2, figsize=(8, 3.2))
    plots.bar_compare(axs[0], np.bincount(n_fit[ite], minlength=N_MAX + 1), np.bincount(nf_gen, minlength=N_MAX + 1), "fitted PFPs per slice")
    plots.bar_compare(axs[1], np.array(metrics["cohpi_counts_test"]["n_trk"]["dist_real"]), np.array(metrics["cohpi_counts_test"]["n_trk"]["dist_gen"]), "cohpi track candidates (recomputed from PFPs)")
    plots.save(fig, out / "figures" / "m4_counts.png")
    torch.save({"state": model.state_dict(), "dim": Zs.shape[-1], "cond_dim": cond.shape[1], "d_model": d_model, "n_layers": n_layers, "n_part_layers": n_part_layers, "geo": geo, "layout": LAYOUT, "pt_loc": pt.loc, "pt_scale": pt.scale,
                "std_m": std.m, "std_s": std.s, "bt_med": bt.med, "bt_mad": bt.mad, "topology_classes": topo.classes, "topology_other_pool": topo.other_pool,
                "n_fit_head": heads["n_fit"].net.state_dict(), "n_empty_head": heads["n_empty"].net.state_dict()}, out / "model.pt")
    metrics["runtime_s"] = time.time() - t0; (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    print(f"done in {metrics['runtime_s']:.0f} s -> {out}", flush=True)
    return metrics
