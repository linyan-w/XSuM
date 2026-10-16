"""M1 baselines on the XSuM production table: GBDT classifiers for Tier 0 (matched slice, cohpi selection)
and PFP multiplicity, MDNs for the vertex residual, the leading-track response and the cohpi two-track
response. Trained on the file-level train split, early-stopped on val, reported on test. Adapted from
opendata-sim2reco M1.
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from ..data.splits import split_by_level
from ..eval import plots
from ..eval.metrics import binary_metrics, calibration_by_bin, confusion, multiclass_metrics, sample_vs_real_1d
from ..models.mdn import MDN

N_PFP_MAX = 8  # classes 1..7 and "8+"


def gbdt(Xtr, ytr, seed=0, max_iter=500):
    clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                         l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                         n_iter_no_change=20, random_state=seed)
    clf.fit(Xtr, ytr)
    return clf


class Standardizer:
    def __init__(self, X):
        Xa = np.arcsinh(X)
        self.m, self.s = Xa.mean(0), Xa.std(0) + 1e-6

    def __call__(self, X):
        return ((np.arcsinh(X) - self.m) / self.s).astype(np.float32)


def train_mdn(Xtr, Ytr, Xva, Yva, n_components=8, epochs=60, bs=4096, lr=2e-3, device="cuda", seed=0, clip=25.0):
    torch.manual_seed(seed)
    ymean = np.median(Ytr, 0)
    ystd = 1.4826 * np.median(np.abs(Ytr - ymean), 0) + 1e-6
    model = MDN(Xtr.shape[1], Ytr.shape[1], n_components).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt = torch.tensor(Xtr, device=device)
    Yt = torch.tensor(np.clip((Ytr - ymean) / ystd, -clip, clip), dtype=torch.float32, device=device)
    Xv = torch.tensor(Xva, device=device)
    Yv = torch.tensor(np.clip((Yva - ymean) / ystd, -clip, clip), dtype=torch.float32, device=device)
    best, best_state, hist = np.inf, None, []
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            loss = model.nll(Xt[idx], Yt[idx]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step(); model.eval()
        with torch.no_grad():
            vl = np.mean([model.nll(Xv[i:i + 65536], Yv[i:i + 65536]).mean().item() for i in range(0, len(Xv), 65536)])
        hist.append(float(vl))
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.ymean, model.ystd = ymean, ystd
    return model, hist


def mdn_nll(model, X, Y, device="cuda", clip=25.0):
    with torch.no_grad():
        out = []
        for i in range(0, len(X), 65536):
            Xt = torch.tensor(X[i:i + 65536], device=device)
            Yt = torch.tensor(np.clip((Y[i:i + 65536] - model.ymean) / model.ystd, -clip, clip), dtype=torch.float32, device=device)
            out.append(model.nll(Xt, Yt).cpu().numpy())
    return float(np.concatenate(out).mean() + np.log(model.ystd).sum())


def mdn_sample(model, X, device="cuda"):
    with torch.no_grad():
        out = [model.sample(torch.tensor(X[i:i + 65536], device=device), 1)[:, 0].cpu().numpy() for i in range(0, len(X), 65536)]
    return np.concatenate(out) * model.ystd + model.ymean


def quantile_rows(x, real, fake, edges, min_n=50):
    rows = []
    for b in range(len(edges) - 1):
        s = (x >= edges[b]) & (x < edges[b + 1])
        if s.sum() > min_n:
            rows.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(s.sum()),
                         "real_q16_50_84": np.percentile(real[s], [16, 50, 84]).tolist(), "fake_q16_50_84": np.percentile(fake[s], [16, 50, 84]).tolist()})
    return rows


def band_panel(ax, rows, xlabel, ylabel, labels=(plots.REAL_LABEL, "MDN (K=8)"), logx=False):
    x = [(q["lo"] + q["hi"]) / 2 for q in rows]
    for k, (c, lab, key, st) in enumerate([(plots.PALETTE["real"], labels[0], "real_q16_50_84", "o-"), (plots.PALETTE["model"], labels[1], "fake_q16_50_84", "s--")]):
        ax.plot(x, [q[key][1] for q in rows], st, color=c, ms=4, label=lab)
        ax.fill_between(x, [q[key][0] for q in rows], [q[key][2] for q in rows], color=c, alpha=0.15)
    if logx: ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend(frameon=False)


def run(table_dir: str, out_dir: str, device: str = "cuda", epochs: int = 60, seed: int = 0, max_rows: int | None = None):
    out = pathlib.Path(out_dir); (out / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    F = pd.read_parquet(f"{table_dir}/features.parquet")
    T = pd.read_parquet(f"{table_dir}/targets.parquet")
    L = pd.read_parquet(f"{table_dir}/labels.parquet")
    if max_rows and len(F) > max_rows:
        keep = np.random.default_rng(seed).choice(len(F), max_rows, replace=False); keep.sort()
        F, T, L = F.iloc[keep], T.iloc[keep], L.iloc[keep]
    names = list(F.columns)
    X = F.to_numpy(np.float32)
    file_key = pd.MultiIndex.from_arrays([F.index.get_level_values("chunk"), F.index.get_level_values("__ntuple")])
    split = split_by_level(pd.MultiIndex.from_arrays([file_key.map(lambda t: t[0] * 100000 + t[1])], names=["file"]), level="file", seed=seed)
    tr, va, te = split["train"], split["val"], split["test"]
    matched = T.matched.to_numpy().astype(bool)
    fv = L.is_fv.to_numpy().astype(bool)
    metrics = {"n": int(len(F)), "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()), "features": names,
               "frac_matched": float(matched.mean()), "frac_selected": float(T.selected.mean())}
    print(f"table: {len(F)} neutrinos, {len(names)} features, matched {matched.mean():.3f}, selected {T.selected.mean():.5f}  ({time.time()-t0:.0f} s)", flush=True)

    def col(n): return X[:, names.index(n)]
    lead_p_mu, vtx_z, n_charged, sumke = col("lead_p_mu"), col("vtx_z"), col("n_charged"), col("sumke_visible")

    # ---- Tier 0 --------------------------------------------------------------------------------------------
    tier0 = {}
    y = matched.astype(int)
    clf = gbdt(X[tr], y[tr], seed); p = clf.predict_proba(X[te])[:, 1]
    tier0["matched"] = binary_metrics(y[te], p)
    tier0["matched"]["calibration"] = {
        "vtx_z_cm": calibration_by_bin(vtx_z[te], y[te], p, np.arange(-100, 601, 50)),
        "lead_p_mu_GeV": calibration_by_bin(lead_p_mu[te], y[te], p, np.array([0, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5])),
        "sumke_visible_GeV": calibration_by_bin(sumke[te], y[te], p, np.array([0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 4])),
        "n_charged": calibration_by_bin(n_charged[te], y[te], p, np.arange(-0.5, 8.5, 1))}
    fig, axs = plots.plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, (k, lab) in zip(axs, [("vtx_z_cm", "true vertex z [cm]"), ("lead_p_mu_GeV", "true muon momentum [GeV]"), ("sumke_visible_GeV", "visible true KE [GeV]"), ("n_charged", "charged primaries above threshold")]):
        plots.calibration_panel(ax, tier0["matched"]["calibration"][k], lab)
    axs[0].set_title("P(a slice is truth-matched to the neutrino), test split", loc="left", fontsize=10)
    plots.save(fig, out / "figures" / "m1_matched_calibration.png")
    print("tier0 matched:", {k: round(v, 4) for k, v in tier0["matched"].items() if not isinstance(v, dict)}, flush=True)

    ys = T.selected.to_numpy().astype(int)
    sig = L.is_signal_cohpi.to_numpy().astype(bool)
    if ys[tr].sum() < 20 or ys[te].sum() < 5:
        print(f"selected_cohpi: too few positives (train {int(ys[tr].sum())}, test {int(ys[te].sum())}), skipped", flush=True)
        ps = np.zeros(len(X)); tier0["selected_cohpi"] = {"skipped": True, "n_pos_train": int(ys[tr].sum())}
    else:
        clf = gbdt(X[tr], ys[tr], seed); ps = np.zeros(len(X)); ps[te] = clf.predict_proba(X[te])[:, 1]
        tier0["selected_cohpi"] = binary_metrics(ys[te], ps[te])
    tier0["selected_cohpi"]["signal_only"] = binary_metrics(ys[te & sig], ps[te & sig]) if (te & sig).sum() > 100 else None
    tier0["selected_cohpi"]["calibration"] = {
        "lead_p_mu_GeV_signal": calibration_by_bin(lead_p_mu[te & sig], ys[te & sig], ps[te & sig], np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2, 3])),
        "lead_p_pi_GeV_signal": calibration_by_bin(col("lead_p_pi")[te & sig], ys[te & sig], ps[te & sig], np.array([0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2, 2])),
        "nuint_categ": [{"categ": int(c), "n": int(((L.nuint_categ.to_numpy() == c) & te).sum()), "observed": float(ys[te & (L.nuint_categ.to_numpy() == c)].mean()),
                         "predicted": float(ps[te & (L.nuint_categ.to_numpy() == c)].mean())} for c in sorted(L.nuint_categ.unique()) if (te & (L.nuint_categ.to_numpy() == c)).sum() > 0]}
    fig, axs = plots.plt.subplots(1, 2, figsize=(7.5, 3.2))
    plots.calibration_panel(axs[0], tier0["selected_cohpi"]["calibration"]["lead_p_mu_GeV_signal"], "true muon momentum [GeV], signal")
    plots.calibration_panel(axs[1], tier0["selected_cohpi"]["calibration"]["lead_p_pi_GeV_signal"], "true pion momentum [GeV], signal")
    axs[0].set_title("P(cohpi selection | truth), test split", loc="left", fontsize=10)
    for ax in axs: ax.set_ylim(0, max(0.05, ax.get_ylim()[1]))
    plots.save(fig, out / "figures" / "m1_selected_calibration.png")
    metrics["tier0"] = tier0
    print("tier0 selected:", {k: round(v, 5) for k, v in tier0["selected_cohpi"].items() if isinstance(v, float)}, flush=True)

    # ---- PFP multiplicity of the matched slice ------------------------------------------------------------
    npfp = np.clip(np.nan_to_num(T.n_pfp.to_numpy(), nan=0), 1, N_PFP_MAX).astype(int) - 1
    clf = gbdt(X[tr & matched], npfp[tr & matched], seed)
    Pn = np.zeros((int((te & matched).sum()), N_PFP_MAX)); Pn[:, clf.classes_] = clf.predict_proba(X[te & matched])
    mm = multiclass_metrics(npfp[te & matched], Pn)
    rng = np.random.default_rng(seed)
    samp = (Pn.cumsum(1) > rng.random(len(Pn))[:, None]).argmax(1)
    K = N_PFP_MAX
    mm["confusion_true_vs_sampled"] = confusion(npfp[te & matched], samp, K).tolist()
    mm["marginal_true"] = np.bincount(npfp[te & matched], minlength=K).tolist()
    mm["marginal_sampled"] = np.bincount(samp, minlength=K).tolist()
    nvis = n_charged[te & matched]
    mm["mean_npfp_by_ncharged"] = [{"n_charged": int(k), "n": int((nvis == k).sum()), "true": float(npfp[te & matched][nvis == k].mean() + 1), "sampled": float(samp[nvis == k].mean() + 1)}
                                   for k in range(0, 8) if (nvis == k).sum() > 50]
    metrics["multiplicity"] = mm
    fig, axs = plots.plt.subplots(1, 2, figsize=(7.5, 3.2))
    plots.bar_compare(axs[0], np.array(mm["marginal_true"]), np.array(mm["marginal_sampled"]), "PFPs in the matched slice (8 = 8+), classes 1..8")
    r = mm["mean_npfp_by_ncharged"]
    axs[1].plot([q["n_charged"] for q in r], [q["true"] for q in r], "o-", color=plots.PALETTE["real"], ms=4, label=plots.REAL_LABEL)
    axs[1].plot([q["n_charged"] for q in r], [q["sampled"] for q in r], "s--", color=plots.PALETTE["model"], ms=4, label="GBDT sampled")
    axs[1].set_xlabel("charged primaries above threshold"); axs[1].set_ylabel("mean PFPs"); axs[1].legend(frameon=False)
    plots.save(fig, out / "figures" / "m1_multiplicity.png")
    print("multiplicity:", {k: round(v, 4) for k, v in mm.items() if isinstance(v, float)}, flush=True)

    # ---- MDNs ----------------------------------------------------------------------------------------------
    std = Standardizer(X[tr]); Xs = std(X)
    mdn_res = {}

    def fit_report(tag, mask, Y, ynames, figspec):
        good = mask & np.isfinite(Y).all(1)
        res = {}
        for K_, name in ((1, "gauss"), (8, "mdn8")):
            model, hist = train_mdn(Xs[tr & good], Y[tr & good], Xs[va & good], Y[va & good], K_, epochs, device=device, seed=seed)
            r = {"test_nll": mdn_nll(model, Xs[te & good], Y[te & good], device), "val_curve": hist, "n_train": int((tr & good).sum()), "n_test": int((te & good).sum())}
            if name == "mdn8":
                S = mdn_sample(model, Xs[te & good], device); Yte = Y[te & good]
                for j, nm in enumerate(ynames):
                    r[nm] = sample_vs_real_1d(Yte[:, j], S[:, j])
                r["by_bin"] = {}
                for nm, (xcol, edges, j, logx) in figspec.items():
                    r["by_bin"][nm] = quantile_rows(X[te & good, names.index(xcol)], Yte[:, j], S[:, j], edges)
                fig, axs = plots.plt.subplots(1, len(ynames) + len(figspec), figsize=(3.6 * (len(ynames) + len(figspec)), 3.2))
                for j, nm in enumerate(ynames):
                    lo, hi = np.percentile(Yte[:, j], [0.5, 99.5])
                    plots.hist_compare(axs[j], Yte[:, j], S[:, j], np.linspace(lo, hi, 80), nm, (plots.REAL_LABEL, "MDN (K=8)"))
                for k_, (nm, (xcol, edges, j, logx)) in enumerate(figspec.items()):
                    band_panel(axs[len(ynames) + k_], r["by_bin"][nm], xcol, ynames[j] + " median, 16-84%", logx=logx)
                plots.save(fig, out / "figures" / f"m1_{tag}.png")
            res[name] = r
        mdn_res[tag] = res
        print(f"{tag}: NLL gauss {res['gauss']['test_nll']:.4f} mdn8 {res['mdn8']['test_nll']:.4f}; W1 " +
              ", ".join(f"{nm} {res['mdn8'][nm]['w1']:.4f}" for nm in ynames), flush=True)

    # vertex residual of the matched slice, FV neutrinos
    Yv = np.stack([T.vtx_x.to_numpy() - col("vtx_x"), T.vtx_y.to_numpy() - col("vtx_y"), T.vtx_z.to_numpy() - col("vtx_z")], 1).astype(np.float64)
    fit_report("vertex", matched & fv, Yv, ["dvtx_x [cm]", "dvtx_y [cm]", "dvtx_z [cm]"],
               {"vs_sumke": ("sumke_visible", np.array([0, 0.05, 0.1, 0.2, 0.4, 0.7, 1, 2, 5]), 2, True)})
    # leading-track response: longest track direction and length vs the leading charged primary
    lead_cosz = np.where(col("n_mu") > 0, col("lead_cosz_mu"), np.where(col("n_pi") > 0, col("lead_cosz_pi"), col("lead_cosz_p")))
    Yl = np.stack([T.long_dirz.to_numpy() - lead_cosz, np.log(np.clip(T.long_trk_len.to_numpy(), 0.1, None))], 1).astype(np.float64)
    fit_report("leadtrack", matched & fv & (n_charged > 0), Yl, ["long_dirz - lead cos theta", "log(long track length [cm])"],
               {"vs_lead_p_mu": ("lead_p_mu", np.array([0, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5]), 1, True)})
    # cohpi two-track response on 1mu1pi events that pass the two-track topology
    two = matched & fv & (col("n_mu") == 1) & (col("n_pi") == 1) & (np.nan_to_num(T.pass_2trk.to_numpy(), nan=0) > 0)
    Y2 = np.stack([np.log(T.range_p_mu.to_numpy() / np.clip(col("lead_p_mu"), 1e-3, None)), np.log(T.range_p_cpi.to_numpy() / np.clip(col("lead_p_pi"), 1e-3, None)),
                   T.long_dirz.to_numpy() - col("lead_cosz_mu")], 1).astype(np.float64)
    if two.sum() > 2000:
        fit_report("twotrack", two, Y2, ["log(range_p_mu / p_mu_true)", "log(range_p_cpi / p_pi_true)", "long_dirz - cos theta_mu"],
                   {"vs_p_mu": ("lead_p_mu", np.array([0, 0.2, 0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 3]), 0, False)})
    metrics["mdn"] = mdn_res
    metrics["runtime_s"] = time.time() - t0
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    print(f"done in {metrics['runtime_s']:.0f} s -> {out}", flush=True)
    return metrics
