#!/usr/bin/env python
"""Correlation closure for the slice surrogate: real (Pandora) vs generated slices of the same test neutrinos.

usage: python scripts/plot_correlations.py reports/m2_official_v3 data/prod_v10_14_02_03_official/table_matched [--out reports/m2_official_v3/figures]
Figures: m2_corr_matrix.png (Spearman matrices real / generated / difference), m2_corr_pairs.png (reco-reco joint
densities), m2_corr_truth.png (reco vs truth joint densities). Numbers go to correlations.json.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.colors import LinearSegmentedColormap, LogNorm

INK, MUTED = "#1f1f1f", "#6b6b6b"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
DIVERGE = LinearSegmentedColormap.from_list("div", ["#2a6fb0", "#f2f2f2", "#c4582a"])   # blue - neutral gray - orange
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f7fb", "#9ec3e6", "#2a6fb0", "#0b2f5a"])  # one hue, light -> dark

VARS = [("vtx_z", "lin"), ("total_pe", "log"), ("nu_score", "lin"), ("bcfm_score", "lin"), ("n_pfp", "lin"), ("n_trk", "lin"),
        ("long_trk_len", "log"), ("long_dirz", "lin"), ("long_trk_chi2_mu_new", "log1p"), ("long_trk_chi2_pro_new", "log1p"),
        ("short_trk_len", "log"), ("short_dirz", "lin"), ("short_trk_chi2_pro_new", "log1p"),
        ("range_p_mu", "log"), ("range_p_cpi", "log"), ("opening_angle", "lin"), ("reco_t", "log"),
        ("sum_integ_1cm", "log"), ("sum_integ_4cm", "log")]


def tf(x, kind):
    x = np.asarray(x, dtype=float)
    if kind == "log": return np.log(np.where(x > 0, x, np.nan))
    if kind == "log1p": return np.log1p(np.where(x >= 0, x, np.nan))
    return x


def spearman(df):
    r = df.rank(); return r.corr(min_periods=2000)


def hist2(ax, x, y, bins, title, norm):
    ok = np.isfinite(x) & np.isfinite(y)
    h = ax.hist2d(x[ok], y[ok], bins=bins, cmap=SEQ, norm=norm, rasterized=True)
    ax.set_title(title, fontsize=9, loc="left", color=INK)
    return h


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("m2_dir"); ap.add_argument("table"); ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=250_000); a = ap.parse_args()
    out = a.out or f"{a.m2_dir}/figures"; os.makedirs(out, exist_ok=True)
    R = pd.read_parquet(f"{a.m2_dir}/real_test.parquet"); G = pd.read_parquet(f"{a.m2_dir}/generated_test.parquet")
    rng = np.random.default_rng(0); sel = np.sort(rng.choice(len(R), min(a.n, len(R)), replace=False)); R, G = R.iloc[sel], G.iloc[sel]
    names = [v for v, _ in VARS]
    Rt = pd.DataFrame({v: tf(R[v], k) for v, k in VARS}, index=R.index); Gt = pd.DataFrame({v: tf(G[v], k) for v, k in VARS}, index=G.index)
    cr, cg = spearman(Rt), spearman(Gt); d = cg - cr
    iu = np.triu_indices(len(names), 1); dv = d.values[iu]; dv = dv[np.isfinite(dv)]
    summary = {"n_slices": int(len(R)), "n_pairs": int(len(dv)), "mean_abs_delta_rho": float(np.mean(np.abs(dv))), "max_abs_delta_rho": float(np.max(np.abs(dv))),
               "frac_pairs_abs_delta_lt_0.05": float(np.mean(np.abs(dv) < 0.05))}
    worst = sorted(((abs(d.values[i, j]), names[i], names[j], cr.values[i, j], cg.values[i, j]) for i, j in zip(*iu) if np.isfinite(d.values[i, j])), reverse=True)[:8]
    summary["worst_pairs"] = [{"a": a_, "b": b_, "rho_real": float(x), "rho_gen": float(y)} for _, a_, b_, x, y in worst]

    # ---- figure 1: correlation matrices -----------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(19, 6.2), gridspec_kw={"wspace": 0.55})
    for ax, M, title, vmax in ((axs[0], cr, "Pandora (cafpyana)", 1), (axs[1], cg, "XSuM generated", 1), (axs[2], d, "generated - Pandora", 0.2)):
        im = ax.imshow(M.values, cmap=DIVERGE, vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=7); ax.set_yticklabels(names, fontsize=7)
        ax.set_title(f"{title}: Spearman $\\rho$", fontsize=10, loc="left", color=INK)
        for s in ax.spines.values(): s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
    fig.suptitle(f"Slice-variable correlations on {len(R):,} test slices; mean |$\\Delta\\rho$| = {summary['mean_abs_delta_rho']:.3f}, "
                 f"{100*summary['frac_pairs_abs_delta_lt_0.05']:.0f}% of pairs within 0.05", x=0.02, ha="left", fontsize=10, color=INK)
    fig.savefig(f"{out}/m2_corr_matrix.png", bbox_inches="tight"); plt.close(fig)

    # ---- figure 2: reco-reco joint densities ------------------------------------------------------------------
    two = np.isfinite(R.reco_t.to_numpy()) & np.isfinite(G.reco_t.to_numpy())   # two-track block present in both
    pairs = [("long_trk_len", "log", "range_p_mu", "log", None, "longest track: length vs range momentum (two-track)"),
             ("long_dirz", "lin", "short_dirz", "lin", None, "cos theta of the two tracks (two-track)"),
             ("sum_integ_1cm", "log", "sum_integ_4cm", "log", "all", "vertex charge, 1 cm vs 4 cm sphere"),
             ("long_trk_len", "log", "long_trk_chi2_pro_new", "log1p", "all", "longest track: length vs chi2 proton"),
             ("vtx_z", "lin", "total_pe", "log", "all", "vertex z vs total light"),
             ("opening_angle", "lin", "reco_t", "log", None, "opening angle vs |t| (two-track)")]
    fig, axs = plt.subplots(len(pairs), 2, figsize=(8.4, 3.0 * len(pairs)), gridspec_kw={"hspace": 0.55, "wspace": 0.3})
    for row, (xv, xk, yv, yk, scope, label) in enumerate(pairs):
        m = np.ones(len(R), bool) if scope == "all" else two
        xr, yr, xg, yg = tf(R[xv], xk)[m], tf(R[yv], yk)[m], tf(G[xv], xk)[m], tf(G[yv], yk)[m]
        both = np.concatenate([xr, xg]); bothy = np.concatenate([yr, yg])
        bx = np.linspace(*np.nanpercentile(both, [0.5, 99.5]), 60); by = np.linspace(*np.nanpercentile(bothy, [0.5, 99.5]), 60)
        rr = pd.Series(xr).corr(pd.Series(yr), method="spearman"); rg = pd.Series(xg).corr(pd.Series(yg), method="spearman")
        norm = LogNorm(vmin=1, vmax=max(1, np.histogram2d(xr[np.isfinite(xr) & np.isfinite(yr)], yr[np.isfinite(xr) & np.isfinite(yr)], bins=[bx, by])[0].max()))
        hist2(axs[row, 0], xr, yr, [bx, by], f"Pandora   $\\rho$={rr:.3f}", norm); hist2(axs[row, 1], xg, yg, [bx, by], f"XSuM   $\\rho$={rg:.3f}", norm)
        for ax in axs[row]:
            ax.set_xlabel(f"{xk}({xv})" if xk != "lin" else xv); ax.set_ylabel(f"{yk}({yv})" if yk != "lin" else yv)
        axs[row, 0].text(0, 1.18, label, transform=axs[row, 0].transAxes, fontsize=10, color=INK, fontweight="bold")
    fig.savefig(f"{out}/m2_corr_pairs.png", bbox_inches="tight"); plt.close(fig)

    # ---- figure 3: reco vs truth (the conditional the surrogate must learn) -----------------------------------
    pf = pq.ParquetFile(f"{a.table}/features.parquet")
    idx_cols = list(R.index.names)
    Ft = pf.read(columns=idx_cols + ["vtx_z", "lead_p_mu", "lead_cosz_mu", "n_mu", "n_pi", "sumke_visible"]).to_pandas()
    if not isinstance(Ft.index, pd.MultiIndex): Ft = Ft.set_index(idx_cols)
    F = Ft.reindex(R.index); print("features found for", int(F.vtx_z.notna().sum()), "of", len(F), "test slices", flush=True)
    tpairs = [("lead_p_mu", "log", "range_p_mu", "log", "range momentum vs true muon momentum (1 mu, two-track)", (F.n_mu.to_numpy() == 1) & two),
              ("lead_cosz_mu", "lin", "long_dirz", "lin", "longest-track cos theta vs true muon cos theta (1 mu)", F.n_mu.to_numpy() == 1),
              ("vtx_z", "lin", "vtx_z", "lin", "reconstructed vs true vertex z", np.ones(len(R), bool)),
              ("sumke_visible", "log", "sum_integ_4cm", "log", "4 cm vertex charge vs true visible kinetic energy", np.ones(len(R), bool))]
    fig, axs = plt.subplots(len(tpairs), 2, figsize=(8.4, 3.0 * len(tpairs)), gridspec_kw={"hspace": 0.55, "wspace": 0.3})
    tsum = {}
    for row, (xv, xk, yv, yk, label, m) in enumerate(tpairs):
        x = tf(F[xv], xk)[m]; yr, yg = tf(R[yv], yk)[m], tf(G[yv], yk)[m]
        bx = np.linspace(*np.nanpercentile(x, [0.5, 99.5]), 60); by = np.linspace(*np.nanpercentile(np.concatenate([yr, yg]), [0.5, 99.5]), 60)
        ok = np.isfinite(x) & np.isfinite(yr)
        norm = LogNorm(vmin=1, vmax=max(1, np.histogram2d(x[ok], yr[ok], bins=[bx, by])[0].max()))
        rr = pd.Series(x).corr(pd.Series(yr), method="spearman"); rg = pd.Series(x).corr(pd.Series(yg), method="spearman")
        tsum[f"{yv}|{xv}"] = {"rho_real": float(rr), "rho_gen": float(rg)}
        hist2(axs[row, 0], x, yr, [bx, by], f"Pandora   $\\rho$={rr:.3f}", norm); hist2(axs[row, 1], x, yg, [bx, by], f"XSuM   $\\rho$={rg:.3f}", norm)
        for ax in axs[row]:
            ax.set_xlabel(("true " + (f"{xk}({xv})" if xk != "lin" else xv))); ax.set_ylabel(f"{yk}({yv})" if yk != "lin" else yv)
        axs[row, 0].text(0, 1.18, label, transform=axs[row, 0].transAxes, fontsize=10, color=INK, fontweight="bold")
    fig.savefig(f"{out}/m2_corr_truth.png", bbox_inches="tight"); plt.close(fig)
    summary["truth_reco"] = tsum
    json.dump(summary, open(f"{a.m2_dir}/correlations.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "worst_pairs"}, indent=1)); print("worst:", summary["worst_pairs"][:4])


if __name__ == "__main__":
    main()
