"""M2: conditional flow-matching surrogate of the complete `slc` reconstruction block.

Per matched slice the block is split into groups that exist or not (short track, two-track topology, charge
spheres); group presence and the two containment flags are modelled with GBDT classifiers on the truth
features, the continuous variables with one flow-matching model over the standardised block, masked per
event to the groups that exist (sim2reco dim_mask). Sampling: presence -> flow -> inverse transforms ->
a DataFrame with the production column names, on which the analysis cuts can be recomputed.
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
from ..eval.metrics import binary_metrics, sample_vs_real_1d
from ..models.flow import FlowMatcher
from .baselines import Standardizer, gbdt

# (column, transform, group). Transforms: lin, log (x>0), log1p (x>=0), atanh (x in [-1,1]).
SPEC = [
    ("vtx_x", "lin", "base"), ("vtx_y", "lin", "base"), ("vtx_z", "lin", "base"),
    ("nu_score", "lin", "base"), ("bcfm_score", "lin", "base"), ("total_pe", "log", "base"),
    ("long_dirx", "atanh", "base"), ("long_diry", "atanh", "base"), ("long_dirz", "atanh", "base"), ("long_trk_len", "log", "base"),
    ("long_trk_pid", "lin", "base"), ("long_trk_chi2_mu", "log1p", "base"), ("long_trk_chi2_pro", "log1p", "base"),
    ("long_trk_chi2_mu_new", "log1p", "base"), ("long_trk_chi2_pro_new", "log1p", "base"),
    ("short_dirx", "atanh", "short"), ("short_diry", "atanh", "short"), ("short_dirz", "atanh", "short"), ("short_trk_len", "log", "short"),
    ("short_trk_pid", "lin", "short"), ("short_trk_chi2_mu", "log1p", "short"), ("short_trk_chi2_pro", "log1p", "short"),
    ("short_trk_chi2_mu_new", "log1p", "short"), ("short_trk_chi2_pro_new", "log1p", "short"),
    ("reco_t", "log", "two"), ("opening_angle", "atanh", "two"), ("range_p_mu", "log", "two"), ("range_p_cpi", "log", "two"),
    ("muon_start_x", "lin", "two"), ("muon_start_y", "lin", "two"), ("muon_start_z", "lin", "two"),
    ("muon_end_x", "lin", "two"), ("muon_end_y", "lin", "two"), ("muon_end_z", "lin", "two"),
    ("cpi_start_x", "lin", "two"), ("cpi_start_y", "lin", "two"), ("cpi_start_z", "lin", "two"),
    ("cpi_end_x", "lin", "two"), ("cpi_end_y", "lin", "two"), ("cpi_end_z", "lin", "two"),
    ("sum_integ_1cm", "log", "s1"), ("sum_integ_2cm", "log", "s2"), ("sum_integ_3cm", "log", "s3"), ("sum_integ_4cm", "log", "s4"),
]
GROUPS = ["short", "two", "s1", "s2", "s3", "s4"]
FLAGS = ["is_muon_contained", "is_cpi_contained"]   # Bernoulli, defined on the two-track group
COUNT_COLS = set()
TOPO_COLS = ["n_pfp", "n_prong", "n_trk", "n_proton", "n_shower"]   # joint categorical head, top patterns + "other"
N_TOPO = 120


EDGE = 1 - 1e-5


def fwd(x, tf):
    x = np.asarray(x, dtype=np.float64)
    if tf == "log": return np.log(np.where(x > 0, x, np.nan))
    if tf == "log1p": return np.log1p(np.where(x >= 0, x, np.nan))
    if tf == "atanh": return np.arctanh(np.clip(x, -EDGE, EDGE))
    return x


def inv(y, tf):
    if tf == "log": return np.exp(y)
    if tf == "log1p": return np.maximum(np.expm1(y), 0.0)
    if tf == "atanh": return np.tanh(y)
    return y


# point masses at detector / algorithm boundaries: (column, membership test, value given the truth features)
SPIKES = {
    "vtx_z": (lambda v: np.abs(v) < 0.5, lambda F: np.zeros(len(F))),
    "vtx_x": (lambda v: np.abs(v) > 199.5, lambda F: 200.0 * np.sign(F["vtx_x"].to_numpy())),
    "vtx_y": (lambda v: np.abs(v) > 199.5, lambda F: 200.0 * np.sign(F["vtx_y"].to_numpy())),
    "opening_angle": (lambda v: v <= -0.999, lambda F: -np.ones(len(F))),
}


class BlockTransform:
    """DataFrame of slice columns <-> standardised matrix with NaN mask; robust per-column scaling."""
    def __init__(self, spec=SPEC):
        self.spec = spec; self.cols = [c for c, _, _ in spec]; self.groups = np.array([g for _, _, g in spec])

    def matrix(self, df):
        Y = np.stack([fwd(df[c].to_numpy(), tf) for c, tf, _ in self.spec], 1)
        Y[:, self.cols.index("bcfm_score")][df["bcfm_score"].to_numpy() < -100] = np.nan   # -9999 sentinel: no flash match
        return Y

    def fit(self, Y):
        self.med = np.nanmedian(Y, 0); self.mad = 1.4826 * np.nanmedian(np.abs(Y - self.med), 0) + 1e-6
        return self

    def standardise(self, Y, clip=12.0):
        Z = (Y - self.med) / self.mad
        M = np.isfinite(Z)
        return np.clip(np.nan_to_num(Z, nan=0.0), -clip, clip).astype(np.float32), M.astype(np.float32)

    def frame(self, Z, M, index):
        Y = Z * self.mad + self.med
        out = {}
        for j, (c, tf, _) in enumerate(self.spec):
            v = inv(Y[:, j], tf)
            out[c] = np.where(M[:, j] > 0, v, np.nan)
        df = pd.DataFrame(out, index=index)
        for t in ("long", "short"):
            n = np.sqrt(df[f"{t}_dirx"]**2 + df[f"{t}_diry"]**2 + df[f"{t}_dirz"]**2)
            for a in ("x", "y", "z"):
                df[f"{t}_dir{a}"] = df[f"{t}_dir{a}"] / n.where(n > 0, 1.0)
        return df


class Topology:
    """Joint categorical over (n_pfp, n_prong, n_trk, n_proton, n_shower): the N_TOPO most frequent tuples + other."""
    def __init__(self, T, train_mask, n=N_TOPO):
        tup = T.loc[train_mask, TOPO_COLS].fillna(-1).astype(int).apply(tuple, axis=1)
        self.classes = list(tup.value_counts().index[:n])
        self.lookup = {t: i for i, t in enumerate(self.classes)}
        self.other = len(self.classes)
        self.other_pool = np.array([t for t in tup if t not in self.lookup], dtype=float)   # rare tuples, sampled when "other"
        if len(self.other_pool) == 0: self.other_pool = np.array([self.classes[-1]], dtype=float)

    def encode(self, T):
        tup = T[TOPO_COLS].fillna(-1).astype(int).apply(tuple, axis=1)
        return tup.map(self.lookup).fillna(self.other).astype(int).to_numpy()

    def onehot(self, k):
        H = np.zeros((len(k), self.other + 1), dtype=np.float32); H[np.arange(len(k)), k] = 1.0; return H

    def decode(self, k, rng=None):
        rng = rng or np.random.default_rng(0)
        arr = np.array(self.classes + [(-1,) * len(TOPO_COLS)], dtype=float)
        out = arr[k]
        oth = k == self.other
        out[oth] = self.other_pool[rng.integers(0, len(self.other_pool), int(oth.sum()))]
        return pd.DataFrame(out, columns=TOPO_COLS)


def presence(T, bt):
    """Which groups exist per slice, from the real table."""
    P = {"short": np.isfinite(T["short_trk_len"].to_numpy()), "two": np.isfinite(T["reco_t"].to_numpy())}
    for r in (1, 2, 3, 4):
        P[f"s{r}"] = np.isfinite(T[f"sum_integ_{r}cm"].to_numpy()) & (T[f"sum_integ_{r}cm"].to_numpy() > 0)
    return P


def mask_from_presence(P, bt):
    M = np.ones((len(next(iter(P.values()))), len(bt.cols)), dtype=np.float32)
    for g in GROUPS:
        M[:, bt.groups == g] = P[g][:, None]
    return M


def cohpi_cutflow(df):
    """The cohpi selection recomputed from slice columns (make_xsumdf flags), as a dict stage -> count."""
    c = {}
    s = pd.Series(True, index=df.index)
    s &= (df.total_pe > 1500.) & (df.bcfm_score > 0.05); c["pe_bcfm"] = int(s.sum())
    s &= (df.n_prong == 2) & (df.n_trk == 2) & (df.n_proton == 0) & (df.n_shower == 0); c["2trk"] = int(s.sum())
    s &= (df.long_dirz > 0.7) & (df.short_dirz > 0.5) & (df.opening_angle > 0.5); c["dir"] = int(s.sum())
    s &= (df.long_trk_chi2_pro_new > 100.) & (df.short_trk_chi2_pro_new > 75.); c["chi2"] = int(s.sum())
    s &= (df.sum_integ_1cm.fillna(0) < 4000) & (df.sum_integ_2cm.fillna(0) < 7000) & (df.sum_integ_3cm.fillna(0) < 10000) & (df.sum_integ_4cm.fillna(0) < 13000)
    s &= (df.sum_integ_3cm / (df.sum_integ_4cm - df.sum_integ_3cm)).fillna(0) < 6.; c["sphere"] = int(s.sum())
    s &= (df.muon_start_x.abs() > 10) & (df.muon_end_x.abs() > 10) & (df.muon_start_x * df.muon_end_x > 0) & (df.cpi_start_x.abs() > 10) & (df.cpi_end_x.abs() > 10) & (df.cpi_start_x * df.cpi_end_x > 0)
    c["cathode"] = int(s.sum()); c["sr"] = int((s & (df.reco_t < 0.04)).sum())
    return c


class MLPClassifier:
    """Softmax MLP on the standardised features; fast replacement for a many-class GBDT. sklearn-like API."""
    def __init__(self, n_in, n_classes, hidden=256, n_layers=3, device="cuda", epochs=15, bs=8192, lr=2e-3, seed=0):
        torch.manual_seed(seed)
        layers, d = [], n_in
        for _ in range(n_layers):
            layers += [torch.nn.Linear(d, hidden), torch.nn.SiLU()]; d = hidden
        self.net = torch.nn.Sequential(*layers, torch.nn.Linear(hidden, n_classes)).to(device)
        self.device, self.epochs, self.bs, self.lr = device, epochs, bs, lr
        self.classes_ = np.arange(n_classes)

    def fit(self, X, y, Xv=None, yv=None):
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device); yt = torch.tensor(y, dtype=torch.long, device=self.device)
        best, best_state = np.inf, None
        for ep in range(self.epochs):
            self.net.train(); perm = torch.randperm(len(Xt), device=self.device)
            for i in range(0, len(Xt), self.bs):
                idx = perm[i:i + self.bs]
                loss = torch.nn.functional.cross_entropy(self.net(Xt[idx]), yt[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            if Xv is not None:
                vl = -np.log(np.clip(self.predict_proba(Xv)[np.arange(len(yv)), yv], 1e-9, None)).mean()
                if vl < best: best, best_state = vl, {k: v.clone() for k, v in self.net.state_dict().items()}
        if best_state is not None: self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        self.net.eval(); out = []
        for i in range(0, len(X), 65536):
            out.append(torch.softmax(self.net(torch.tensor(X[i:i + 65536], dtype=torch.float32, device=self.device)), -1).cpu().numpy())
        return np.concatenate(out)


def train_flow(Ztr, Mtr, Ctr, Zva, Mva, Cva, epochs, device, hidden=512, n_layers=5, bs=4096, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    model = FlowMatcher(Ztr.shape[1], Ctr.shape[1], hidden, n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    t = lambda a: torch.tensor(a, device=device)
    Zt, Mt, Ct, Zv, Mv, Cv = t(Ztr), t(Mtr), t(Ctr), t(Zva), t(Mva), t(Cva)
    best, best_state, hist = np.inf, None, []
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Zt), device=device); tl = 0.0; nb = 0
        for i in range(0, len(Zt), bs):
            idx = perm[i:i + bs]
            loss = model.loss(Zt[idx], Ct[idx], Mt[idx]).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tl += loss.item(); nb += 1
        sched.step(); model.eval()
        with torch.no_grad():
            vl = float(np.mean([model.loss(Zv[i:i + 65536], Cv[i:i + 65536], Mv[i:i + 65536]).mean().item() for i in range(0, len(Zv), 65536)]))
        hist.append({"epoch": ep, "train": tl / max(nb, 1), "val": vl})
        print(f"epoch {ep:3d} train {tl / max(nb, 1):.4f} val {vl:.4f}", flush=True)
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, hist


@torch.no_grad()
def sample_flow(model, C, M, device, n_steps=64, bs=65536):
    out = []
    for i in range(0, len(C), bs):
        out.append(model.sample(torch.tensor(C[i:i + bs], device=device), n_steps=n_steps, dim_mask=torch.tensor(M[i:i + bs], device=device)).cpu().numpy())
    return np.concatenate(out)


def run(table_dir, out_dir, device="cuda", epochs=30, seed=0, max_rows=None, hidden=512, n_layers=5):
    out = pathlib.Path(out_dir); (out / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    F = pd.read_parquet(f"{table_dir}/features.parquet"); T = pd.read_parquet(f"{table_dir}/targets.parquet"); L = pd.read_parquet(f"{table_dir}/labels.parquet")
    if max_rows and len(F) > max_rows:
        keep = np.sort(np.random.default_rng(seed).choice(len(F), max_rows, replace=False)); F, T, L = F.iloc[keep], T.iloc[keep], L.iloc[keep]
    names = list(F.columns); X = F.to_numpy(np.float32)
    file_id = F.index.get_level_values("chunk").to_numpy() * 100000 + F.index.get_level_values("__ntuple").to_numpy()
    split = split_by_level(pd.MultiIndex.from_arrays([file_id], names=["file"]), level="file", seed=seed)
    tr, va, te = split["train"], split["val"], split["test"]
    matched = T.matched.to_numpy().astype(bool)
    std = Standardizer(X[tr]); C = std(X)
    metrics = {"n": int(len(F)), "n_matched": int(matched.sum()), "epochs": epochs, "hidden": hidden, "n_layers": n_layers}
    print(f"table {len(F)} neutrinos, matched {matched.sum()}  ({time.time()-t0:.0f} s)", flush=True)

    # ---- slice cardinality and group presence (GBDT) ---------------------------------------------------------
    heads = {}
    ncl = np.clip(T.n_matched_slices.to_numpy(), 0, 2)
    heads["n_matched_slices"] = gbdt(X[tr], ncl[tr], seed)
    Pn = np.zeros((int(te.sum()), 3)); Pn[:, heads["n_matched_slices"].classes_] = heads["n_matched_slices"].predict_proba(X[te])
    metrics["n_matched_slices"] = {"marginal_true": np.bincount(ncl[te], minlength=3).tolist(), "mean_prob": Pn.mean(0).tolist(),
                                   "logloss": float(-np.log(np.clip(Pn[np.arange(len(Pn)), ncl[te]], 1e-9, None)).mean())}
    # topology: joint categorical over the PFP counts, sampled first and fed to every other component
    topo = Topology(T, tr & matched); ktrue = topo.encode(T)
    heads["topology"] = MLPClassifier(C.shape[1], topo.other + 1, device=device, seed=seed).fit(C[tr & matched], ktrue[tr & matched], C[va & matched], ktrue[va & matched])
    Pt = heads["topology"].predict_proba(C[te & matched])
    metrics["topology"] = {"n_classes": topo.other + 1, "frac_other_train": float((ktrue[tr & matched] == topo.other).mean()),
                           "logloss": float(-np.log(np.clip(Pt[np.arange(len(Pt)), ktrue[te & matched]], 1e-9, None)).mean()),
                           "logloss_marginal": float(-np.log(np.clip(np.bincount(ktrue[tr & matched], minlength=topo.other + 1)[ktrue[te & matched]] / (tr & matched).sum(), 1e-9, None)).mean()),
                           "accuracy": float((Pt.argmax(1) == ktrue[te & matched]).mean())}
    print("topology:", {k: round(v, 4) for k, v in metrics["topology"].items()}, flush=True)
    Hk = topo.onehot(ktrue)                      # real topology one-hot (training / teacher forcing)
    XH = np.concatenate([X, Hk], 1)              # GBDT heads see features + topology
    bt = BlockTransform(); P = presence(T, bt)
    pres_metrics = {}
    for g in GROUPS:
        y = P[g].astype(int); m = matched
        heads[g] = gbdt(XH[tr & m], y[tr & m], seed, max_iter=300)
        pres_metrics[g] = binary_metrics(y[te & m], heads[g].predict_proba(XH[te & m])[:, 1])
    for fl in FLAGS:
        y = np.nan_to_num(T[fl].to_numpy(), nan=0).astype(int); m = matched & P["two"]
        heads[fl] = gbdt(XH[tr & m], y[tr & m], seed, max_iter=300)
        pres_metrics[fl] = binary_metrics(y[te & m], heads[fl].predict_proba(XH[te & m])[:, 1])
    for c, (test, _) in SPIKES.items():
        v = T[c].to_numpy(); ok = np.isfinite(v); y = (test(np.nan_to_num(v, nan=1e9)) & ok).astype(int)
        m = matched & ok & (P["two"] if c == "opening_angle" else True)
        heads["spike_" + c] = gbdt(XH[tr & m], y[tr & m], seed, max_iter=300)
        pres_metrics["spike_" + c] = binary_metrics(y[te & m], heads["spike_" + c].predict_proba(XH[te & m])[:, 1])
    metrics["presence"] = pres_metrics
    print("presence:", {k: round(v["auc"], 3) for k, v in pres_metrics.items()}, flush=True)

    # ---- flow over the standardised block ----------------------------------------------------------------------
    Y = bt.matrix(T); bt.fit(Y[tr & matched])
    Z, Mfin = bt.standardise(Y)
    Mg = mask_from_presence(P, bt) * Mfin          # groups absent or values non-finite are masked
    Sreal = {}
    for c, (test, _) in SPIKES.items():
        j = bt.cols.index(c); v = T[c].to_numpy(); Sreal[c] = test(np.nan_to_num(v, nan=1e9)) & np.isfinite(v)
        Mg[Sreal[c], j] = 0.0                      # point masses are not modelled by the flow
    CH = np.concatenate([C, Hk], 1)
    model, hist = train_flow(Z[tr & matched], Mg[tr & matched], CH[tr & matched], Z[va & matched], Mg[va & matched], CH[va & matched], epochs, device, hidden, n_layers, seed=seed)
    metrics["flow_val_curve"] = hist

    # ---- closure on the test split: real vs generated slice blocks ---------------------------------------------
    tm = te & matched; idx = T.index[tm]
    rng = np.random.default_rng(seed)
    Ptm = heads["topology"].predict_proba(C[tm])
    kgen = (Ptm.cumsum(1) > rng.random(len(Ptm))[:, None]).argmax(1)
    XHg = np.concatenate([X[tm], topo.onehot(kgen)], 1); CHg = np.concatenate([C[tm], topo.onehot(kgen)], 1)
    Pg = {g: rng.random(int(tm.sum())) < heads[g].predict_proba(XHg)[:, 1] for g in GROUPS}
    Mgen = mask_from_presence(Pg, bt)
    Sgen = {}
    for c, (test, value) in SPIKES.items():
        j = bt.cols.index(c); p = heads["spike_" + c].predict_proba(XHg)[:, 1]
        Sgen[c] = (rng.random(len(p)) < p) & (Mgen[:, j] > 0); Mgen[Sgen[c], j] = 0.0
    Zgen = sample_flow(model, CHg, Mgen, device)
    gen = bt.frame(Zgen, Mgen, idx)
    real = bt.frame(Z[tm], Mg[tm], idx)           # same transform round trip for a fair comparison
    cnt_gen = topo.decode(kgen, rng); cnt_gen.index = idx
    for c in TOPO_COLS:
        gen[c] = cnt_gen[c]; real[c] = T[c].to_numpy()[tm]
    Ftm = F[tm]
    for c, (test, value) in SPIKES.items():
        gen.loc[Sgen[c], c] = value(Ftm)[Sgen[c]]
        real.loc[Sreal[c][tm], c] = T[c].to_numpy()[tm][Sreal[c][tm]]
    for fl in FLAGS:
        p = heads[fl].predict_proba(XHg)[:, 1]; gen[fl] = np.where(Pg["two"], (rng.random(len(p)) < p).astype(float), np.nan)
        real[fl] = T[fl].to_numpy()[tm]
    metrics["topology_sampled_vs_true_test"] = {"frac_equal": float((kgen == ktrue[tm]).mean()),
                                                "two_track_tuple_rate": {"real": float(((real.n_prong == 2) & (real.n_trk == 2) & (real.n_proton == 0) & (real.n_shower == 0)).mean()),
                                                                         "generated": float(((gen.n_prong == 2) & (gen.n_trk == 2) & (gen.n_proton == 0) & (gen.n_shower == 0)).mean())}}
    percol = {}
    for c in TOPO_COLS:
        r, f = real[c].to_numpy(), gen[c].to_numpy(); ok_r, ok_f = np.isfinite(r), np.isfinite(f)
        percol[c] = {"frac_present_real": float(ok_r.mean()), "frac_present_gen": float(ok_f.mean()), **sample_vs_real_1d(r[ok_r], f[ok_f])}
    for j, (c, tf, g) in enumerate(SPEC):   # compared in transformed units (log / atanh / lin), point masses included
        r, f = fwd(real[c].to_numpy(), tf), fwd(gen[c].to_numpy(), tf); ok_r, ok_f = np.isfinite(r), np.isfinite(f)
        percol[c] = {"frac_present_real": float(ok_r.mean()), "frac_present_gen": float(ok_f.mean()), **sample_vs_real_1d(r[ok_r], f[ok_f])}
    metrics["per_column"] = percol
    cf_real, cf_gen = cohpi_cutflow(real), cohpi_cutflow(gen)
    metrics["cohpi_cutflow_test"] = {"real": cf_real, "generated": cf_gen}
    sig = L.is_signal_cohpi.to_numpy()[tm]
    metrics["cohpi_cutflow_test_signal"] = {"real": cohpi_cutflow(real[sig]), "generated": cohpi_cutflow(gen[sig]), "n_signal_matched": int(sig.sum())}
    print("cutflow real:", cf_real); print("cutflow gen: ", cf_gen, flush=True)
    print("W1 (transformed units):", {c: round(v["w1"], 3) for c, v in percol.items()}, flush=True)

    # figures: a representative set of marginals
    show = ["vtx_z", "total_pe", "n_pfp", "long_trk_len", "long_dirz", "long_trk_chi2_pro_new", "short_trk_len", "reco_t", "range_p_mu", "opening_angle", "sum_integ_2cm", "nu_score"]
    fig, axs = plots.plt.subplots(3, 4, figsize=(14, 9)); axs = axs.ravel()
    for ax, c in zip(axs, show):
        tf = dict((s[0], s[1]) for s in SPEC).get(c, "lin")
        r, f = fwd(real[c].dropna(), tf), fwd(gen[c].dropna(), tf); r, f = r[np.isfinite(r)], f[np.isfinite(f)]
        lo, hi = np.percentile(np.concatenate([r, f]), [0.5, 99.5])
        plots.hist_compare(ax, r, f, np.linspace(lo, hi, 70), f"{tf}({c})" if tf != "lin" else c, (plots.REAL_LABEL, "flow"))
    plots.save(fig, out / "figures" / "m2_marginals.png")
    gen.to_parquet(out / "generated_test.parquet"); real.to_parquet(out / "real_test.parquet")
    torch.save({"state": model.state_dict(), "dim": Z.shape[1], "cond_dim": CH.shape[1], "topology_classes": topo.classes, "hidden": hidden, "n_layers": n_layers,
                "std_m": std.m, "std_s": std.s, "bt_med": bt.med, "bt_mad": bt.mad, "features": names, "spec": SPEC}, out / "model.pt")
    import pickle
    topo_head = heads.pop("topology"); torch.save({"state": topo_head.net.state_dict(), "n_in": C.shape[1], "n_classes": topo.other + 1}, out / "topology_head.pt")
    pickle.dump(heads, open(out / "heads.pkl", "wb"))
    metrics["runtime_s"] = time.time() - t0
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    print(f"done in {metrics['runtime_s']:.0f} s -> {out}", flush=True)
    return metrics
