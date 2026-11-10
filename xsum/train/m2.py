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
# Wall targets (M2 only; M4 keeps conditioning on SPEC): an exiting longest track has a length equal to the path from the
# vertex to the TPC wall along its direction, and an exiting cohpi track ends on a wall plane. A flow over the plain
# coordinates smears both. Per event: a flag that the longest track's length / path is within WALL_LEN_BAND of 1 and the
# offset (ratio - 1); per axis of the cohpi muon and pion-candidate end points, a flag that the end lies in the wall band
# (xsum.prep.pfpset.WALL_REF / WALL_BAND, the planes read off the real PFP track ends) and the outward offset. Flags
# are dequantised flow dimensions, offsets of tracks not on the wall are filled with noise; after generation flagged
# lengths / coordinates are snapped (apply_walls).
WALL_LEN_BAND = 0.03
AV_LO, AV_HI = np.array([-201.3, -203.7, 0.0]), np.array([201.3, 203.7, 500.8])   # active volume for the path to the wall
WALL_SPEC = [("long_wall", "flag", "base"), ("long_woff", "lin", "base")] + [
    (f"{p}_end_{k}_{a}", "flag" if k == "wall" else "lin", "two") for p in ("muon", "cpi") for k in ("wall", "woff") for a in "xyz"]


# Range momenta: the cohpi range_p_mu / range_p_cpi are functions of long_trk_len / short_trk_len (median deviation 0.03%,
# 97% / 93% of slices within 2%); the flow generates a flag "on the curve" per momentum and flagged values are replaced
# by the length-momentum curve (median log p in quantile bins of log length, fitted on the training slices) after the
# wall snap, so exiting tracks get the momentum of their truncated length (apply_range).
RANGE_BAND = 0.02
RANGE_PAIRS = {"mu": ("range_p_mu", "long_trk_len"), "cpi": ("range_p_cpi", "short_trk_len")}
RANGE_SPEC = [("range_mu_on", "flag", "two"), ("range_cpi_on", "flag", "two")]


def fit_range_curves(T):
    out = {}
    for k, (pc, lc) in RANGE_PAIRS.items():
        p, L = T[pc].to_numpy(np.float64), T[lc].to_numpy(np.float64); ok = np.isfinite(p) & np.isfinite(L) & (p > 0) & (L > 0)
        x, y = np.log(L[ok]), np.log(p[ok]); e = np.unique(np.quantile(x, np.linspace(0, 1, 301))); c = 0.5 * (e[1:] + e[:-1])
        b = np.clip(np.digitize(x, e) - 1, 0, len(c) - 1); med = np.array([np.median(y[b == i]) if (b == i).any() else np.nan for i in range(len(c))])
        g = np.isfinite(med); out[k] = (c[g], med[g])
    return out


def range_from_len(L, curve):
    with np.errstate(divide="ignore", invalid="ignore"): return np.exp(np.interp(np.log(L), *curve))


def add_range_targets(T, curves):
    for k, (pc, lc) in RANGE_PAIRS.items():
        p = T[pc].to_numpy(np.float64); ok = np.isfinite(p)
        r = p / range_from_len(T[lc].to_numpy(np.float64), curves[k]) - 1
        T[f"range_{k}_on"] = np.where(ok, np.abs(r) < RANGE_BAND, np.nan).astype(np.float32)
    return T


def apply_range(df, curves):
    """In place: flagged range momenta from the (wall-snapped) generated lengths."""
    if curves is None or "range_mu_on" not in df.columns: return df
    for k, (pc, lc) in RANGE_PAIRS.items():
        on = df[f"range_{k}_on"].to_numpy() > 0.5; new = range_from_len(df[lc].to_numpy(np.float64), curves[k])
        df[pc] = np.where(on & np.isfinite(new) & np.isfinite(df[pc].to_numpy()), new, df[pc].to_numpy())
    return df


def path_to_wall(vtx, d):
    """Distance from vtx [n,3] along the unit direction d [n,3] to the active-volume boundary (inf where undefined)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(d > 0, (AV_HI - vtx) / d, np.where(d < 0, (AV_LO - vtx) / d, np.inf))
    return np.nan_to_num(t, nan=np.inf).min(1)


def add_wall_targets(T, seed=0):
    """Adds the WALL_SPEC columns (float32) to a slice table with the real reco values."""
    from ..prep.pfpset import WALL_REF, WALL_BAND
    rng = np.random.default_rng(seed); n = len(T)
    L = T["long_trk_len"].to_numpy(np.float64)
    r = L / path_to_wall(T[["vtx_x", "vtx_y", "vtx_z"]].to_numpy(np.float64), T[["long_dirx", "long_diry", "long_dirz"]].to_numpy(np.float64)) - 1
    has = np.isfinite(L) & (L > 0); on = has & (np.abs(r) < WALL_LEN_BAND)
    T["long_wall"] = np.where(has, on, np.nan).astype(np.float32)
    T["long_woff"] = np.where(on, r, rng.uniform(-WALL_LEN_BAND, WALL_LEN_BAND, n)).astype(np.float32)
    for p in ("muon", "cpi"):
        for a in "xyz":
            e = T[f"{p}_end_{a}"].to_numpy(np.float64); lo, hi = WALL_REF[a]; (bl, bh) = WALL_BAND[a]
            off = np.where(e > 0.5 * (lo + hi), e - hi, lo - e); ok = np.isfinite(e); on = ok & (off >= bl) & (off <= bh)
            T[f"{p}_end_wall_{a}"] = np.where(ok, on, np.nan).astype(np.float32)
            T[f"{p}_end_woff_{a}"] = np.where(on, off, rng.uniform(bl, bh, n)).astype(np.float32)
    return T


def apply_walls(df):
    """In place on a generated slice frame with the WALL_SPEC columns: snap flagged longest-track lengths onto the path
    to the wall and flagged cohpi end coordinates onto the wall plane plus the offset."""
    from ..prep.pfpset import WALL_REF
    if "long_wall" not in df.columns: return df
    on = df["long_wall"].to_numpy() > 0.5
    path = path_to_wall(df[["vtx_x", "vtx_y", "vtx_z"]].to_numpy(np.float64), df[["long_dirx", "long_diry", "long_dirz"]].to_numpy(np.float64))
    with np.errstate(invalid="ignore", over="ignore"): new = path * (1 + np.nan_to_num(df["long_woff"].to_numpy(np.float64), nan=0.0))
    df["long_trk_len"] = np.where(on & np.isfinite(new) & (new > 0), new, df["long_trk_len"].to_numpy())
    from ..prep.pfpset import WALL_BAND
    for p in ("muon", "cpi"):
        for a in "xyz":
            on = df[f"{p}_end_wall_{a}"].to_numpy() > 0.5; e = df[f"{p}_end_{a}"].to_numpy(np.float64)
            lo, hi = WALL_REF[a]; (bl, bh) = WALL_BAND[a]
            # the generated offset of a flagged end is kept within the wall band (3-7% of the flagged ends fall outside)
            off = np.clip(np.nan_to_num(df[f"{p}_end_woff_{a}"].to_numpy(np.float64), nan=0.0), bl, bh)
            e = np.where(on & np.isfinite(e), np.where(e > 0.5 * (lo + hi), hi + off, lo - off), e)
            # unflagged ends beyond the wall band (real: none; generated: ~2% per axis, up to tens of cm) are
            # mirrored back inside about the wall plane, so the value stays within the active volume
            upper = e > 0.5 * (lo + hi); beyond = np.where(upper, e - hi, lo - e)
            fix = ~on & np.isfinite(e) & (beyond > bh)
            e = np.where(fix, np.clip(np.where(upper, 2 * hi - e, 2 * lo - e), lo, hi), e)
            df[f"{p}_end_{a}"] = e
    return df


def wall_closure(real, gen):
    """Wall consistency, real vs generated frames: longest track length / path within +-2%, cohpi muon end on the wall
    (+-3 cm of the active volume) and outside it."""
    out = {}
    for lab, d in (("real", real), ("gen", gen)):
        L = d["long_trk_len"].to_numpy(np.float64); ok = np.isfinite(L) & (L > 0)
        r = L[ok] / path_to_wall(d[["vtx_x", "vtx_y", "vtx_z"]].to_numpy(np.float64)[ok], d[["long_dirx", "long_diry", "long_dirz"]].to_numpy(np.float64)[ok])
        e = d[["muon_end_x", "muon_end_y", "muon_end_z"]].to_numpy(np.float64); e = e[np.isfinite(e[:, 0])]
        dw = np.minimum.reduce([AV_HI[0] - np.abs(e[:, 0]), AV_HI[1] - np.abs(e[:, 1]), e[:, 2], AV_HI[2] - e[:, 2]])
        out[lab] = {"long_len_at_path": float(np.mean(np.abs(r - 1) < 0.02)), "long_len_beyond_path": float(np.mean(r > 1.05)),
                    "muon_end_on_wall": float(np.mean(np.abs(dw) < 3)), "muon_end_outside": float(np.mean(dw < -3))}
    return out


GROUPS = ["short", "two", "s1", "s2", "s3", "s4"]
FLAGS = ["is_muon_contained", "is_cpi_contained"]   # Bernoulli, defined on the two-track group
COUNT_COLS = set()
TOPO_COLS = ["n_pfp", "n_prong", "n_trk", "n_proton", "n_shower"]   # joint categorical head, top patterns + "other"
N_TOPO = 120


EDGE = 1 - 1e-5
VTX_COLS = ["vtx_x", "vtx_y", "vtx_z"]
VTX_SCALE = 1.0   # cm: the reco vertex as the residual to the true vertex on an asinh scale (vtx_resid models)


def fwd(x, tf):
    x = np.asarray(x, dtype=np.float64)
    if tf == "log": return np.log(np.where(x > 0, x, np.nan))
    if tf == "log1p": return np.log1p(np.where(x >= 0, x, np.nan))
    if tf == "atanh": return np.arctanh(np.clip(x, -EDGE, EDGE))
    if tf == "asinh1": return np.arcsinh(x / VTX_SCALE)
    return x   # lin, flag, count


def inv(y, tf):
    if tf == "log": return np.exp(y)
    if tf == "log1p": return np.maximum(np.expm1(y), 0.0)
    if tf == "atanh": return np.tanh(y)
    if tf == "flag": return (y > 0.5).astype(np.float64)
    if tf == "count": return np.maximum(np.round(y), 0.0)
    if tf == "asinh1": return np.sinh(np.clip(y, -7.6, 7.6)) * VTX_SCALE   # residual within 1000 cm
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

    def matrix(self, df, seed=0):
        Y = np.stack([fwd(df[c].to_numpy(), tf) for c, tf, _ in self.spec], 1)
        rng = np.random.default_rng(seed)
        fl = [j for j, (_, tf, _) in enumerate(self.spec) if tf == "flag"]
        if fl: Y[:, fl] += rng.uniform(-0.4, 0.4, (len(Y), len(fl)))   # dequantised flags
        ct = [j for j, (_, tf, _) in enumerate(self.spec) if tf == "count"]
        if ct: Y[:, ct] += rng.uniform(-0.5, 0.5, (len(Y), len(ct)))   # dequantised integer counts
        if "bcfm_score" in self.cols:
            Y[:, self.cols.index("bcfm_score")][df["bcfm_score"].to_numpy() < -100] = np.nan   # -9999 sentinel: no flash match
        return Y

    def fit(self, Y):
        self.med = np.nanmedian(Y, 0); self.mad = 1.4826 * np.nanmedian(np.abs(Y - self.med), 0) + 1e-6
        for j, (_, tf, _) in enumerate(self.spec):   # flags: mean / std (the MAD of a 0/1 column can vanish)
            if tf in ("flag", "count"): self.med[j] = np.nanmean(Y[:, j]); self.mad[j] = np.nanstd(Y[:, j]) + 1e-6
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
        for t in [t for t in ("long", "short") if f"{t}_dirx" in df.columns]:
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


def train_flow(Ztr, Mtr, Ctr, ktr, Zva, Mva, Cva, kva, n_classes, epochs, device, hidden=512, n_layers=5, bs=4096, lr=1e-3, seed=0):
    """Conditioning = standardised features + one-hot topology class, assembled per batch on the GPU."""
    torch.manual_seed(seed)
    model = FlowMatcher(Ztr.shape[1], Ctr.shape[1] + n_classes, hidden, n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    t = lambda a: torch.tensor(a, device=device)
    Zt, Mt, Ct, kt = t(Ztr), t(Mtr), t(Ctr), torch.tensor(ktr, dtype=torch.long, device=device)
    Zv, Mv, Cv, kv = t(Zva), t(Mva), t(Cva), torch.tensor(kva, dtype=torch.long, device=device)
    cond = (lambda C, k: C) if n_classes == 0 else (lambda C, k: torch.cat([C, torch.nn.functional.one_hot(k, n_classes).to(C.dtype)], -1))
    best, best_state, hist = np.inf, None, []
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Zt), device=device); tl = 0.0; nb = 0
        for i in range(0, len(Zt), bs):
            idx = perm[i:i + bs]
            loss = model.loss(Zt[idx], cond(Ct[idx], kt[idx]), Mt[idx]).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tl += loss.item(); nb += 1
        sched.step(); model.eval()
        with torch.no_grad():
            vl = float(np.mean([model.loss(Zv[i:i + 65536], cond(Cv[i:i + 65536], kv[i:i + 65536]), Mv[i:i + 65536]).mean().item() for i in range(0, len(Zv), 65536)]))
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


def topo_input(C, Z, M):
    """Input of the topology head in topo_last models: truth features + the (generated or real) standardised slice
    block, zero where masked, + the mask."""
    return np.concatenate([C, Z * M, M], 1).astype(np.float32)


def report(real, gen, T, tm, L, kgen, ktrue, metrics, out):
    """Closure metrics (wall, topology, per column, cohpi cutflow), marginal figure and test parquets."""
    for c in ("range_p_mu", "range_p_cpi", "long_trk_len", "muon_end_x", "muon_end_y", "muon_end_z", "cpi_end_x", "cpi_end_y", "cpi_end_z"):
        real[c] = T[c].to_numpy()[tm]            # real values without the transform round trip (exact wall positions)
    metrics["wall_closure_test"] = wall_closure(real, gen); print("wall closure:", metrics["wall_closure_test"], flush=True)
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


def run(table_dir, out_dir, device="cuda", epochs=30, seed=0, max_rows=None, hidden=512, n_layers=5, card_table=None, walls=False,
        topo_last=False, range_curve=False, two_stage=False, topo_onehot=False, topo_cls=False, vtx_resid=False):
    """card_table: optional directory with features/targets (matched, n_matched_slices) for the cardinality head;
    then table_dir is expected to hold matched slices only (see scripts/split_official_table.py).
    Shipped M2 (m2_official_m2vw, 2026-11-09) = defaults + vtx_resid + walls; the defaults alone are the previous shipped
    run m2_official_m3. Studies (technote Sec. 8): walls alone (M2w), vtx_resid alone (M2v), topo_last + range_curve (M2t).
    vtx_resid: the reco vertex generated as asinh((reco - true) / 1 cm) and added back to the true vertex at generation.
    walls: wall flag / offset targets for the longest-track length and the cohpi end points (WALL_SPEC, apply_walls).
    topo_last: the topology class (joint PFP counts) is sampled AFTER the flow, from the truth features and the
    generated slice block, so that counts follow the generated tracks (e.g. an exiting longest track <-> fewer PFPs);
    presence and spike heads then see the truth features only and the flow is not conditioned on the topology.
    range_curve: range momenta flagged on / off the length-momentum curve (RANGE_SPEC, apply_range).
    two_stage: stage 1 = one flow over the always-present block, the PFP counts and the group-presence flags; stage 2 =
    a flow over the optional groups conditioned on the truth and the stage-1 block, masked by the stage-1 presence
    (run_two_stage).
    topo_onehot (with two_stage): the PFP counts enter stage 1 as the topology class (joint count tuple), a one-hot
    over the classes dequantised to +-1 +- U(0.8), decoded by argmax, instead of five dequantised integers; generated
    slices then carry only count tuples of the real table.
    topo_cls (with two_stage): stage 1 carries no counts; the topology class is sampled from a classifier on the truth
    and the generated stage-1 block (values + presence flags), restricted to the classes compatible with the sampled
    presence; stage 2 is also conditioned on the class.
    vtx_resid: the reco vertex is generated as the residual to the true vertex, asinh(d / 1 cm), instead of the absolute
    position under a robust linear scaling (which hides the 0.5 cm resolution core next to the 10 cm-scale
    misplacements); the wall point masses stay absolute. Added back at generation."""
    out = pathlib.Path(out_dir); (out / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tcols = ["matched", "n_matched_slices"] + [c for c, _, _ in SPEC] + TOPO_COLS + FLAGS
    # memory-aware loading: the cardinality head sees all neutrinos (on a subsample), everything else only matched slices
    if card_table:                                    # cardinality head from its own (subsampled) table
        cF = pd.read_parquet(f"{card_table}/features.parquet"); cT = pd.read_parquet(f"{card_table}/targets.parquet", columns=["matched", "n_matched_slices"])
        cfid = cF.index.get_level_values("chunk").to_numpy() * 100000 + cF.index.get_level_values("__ntuple").to_numpy()
        csp = split_by_level(pd.MultiIndex.from_arrays([cfid], names=["file"]), level="file", seed=seed)
        cX = np.ascontiguousarray(cF.to_numpy(np.float32)); del cF
        ncl = np.clip(cT.n_matched_slices.to_numpy(), 0, 2).astype(int); del cT
        itr = np.flatnonzero(csp["train"]); itr = np.sort(np.random.default_rng(seed).choice(itr, min(len(itr), 2_000_000), replace=False))
        heads = {"n_matched_slices": gbdt(cX[itr], ncl[itr], seed)}
        cte = csp["test"]; Pn = np.zeros((int(cte.sum()), 3)); Pn[:, heads["n_matched_slices"].classes_] = heads["n_matched_slices"].predict_proba(cX[cte])
        card_metrics = {"marginal_true": np.bincount(ncl[cte], minlength=3).tolist(), "mean_prob": Pn.mean(0).tolist(),
                        "logloss": float(-np.log(np.clip(Pn[np.arange(len(Pn)), ncl[cte]], 1e-9, None)).mean()), "n_card_rows": int(len(cX))}
        del cX, ncl, Pn, itr, csp
        print("cardinality head trained on the card table", flush=True)
    card = pd.read_parquet(f"{table_dir}/targets.parquet", columns=["matched", "n_matched_slices"])
    F = pd.read_parquet(f"{table_dir}/features.parquet")
    sub_rows = None
    if max_rows and len(F) > max_rows:
        sub_rows = np.sort(np.random.default_rng(seed).choice(len(F), max_rows, replace=False)); F, card = F.iloc[sub_rows], card.iloc[sub_rows]
    names = list(F.columns); X = np.ascontiguousarray(F.to_numpy(np.float32))
    F = F[VTX_COLS]                    # the true vertex: point-mass signs, and the reference of the vertex residual
    file_id = F.index.get_level_values("chunk").to_numpy() * 100000 + F.index.get_level_values("__ntuple").to_numpy()
    split = split_by_level(pd.MultiIndex.from_arrays([file_id], names=["file"]), level="file", seed=seed)
    tr, va, te = split["train"], split["val"], split["test"]
    matched = card.matched.to_numpy().astype(bool)
    metrics = {"n": int(len(F)), "n_matched": int(matched.sum()), "epochs": epochs, "hidden": hidden, "n_layers": n_layers}
    print(f"table {len(F)} neutrinos, matched {matched.sum()}  ({time.time()-t0:.0f} s)", flush=True)

    # ---- slice cardinality (GBDT on at most 2M training neutrinos) --------------------------------------------
    if card_table:
        metrics["n_matched_slices"] = card_metrics; del card
    else:
        heads = {}
        ncl = np.clip(card.n_matched_slices.to_numpy(), 0, 2).astype(int)
        itr = np.flatnonzero(tr); itr = np.sort(np.random.default_rng(seed).choice(itr, min(len(itr), 2_000_000), replace=False))
        heads["n_matched_slices"] = gbdt(X[itr], ncl[itr], seed)
        Pn = np.zeros((int(te.sum()), 3)); Pn[:, heads["n_matched_slices"].classes_] = heads["n_matched_slices"].predict_proba(X[te])
        metrics["n_matched_slices"] = {"marginal_true": np.bincount(ncl[te], minlength=3).tolist(), "mean_prob": Pn.mean(0).tolist(),
                                       "logloss": float(-np.log(np.clip(Pn[np.arange(len(Pn)), ncl[te]], 1e-9, None)).mean())}
        del card, Pn, itr
    # everything below concerns matched slices only
    keep = matched
    F, X = F[keep], X[keep]
    tr, va, te = tr[keep], va[keep], te[keep]
    T = pd.read_parquet(f"{table_dir}/targets.parquet", columns=tcols)
    if sub_rows is not None: T = T.iloc[sub_rows]          # same rows as the features
    T = T[keep]
    assert T.index.equals(F.index), "targets and features are not row-aligned"
    for c in T.columns: T[c] = T[c].astype(np.float32)
    if walls: T = add_wall_targets(T, seed)
    L = pd.read_parquet(f"{table_dir}/labels.parquet", columns=["is_signal_cohpi"])
    if sub_rows is not None: L = L.iloc[sub_rows]
    L = L[keep]
    matched = np.ones(len(F), dtype=bool)
    std = Standardizer(X[tr]); C = std(X)
    import gc; gc.collect()
    import resource
    def mem(tag): print(f"[mem] {tag}: peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB", flush=True)
    print(f"restricted to {len(F)} matched slices", flush=True); mem("after load")

    if two_stage:
        return run_two_stage(T, X, C, F, L, tr, va, te, heads, metrics, out, device, epochs, hidden, n_layers, seed, walls, range_curve, names, std, t0,
                             topo_onehot=topo_onehot, topo_cls=topo_cls)
    curves = None
    if range_curve:
        curves = fit_range_curves(T[tr]); T = add_range_targets(T, curves)
        metrics["range_on_curve_train"] = {k: float(np.nanmean(T.loc[tr, f"range_{k}_on"])) for k in RANGE_PAIRS}
    Tabs = T[VTX_COLS].copy()                     # absolute reco vertex (point-mass tests, real frame of the closure)
    spec = SPEC + (WALL_SPEC if walls else []) + (RANGE_SPEC if range_curve else [])
    if vtx_resid:
        for c in VTX_COLS: T[c] = T[c] - F[c].to_numpy(np.float32)
        spec = [(c, "asinh1", g) if c in VTX_COLS else (c, tf, g) for c, tf, g in spec]
    bt = BlockTransform(spec); P = presence(T, bt)
    metrics["vtx_resid"] = bool(vtx_resid)
    # ---- standardised block (needed by the topology head of topo_last models) ----------------------------------
    Y = bt.matrix(T).astype(np.float32); bt.fit(Y[tr & matched])
    Z, Mfin = bt.standardise(Y); del Y
    Mg = mask_from_presence(P, bt) * Mfin          # groups absent or values non-finite are masked
    Sreal = {}
    for c, (test, _) in SPIKES.items():
        j = bt.cols.index(c); v = (Tabs[c] if c in VTX_COLS else T[c]).to_numpy(); Sreal[c] = test(np.nan_to_num(v, nan=1e9)) & np.isfinite(v)
        Mg[Sreal[c], j] = 0.0                      # point masses are not modelled by the flow

    # topology: joint categorical over the PFP counts; sampled first (fed to every other component) or, topo_last,
    # after the flow from the truth features and the generated block
    topo = Topology(T, tr & matched); ktrue = topo.encode(T)
    rng0 = np.random.default_rng(seed + 1)
    def sub(mask, cap):
        rows = np.flatnonzero(mask)
        return np.sort(rng0.choice(rows, cap, replace=False)) if len(rows) > cap else rows
    if topo_last:
        rtr, rva, rte = sub(tr & matched, 1_500_000), sub(va & matched, 300_000), np.flatnonzero(te & matched)
        tX = lambda r: topo_input(C[r], Z[r], Mg[r])
        heads["topology"] = MLPClassifier(C.shape[1] + 2 * Z.shape[1], topo.other + 1, device=device, seed=seed).fit(tX(rtr), ktrue[rtr], tX(rva), ktrue[rva])
        Pt = heads["topology"].predict_proba(tX(rte))
    else:
        heads["topology"] = MLPClassifier(C.shape[1], topo.other + 1, device=device, seed=seed).fit(C[tr & matched], ktrue[tr & matched], C[va & matched], ktrue[va & matched])
        Pt = heads["topology"].predict_proba(C[te & matched])
    metrics["topology"] = {"n_classes": topo.other + 1, "frac_other_train": float((ktrue[tr & matched] == topo.other).mean()), "topo_last": bool(topo_last),
                           "logloss": float(-np.log(np.clip(Pt[np.arange(len(Pt)), ktrue[te & matched]], 1e-9, None)).mean()),
                           "logloss_marginal": float(-np.log(np.clip(np.bincount(ktrue[tr & matched], minlength=topo.other + 1)[ktrue[te & matched]] / (tr & matched).sum(), 1e-9, None)).mean()),
                           "accuracy": float((Pt.argmax(1) == ktrue[te & matched]).mean())}
    print("topology:", {k: round(v, 4) for k, v in metrics["topology"].items()}, flush=True); del Pt
    def xh(mask, cap=1_000_000, with_topo=True):
        """features (+ topology one-hot) for the rows of `mask` (subsampled to `cap` rows): built on demand, never for all rows."""
        rows = np.flatnonzero(mask)
        if len(rows) > cap: rows = np.sort(np.random.default_rng(seed).choice(rows, cap, replace=False))
        return (np.concatenate([X[rows], topo.onehot(ktrue[rows])], 1) if with_topo else X[rows]), rows
    pres_metrics = {}
    def fit_head(name, y, m, with_topo=True):
        Xa, ra = xh(tr & m, with_topo=with_topo); heads[name] = gbdt(Xa, y[ra], seed, max_iter=300); del Xa
        Xb, rb = xh(te & m, cap=10**9, with_topo=with_topo); pres_metrics[name] = binary_metrics(y[rb], heads[name].predict_proba(Xb)[:, 1]); del Xb
    for g in GROUPS:
        fit_head(g, P[g].astype(int), matched, with_topo=not topo_last)
    for fl in FLAGS:
        fit_head(fl, np.nan_to_num(T[fl].to_numpy(), nan=0).astype(int), matched & P["two"])
    for c, (test, _) in SPIKES.items():
        v = (Tabs[c] if c in VTX_COLS else T[c]).to_numpy(); ok = np.isfinite(v); y = (test(np.nan_to_num(v, nan=1e9)) & ok).astype(int)
        fit_head("spike_" + c, y, matched & ok & (P["two"] if c == "opening_angle" else True), with_topo=not topo_last)
    metrics["presence"] = pres_metrics; mem("after heads")
    print("presence:", {k: round(v["auc"], 3) for k, v in pres_metrics.items()}, flush=True)

    # ---- flow over the standardised block ----------------------------------------------------------------------
    model, hist = train_flow(Z[tr & matched], Mg[tr & matched], C[tr & matched], ktrue[tr & matched], Z[va & matched], Mg[va & matched], C[va & matched], ktrue[va & matched],
                             0 if topo_last else topo.other + 1, epochs, device, hidden, n_layers, seed=seed)
    metrics["flow_val_curve"] = hist

    # ---- closure on the test split: real vs generated slice blocks ---------------------------------------------
    tm = te & matched; idx = T.index[tm]
    rng = np.random.default_rng(seed)
    if topo_last:
        Xpre, Cflow = X[tm], C[tm]
    else:
        Ptm = heads["topology"].predict_proba(C[tm]); kgen = (Ptm.cumsum(1) > rng.random(len(Ptm))[:, None]).argmax(1)
        Xpre = np.concatenate([X[tm], topo.onehot(kgen)], 1); Cflow = np.concatenate([C[tm], topo.onehot(kgen)], 1)
    Pg = {g: rng.random(int(tm.sum())) < heads[g].predict_proba(Xpre)[:, 1] for g in GROUPS}
    Mgen = mask_from_presence(Pg, bt)
    Sgen = {}
    for c, (test, value) in SPIKES.items():
        j = bt.cols.index(c); p = heads["spike_" + c].predict_proba(Xpre)[:, 1]
        Sgen[c] = (rng.random(len(p)) < p) & (Mgen[:, j] > 0); Mgen[Sgen[c], j] = 0.0
    Zgen = sample_flow(model, Cflow, Mgen, device)
    if topo_last:
        Ptm = heads["topology"].predict_proba(topo_input(C[tm], Zgen, Mgen)); kgen = (Ptm.cumsum(1) > rng.random(len(Ptm))[:, None]).argmax(1)
    XHg = np.concatenate([X[tm], topo.onehot(kgen)], 1)
    gen = bt.frame(Zgen, Mgen, idx)
    real = bt.frame(Z[tm], Mg[tm], idx)           # same transform round trip for a fair comparison
    vtx_range = {c: (float(Tabs[c][tr].min()), float(Tabs[c][tr].max())) for c in VTX_COLS} if vtx_resid else None
    if vtx_resid:                                 # back to absolute positions, within the real range (residual tails of outside-AV neutrinos)
        for c in VTX_COLS:
            gen[c] = (gen[c] + F[c].to_numpy()[tm]).clip(*vtx_range[c]); real[c] = real[c] + F[c].to_numpy()[tm]; T[c] = Tabs[c]
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
    apply_walls(gen); apply_range(gen, curves)
    report(real, gen, T, tm, L, kgen, ktrue, metrics, out)
    torch.save({"state": model.state_dict(), "dim": Z.shape[1], "cond_dim": C.shape[1] + (0 if topo_last else topo.other + 1), "topo_last": bool(topo_last), "range_curves": curves, "vtx_resid": bool(vtx_resid), "vtx_range": vtx_range, "topology_classes": topo.classes, "topology_other_pool": topo.other_pool, "hidden": hidden, "n_layers": n_layers,
                "std_m": std.m, "std_s": std.s, "bt_med": bt.med, "bt_mad": bt.mad, "features": names, "spec": bt.spec}, out / "model.pt")
    import pickle
    topo_head = heads.pop("topology"); torch.save({"state": topo_head.net.state_dict(), "n_in": C.shape[1] + (2 * Z.shape[1] if topo_last else 0), "n_classes": topo.other + 1}, out / "topology_head.pt")
    pickle.dump(heads, open(out / "heads.pkl", "wb"))
    metrics["runtime_s"] = time.time() - t0
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    print(f"done in {metrics['runtime_s']:.0f} s -> {out}", flush=True)
    return metrics


def stage_specs(walls, range_curve, topo_onehot=False, topo_cls=False):
    """(stage-1 spec, stage-2 spec) of the two-stage M2; with topo_onehot (one-hot block) or topo_cls (classifier) the
    counts are not in the spec."""
    full = SPEC + (WALL_SPEC if walls else []) + (RANGE_SPEC if range_curve else [])
    cnt = [] if (topo_onehot or topo_cls) else [(c, "count", "base") for c in TOPO_COLS]
    s1 = [x for x in full if x[2] == "base"] + cnt + [(f"has_{g}", "flag", "base") for g in GROUPS]
    return s1, [x for x in full if x[2] != "base"]


ONEHOT_NOISE = 0.8   # one-hot block: 2 H - 1 + U(-0.8, 0.8); the true class stays the argmax


def onehot_block(k, n, rng):
    H = (rng.random((len(k), n), dtype=np.float32) - 0.5) * np.float32(2 * ONEHOT_NOISE) - 1.0   # float32 throughout (memory)
    H[np.arange(len(k)), k] += 2.0
    return H


def run_two_stage(T, X, C, F, L, tr, va, te, heads, metrics, out, device, epochs, hidden, n_layers, seed, walls, range_curve, names, std, t0,
                  topo_onehot=False, topo_cls=False):
    matched = np.ones(len(T), dtype=bool); trm, vam, tm = tr & matched, va & matched, te & matched
    curves = None
    if range_curve:
        curves = fit_range_curves(T[tr]); T = add_range_targets(T, curves)
    P = presence(T, None)
    for g in GROUPS: T[f"has_{g}"] = P[g].astype(np.float32)
    s1, s2 = stage_specs(walls, range_curve, topo_onehot, topo_cls); bt1, bt2 = BlockTransform(s1), BlockTransform(s2)
    Y = bt1.matrix(T, seed).astype(np.float32); bt1.fit(Y[trm]); Z1, M1 = bt1.standardise(Y); del Y
    Y = bt2.matrix(T, seed + 1).astype(np.float32); bt2.fit(Y[trm]); Z2, M2 = bt2.standardise(Y); del Y
    M2 = mask_from_presence(P, bt2) * M2
    Sreal = {}
    for c, (test, _) in SPIKES.items():           # point masses: masked in their stage, sampled by GBDT heads on truth
        bt = bt1 if c in bt1.cols else bt2; Mx = M1 if c in bt1.cols else M2
        v = T[c].to_numpy(); Sreal[c] = test(np.nan_to_num(v, nan=1e9)) & np.isfinite(v); Mx[Sreal[c], bt.cols.index(c)] = 0.0
    topo = Topology(T, trm); ktrue = topo.encode(T)
    bt1.n_onehot = topo.other + 1 if topo_onehot else 0
    if topo_onehot:                               # counts as the topology class, appended after the stage-1 columns
        Z1 = np.concatenate([Z1, onehot_block(ktrue, bt1.n_onehot, np.random.default_rng(seed + 2))], 1)
        M1 = np.concatenate([M1, np.ones((len(M1), bt1.n_onehot), dtype=np.float32)], 1)
    metrics["two_stage"] = {"dim1": Z1.shape[1], "dim2": len(s2), "n_onehot": bt1.n_onehot}
    pres = {}
    def fit_head(name, y, m, Xin):
        rows = np.flatnonzero(tr & m)
        if len(rows) > 1_000_000: rows = np.sort(np.random.default_rng(seed).choice(rows, 1_000_000, replace=False))
        heads[name] = gbdt(Xin(rows), y[rows], seed, max_iter=300); rt = np.flatnonzero(te & m)
        pres[name] = binary_metrics(y[rt], heads[name].predict_proba(Xin(rt))[:, 1])
    xo = lambda r: X[r]; xt = lambda r: np.concatenate([X[r], topo.onehot(ktrue[r])], 1)
    for fl in FLAGS:
        fit_head(fl, np.nan_to_num(T[fl].to_numpy(), nan=0).astype(int), matched & P["two"], xt)
    for c, (test, _) in SPIKES.items():
        v = T[c].to_numpy(); ok = np.isfinite(v); y = (test(np.nan_to_num(v, nan=1e9)) & ok).astype(int)
        fit_head("spike_" + c, y, matched & ok & (P["two"] if c == "opening_angle" else True), xo)
    metrics["presence"] = pres; print("heads:", {k: round(v["auc"], 3) for k, v in pres.items()}, flush=True)

    print("stage 1 flow", flush=True)
    f1, h1 = train_flow(Z1[trm], M1[trm], C[trm], ktrue[trm], Z1[vam], M1[vam], C[vam], ktrue[vam], 0, epochs, device, hidden, n_layers, seed=seed)
    if topo_cls:                                  # topology class from the truth and the generated stage-1 block
        rng0 = np.random.default_rng(seed + 1)
        def sub(mask, cap):
            rows = np.flatnonzero(mask); return np.sort(rng0.choice(rows, cap, replace=False)) if len(rows) > cap else rows
        rtr, rva, rte = sub(trm, 1_500_000), sub(vam, 300_000), np.flatnonzero(tm)
        tX = lambda r: topo_input(C[r], Z1[r], M1[r])
        print("topology classifier", flush=True)
        heads["topology"] = MLPClassifier(C.shape[1] + 2 * Z1.shape[1], topo.other + 1, device=device, seed=seed).fit(tX(rtr), ktrue[rtr], tX(rva), ktrue[rva])
        Pt = heads["topology"].predict_proba(tX(rte))
        metrics["topology"] = {"n_classes": topo.other + 1, "topo_cls": True,
                               "logloss": float(-np.log(np.clip(Pt[np.arange(len(Pt)), ktrue[rte]], 1e-9, None)).mean()),
                               "logloss_marginal": float(-np.log(np.clip(np.bincount(ktrue[trm], minlength=topo.other + 1)[ktrue[rte]] / trm.sum(), 1e-9, None)).mean()),
                               "accuracy": float((Pt.argmax(1) == ktrue[rte]).mean())}
        print("topology:", {k: round(v, 4) for k, v in metrics["topology"].items()}, flush=True); del Pt
    c2 = lambda m: np.concatenate([C[m], Z1[m] * M1[m]] + ([topo.onehot(ktrue[m])] if topo_cls else []), 1).astype(np.float32)   # train / val rows only (memory)
    print("stage 2 flow", flush=True)
    f2, h2 = train_flow(Z2[trm], M2[trm], c2(trm), ktrue[trm], Z2[vam], M2[vam], c2(vam), ktrue[vam], 0, epochs, device, hidden, n_layers, seed=seed + 1)
    metrics["flow_val_curve"] = h1; metrics["flow2_val_curve"] = h2

    # ---- closure on the test split
    idx = T.index[tm]; rng = np.random.default_rng(seed)
    gen = sample_two_stage(f1, f2, bt1, bt2, heads, topo, X[tm], C[tm], F[tm], idx, device, rng, curves, topo_head=heads.get("topology"))
    kgen = topo.encode(gen)
    nb = len(bt1.cols); real = pd.concat([bt1.frame(Z1[tm][:, :nb], M1[tm][:, :nb], idx), bt2.frame(Z2[tm], M2[tm], idx)], axis=1)
    for c in TOPO_COLS: real[c] = T[c].to_numpy()[tm]
    for c, (test, value) in SPIKES.items(): real.loc[Sreal[c][tm], c] = T[c].to_numpy()[tm][Sreal[c][tm]]
    for fl in FLAGS: real[fl] = T[fl].to_numpy()[tm]
    metrics["two_stage_consistency"] = {
        "ntrk_ge2_without_short": {"real": float(((real.n_trk >= 2) & real.short_trk_len.isna()).mean()), "gen": float(((gen.n_trk >= 2) & gen.short_trk_len.isna()).mean())},
        "presence_rate": {g: {"real": float(P[g][tm].mean()), "gen": float(gen[f"has_{g}"].mean())} for g in GROUPS}}
    metrics["two_stage_consistency"]["rule_fixes"] = gen.attrs.get("rule_fixes", {})
    print("consistency:", metrics["two_stage_consistency"], flush=True)
    real = real.drop(columns=[f"has_{g}" for g in GROUPS]); gen = gen.drop(columns=[f"has_{g}" for g in GROUPS])
    report(real, gen, T, tm, L, kgen, ktrue, metrics, out)
    if topo_cls:
        th = heads.pop("topology"); torch.save({"state": th.net.state_dict(), "n_in": C.shape[1] + 2 * Z1.shape[1], "n_classes": topo.other + 1}, out / "topology_head.pt")
    torch.save({"two_stage": True, "topo_cls": bool(topo_cls), "state1": f1.state_dict(), "state2": f2.state_dict(), "dim1": Z1.shape[1], "dim2": Z2.shape[1], "cond_dim": C.shape[1],
                "cond2_dim": C.shape[1] + Z1.shape[1] + (topo.other + 1 if topo_cls else 0),
                "spec1": bt1.spec, "spec2": bt2.spec, "bt1": (bt1.med, bt1.mad), "bt2": (bt2.med, bt2.mad), "range_curves": curves,
                "topology_classes": topo.classes, "topology_other_pool": topo.other_pool, "hidden": hidden, "n_layers": n_layers, "n_onehot": bt1.n_onehot,
                "std_m": std.m, "std_s": std.s, "features": names, "spec": bt1.spec + bt2.spec}, out / "model.pt")
    import pickle
    pickle.dump(heads, open(out / "heads.pkl", "wb"))
    metrics["runtime_s"] = time.time() - t0
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    print(f"done in {metrics['runtime_s']:.0f} s -> {out}", flush=True)
    return metrics


def compatible_classes(topo, has_short, has_two):
    """[n, n_classes] bool: class allowed given the sampled presence (n_trk >= 2 needs a short track; the two-track block
    needs n_prong = n_trk = 2). The 'other' class (rare, mostly high-multiplicity tuples) needs a short track and no two-track block."""
    arr = np.array(topo.classes, dtype=float); ntrk, nprong = arr[:, 2], arr[:, 1]
    need_short = np.append(ntrk >= 2, True); is22 = np.append((nprong == 2) & (ntrk == 2), False)
    ok = ~need_short[None, :] | has_short[:, None]
    ok &= ~has_two[:, None] | is22[None, :]
    return ok


def sample_two_stage(f1, f2, bt1, bt2, heads, topo, X, C, Fv, idx, device, rng, curves, n_steps=64, topo_head=None):
    """Two-stage generation: stage 1 (block, counts, presence) -> masks -> stage 2 (optional groups); point masses,
    containment flags (with the topology class of the generated counts), wall and range snaps."""
    n = len(X); M1 = np.ones((n, len(bt1.cols)), dtype=np.float32); S = {}
    for c, (test, value) in SPIKES.items():
        if c in bt1.cols:
            S[c] = rng.random(n) < heads["spike_" + c].predict_proba(X)[:, 1]; M1[S[c], bt1.cols.index(c)] = 0.0
    no = getattr(bt1, "n_onehot", 0)
    if no: M1 = np.concatenate([M1, np.ones((n, no), dtype=np.float32)], 1)
    Z1 = sample_flow(f1, C, M1, device, n_steps=n_steps)
    nb = len(bt1.cols); g1 = bt1.frame(Z1[:, :nb], M1[:, :nb], idx)
    if no:                                        # counts = the decoded topology class (only real count tuples)
        cnt = topo.decode(Z1[:, nb:].argmax(1), rng)
        for c in TOPO_COLS: g1[c] = cnt[c].to_numpy()
    # hard relations of the real reco (never violated in the real table): counts nested, a short track whenever
    # n_trk >= 2, the cohpi two-track block only for n_prong = n_trk = 2; the rate of fixes is kept as a diagnostic
    fix = {}
    if "n_pfp" in g1.columns:                    # counts generated in stage 1 (not with a topology classifier)
        for c, ub in (("n_prong", "n_pfp"), ("n_trk", "n_pfp"), ("n_proton", "n_trk")):
            bad = g1[c] > g1[ub]; fix[c] = float(bad.mean()); g1.loc[bad, c] = g1.loc[bad, ub]
        need = (g1.n_trk >= 2) & (g1.has_short < 0.5); fix["short_for_ntrk2"] = float(need.mean()); g1.loc[need, "has_short"] = 1.0
        bad = (g1.has_two > 0.5) & ~((g1.n_prong == 2) & (g1.n_trk == 2)); fix["two_outside_2trk"] = float(bad.mean()); g1.loc[bad, "has_two"] = 0.0
    bad = (g1.has_two > 0.5) & (g1.has_short < 0.5); fix["two_without_short"] = float(bad.mean()); g1.loc[bad, "has_short"] = 1.0
    Pg = {g: g1[f"has_{g}"].to_numpy() > 0.5 for g in GROUPS}
    H = None
    if topo_head is not None:                    # topology class from the truth and the generated stage-1 block, compatible with the presence
        Pt = topo_head.predict_proba(topo_input(C, Z1, M1)) * compatible_classes(topo, Pg["short"], Pg["two"])
        tot = Pt.sum(1); fix["class_no_compatible"] = float((tot <= 0).mean())
        Pt = np.where(tot[:, None] > 0, Pt / np.maximum(tot, 1e-12)[:, None], topo_head.predict_proba(topo_input(C, Z1, M1)))
        kc = (Pt.cumsum(1) > rng.random(len(Pt))[:, None]).argmax(1); cnt = topo.decode(kc, rng)
        for c in TOPO_COLS: g1[c] = cnt[c].to_numpy()
        need = (g1.n_trk >= 2) & (g1.has_short < 0.5); fix["short_for_ntrk2_after_class"] = float(need.mean()); g1.loc[need, "has_short"] = 1.0
        bad = (g1.has_two > 0.5) & ~((g1.n_prong == 2) & (g1.n_trk == 2)); fix["two_outside_2trk_after_class"] = float(bad.mean()); g1.loc[bad, "has_two"] = 0.0
        Pg = {g: g1[f"has_{g}"].to_numpy() > 0.5 for g in GROUPS}; H = topo.onehot(kc)
    M2 = mask_from_presence(Pg, bt2)
    for c, (test, value) in SPIKES.items():
        if c in bt2.cols:
            j = bt2.cols.index(c); S[c] = (rng.random(n) < heads["spike_" + c].predict_proba(X)[:, 1]) & (M2[:, j] > 0); M2[S[c], j] = 0.0
    Z2 = sample_flow(f2, np.concatenate([C, Z1 * M1] + ([H] if H is not None else []), 1).astype(np.float32), M2, device, n_steps=n_steps)
    gen = pd.concat([g1, bt2.frame(Z2, M2, idx)], axis=1)
    for c, (test, value) in SPIKES.items(): gen.loc[S[c], c] = value(Fv)[S[c]]
    k = topo.encode(gen); XH = np.concatenate([X, topo.onehot(k)], 1)
    for fl in FLAGS:
        p = heads[fl].predict_proba(XH)[:, 1]; gen[fl] = np.where(Pg["two"], (rng.random(n) < p).astype(float), np.nan)
    apply_walls(gen); apply_range(gen, curves)
    gen["topology_class"] = k
    gen.attrs["rule_fixes"] = fix
    return gen
