"""Generate surrogated `slc` records from truth features with a trained M2 (v3) surrogate.

generate(sur, F) -> DataFrame with the production slice columns, one row per generated slice, indexed by the
neutrino index plus a `draw` level. Slice cardinality is sampled from the surrogate's head (0, 1 or 2 slices per
neutrino). The cohpi selection flags can be recomputed with `cohpi_flags`.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import torch

from ..models.flow import FlowMatcher
from ..train.m2 import (BlockTransform, FLAGS, GROUPS, MLPClassifier, SPIKES, SPEC, TOPO_COLS, Topology,
                        apply_range, apply_walls, mask_from_presence, sample_flow, sample_two_stage, topo_input)


class Surrogate:
    def __init__(self, model_dir, device="cuda"):
        d = f"{model_dir}"
        ck = torch.load(f"{d}/model.pt", map_location=device, weights_only=False)
        self.device = device
        self.features = ck["features"]; self.std_m, self.std_s = ck["std_m"], ck["std_s"]
        self.heads = pickle.load(open(f"{d}/heads.pkl", "rb"))
        self.two_stage = bool(ck.get("two_stage", False))
        self.topo = Topology.__new__(Topology)
        self.topo.classes = [tuple(c) for c in ck["topology_classes"]]; self.topo.lookup = {t: i for i, t in enumerate(self.topo.classes)}
        self.topo.other = len(self.topo.classes)
        self.topo.other_pool = np.asarray(ck.get("topology_other_pool", np.array([self.topo.classes[-1]], dtype=float)), dtype=float)
        self.range_curves = ck.get("range_curves")
        if self.two_stage:            # stage 1 (block + counts + presence), stage 2 (optional groups)
            self.bt1, self.bt2 = BlockTransform(ck["spec1"]), BlockTransform(ck["spec2"])
            (self.bt1.med, self.bt1.mad), (self.bt2.med, self.bt2.mad) = ck["bt1"], ck["bt2"]
            self.bt1.n_onehot = int(ck.get("n_onehot", 0))   # counts as a one-hot topology class in stage 1
            self.f1 = FlowMatcher(ck["dim1"], ck["cond_dim"], ck["hidden"], ck["n_layers"]).to(device)
            self.f2 = FlowMatcher(ck["dim2"], ck.get("cond2_dim", ck["cond_dim"] + ck["dim1"]), ck["hidden"], ck["n_layers"]).to(device)
            self.f1.load_state_dict(ck["state1"]); self.f2.load_state_dict(ck["state2"]); self.f1.eval(); self.f2.eval()
            self.topo_last = False; self.topo_head = None
            if ck.get("topo_cls"):            # topology class from the truth and the generated stage-1 block
                th = torch.load(f"{d}/topology_head.pt", map_location=device, weights_only=False)
                self.topo_head = MLPClassifier(th["n_in"], th["n_classes"], device=device); self.topo_head.net.load_state_dict(th["state"])
            return
        self.bt = BlockTransform(ck["spec"]); self.bt.med, self.bt.mad = ck["bt_med"], ck["bt_mad"]
        self.flow = FlowMatcher(ck["dim"], ck["cond_dim"], ck["hidden"], ck["n_layers"]).to(device)
        self.flow.load_state_dict(ck["state"]); self.flow.eval()
        th = torch.load(f"{d}/topology_head.pt", map_location=device, weights_only=False)
        self.topo_head = MLPClassifier(th["n_in"], th["n_classes"], device=device); self.topo_head.net.load_state_dict(th["state"])
        self.topo_last = bool(ck.get("topo_last", False))

    def standardise(self, X):
        return ((np.arcsinh(X) - self.std_m) / self.std_s).astype(np.float32)


def generate(sur: Surrogate, F: pd.DataFrame, seed: int = 0, cardinality: bool = True, n_steps: int = 64) -> pd.DataFrame:
    """F: features table (columns = sur.features). Returns generated slices (0, 1 or 2 per neutrino)."""
    rng = np.random.default_rng(seed)
    X = np.ascontiguousarray(F[sur.features].to_numpy(np.float32))
    if cardinality:
        Pn = np.zeros((len(X), 3)); Pn[:, sur.heads["n_matched_slices"].classes_] = sur.heads["n_matched_slices"].predict_proba(X)
        n = (Pn.cumsum(1) > rng.random(len(X))[:, None]).argmax(1)
    else:
        n = np.ones(len(X), dtype=int)
    rows = np.repeat(np.arange(len(X)), n); draw = np.concatenate([np.arange(k) for k in n]) if len(rows) else np.array([], dtype=int)
    if len(rows) == 0:
        return pd.DataFrame()
    Xr = X[rows]; C = sur.standardise(Xr); Fr = F.iloc[rows]
    if sur.two_stage:
        idx = pd.MultiIndex.from_arrays([F.index.get_level_values(i)[rows] for i in range(F.index.nlevels)] + [draw], names=list(F.index.names) + ["draw"])
        out = sample_two_stage(sur.f1, sur.f2, sur.bt1, sur.bt2, sur.heads, sur.topo, Xr, C, Fr, idx, sur.device, rng, sur.range_curves, n_steps=n_steps, topo_head=sur.topo_head)
        return out.drop(columns=[c for c in out.columns if c.startswith("has_")])
    if sur.topo_last:                 # topology sampled after the flow, from the truth features and the generated block
        Xpre, Cflow = Xr, C
    else:
        Pt = sur.topo_head.predict_proba(C); k = (Pt.cumsum(1) > rng.random(len(Pt))[:, None]).argmax(1)
        H = sur.topo.onehot(k); Xpre = np.concatenate([Xr, H], 1); Cflow = np.concatenate([C, H], 1)
    P = {g: rng.random(len(Xr)) < sur.heads[g].predict_proba(Xpre)[:, 1] for g in GROUPS}
    M = mask_from_presence(P, sur.bt)
    S = {}
    for c, (test, value) in SPIKES.items():
        j = sur.bt.cols.index(c); p = sur.heads["spike_" + c].predict_proba(Xpre)[:, 1]
        S[c] = (rng.random(len(p)) < p) & (M[:, j] > 0); M[S[c], j] = 0.0
    Z = sample_flow(sur.flow, Cflow, M, sur.device, n_steps=n_steps)
    if sur.topo_last:
        Pt = sur.topo_head.predict_proba(topo_input(C, Z, M)); k = (Pt.cumsum(1) > rng.random(len(Pt))[:, None]).argmax(1)
    XH = np.concatenate([Xr, sur.topo.onehot(k)], 1)
    idx = pd.MultiIndex.from_arrays([F.index.get_level_values(i)[rows] for i in range(F.index.nlevels)] + [draw], names=list(F.index.names) + ["draw"])
    out = sur.bt.frame(Z, M, idx)
    cnt = sur.topo.decode(k, rng); cnt.index = idx
    for c in TOPO_COLS:
        out[c] = cnt[c]
    for c, (test, value) in SPIKES.items():
        out.loc[S[c], c] = value(Fr)[S[c]]
    for fl in FLAGS:
        p = sur.heads[fl].predict_proba(XH)[:, 1]; out[fl] = np.where(P["two"], (rng.random(len(p)) < p).astype(float), np.nan)
    apply_walls(out)   # no-op for models without the wall targets
    apply_range(out, sur.range_curves)   # no-op for models without the range flags
    out["topology_class"] = k
    return out


def cohpi_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute the cohpi selection stages on a slice table (real or generated), as make_xsumdf does."""
    out = pd.DataFrame(index=df.index)
    out["pass_pe_bcfm"] = (df.total_pe > 1500.) & (df.bcfm_score > 0.05)
    out["pass_2trk"] = (df.n_prong == 2) & (df.n_trk == 2) & (df.n_proton == 0) & (df.n_shower == 0)
    out["pass_dir"] = (df.long_dirz > 0.7) & (df.short_dirz > 0.5) & (df.opening_angle > 0.5)
    out["pass_chi2"] = (df.long_trk_chi2_pro_new > 100.) & (df.short_trk_chi2_pro_new > 75.)
    out["pass_sphere"] = ((df.sum_integ_1cm.fillna(0) < 4000) & (df.sum_integ_2cm.fillna(0) < 7000) & (df.sum_integ_3cm.fillna(0) < 10000)
                          & (df.sum_integ_4cm.fillna(0) < 13000) & ((df.sum_integ_3cm / (df.sum_integ_4cm - df.sum_integ_3cm)).fillna(0) < 6.))
    out["pass_cathode"] = ((df.muon_start_x.abs() > 10) & (df.muon_end_x.abs() > 10) & (df.muon_start_x * df.muon_end_x > 0)
                           & (df.cpi_start_x.abs() > 10) & (df.cpi_end_x.abs() > 10) & (df.cpi_start_x * df.cpi_end_x > 0))
    out["pass_cohpi"] = out.pass_pe_bcfm & out.pass_2trk & out.pass_dir & out.pass_chi2 & out.pass_sphere & out.pass_cathode
    out["in_sr"] = out.pass_cohpi & (df.reco_t < 0.04)
    return out.fillna(False).astype(bool)
