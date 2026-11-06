"""Generate PFP sets for generated slice records with a trained M4 model, and lay them out as a production-style
`pfp` table (the subset of columns the set model carries)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..data.pfpset_tf import PfpTransform
from ..models.pfpflow import PfpFlow
from ..prep.pfpset import FIELDS, N_MAX, NAMES
from ..train.m2 import SPEC, TOPO_COLS, BlockTransform, MLPClassifier, mask_from_presence
from ..train.m4 import N_EMPTY_MAX

# generated field -> production pfp column
PFP_COLS = {"trackScore": "trackScore", "dist_to_vertex": "dist_to_vertex", "is_primary": "parent_is_primary",
            "start_x": "trk_start_x", "start_y": "trk_start_y", "start_z": "trk_start_z", "dir_x": "trk_dir_x", "dir_y": "trk_dir_y", "dir_z": "trk_dir_z",
            "trk_len": "trk_len", "rangeP_mu": "trk_rangeP_mu", "mcsP_mu": "trk_mcsP_mu", "chi2_mu": "chi2_mu", "chi2_pro": "chi2_pro",
            "chi2_mu_new": "chi2_mu_new_I2", "chi2_pro_new": "chi2_pro_new_I2", "chi2_kaon": "chi2_kaon_I2", "chi2_pion": "chi2_pion_I2",
            "pida": "pida", "chi2_ndof": "chi2_ndof", "shw_energy": "shw_bestplane_energy", "shw_len": "shw_len", "shw_open_angle": "shw_open_angle",
            "shw_density": "shw_density", "shw_conversion_gap": "shw_conversion_gap", "shw_dEdx": "shw_bestplane_dEdx"}


class PfpSurrogate:
    def __init__(self, model_dir, device="cuda"):
        ck = torch.load(f"{model_dir}/model.pt", map_location=device, weights_only=False); self.device = device
        self.flow = PfpFlow(ck["dim"], ck["cond_dim"], ck["d_model"], ck["n_layers"]).to(device); self.flow.load_state_dict(ck["state"]); self.flow.eval()
        self.pt = PfpTransform(); self.pt.loc, self.pt.scale = ck["pt_loc"], ck["pt_scale"]
        self.std_m, self.std_s = ck["std_m"], ck["std_s"]
        self.bt = BlockTransform(); self.bt.med, self.bt.mad = ck["bt_med"], ck["bt_mad"]
        self.classes = [tuple(c) for c in ck["topology_classes"]]; self.lookup = {t: i for i, t in enumerate(self.classes)}; self.n_topo = len(self.classes) + 1
        self.n_fit = MLPClassifier(ck["cond_dim"], N_MAX + 1, device=device); self.n_fit.net.load_state_dict(ck["n_fit_head"])
        self.n_empty = MLPClassifier(ck["cond_dim"], N_EMPTY_MAX + 1, device=device); self.n_empty.net.load_state_dict(ck["n_empty_head"])

    def cond(self, F, slc):
        """F: features rows (aligned with slc rows); slc: generated slice records (production columns)."""
        C = ((np.arcsinh(F.to_numpy(np.float32)) - self.std_m) / self.std_s).astype(np.float32)
        tup = slc[TOPO_COLS].fillna(-1).astype(int).apply(tuple, axis=1); k = tup.map(self.lookup).fillna(len(self.classes)).astype(int).to_numpy()
        H = np.zeros((len(k), self.n_topo), dtype=np.float32); H[np.arange(len(k)), k] = 1
        Y = self.bt.matrix(slc).astype(np.float32); Z, Mfin = self.bt.standardise(Y)
        P = {"short": np.isfinite(slc["short_trk_len"].to_numpy()), "two": np.isfinite(slc["reco_t"].to_numpy())}
        for r in (1, 2, 3, 4): P[f"s{r}"] = np.isfinite(slc[f"sum_integ_{r}cm"].to_numpy()) & (slc[f"sum_integ_{r}cm"].to_numpy() > 0)
        Zc = Z * (mask_from_presence(P, self.bt) * Mfin)
        return np.concatenate([C, H, Zc], 1).astype(np.float32)

    @torch.no_grad()
    def generate(self, F, slc, seed=0, n_steps=64, bs=8192):
        """-> pfp DataFrame (index: slc index + rec.slc.reco.pfp..index) and per-slice n_empty."""
        rng = np.random.default_rng(seed); cond = self.cond(F, slc)
        Pn = self.n_fit.predict_proba(cond); nf = (Pn.cumsum(1) > rng.random(len(Pn))[:, None]).argmax(1)
        Pe = self.n_empty.predict_proba(cond); ne = (Pe.cumsum(1) > rng.random(len(Pe))[:, None]).argmax(1)
        pm = np.arange(N_MAX)[None, :] < nf[:, None]
        Z = np.concatenate([self.flow.sample(torch.tensor(cond[i:i + bs], device=self.device), torch.tensor(pm[i:i + bs], device=self.device), n_steps=n_steps).cpu().numpy() for i in range(0, len(cond), bs)]) if len(cond) else np.zeros((0, N_MAX, len(FIELDS)), np.float32)
        X = self.pt.inverse(Z, pm)
        rows, slots = np.nonzero(pm)
        out = {PFP_COLS[n]: X[rows, slots, NAMES.index(n)] for n in PFP_COLS}
        out["parent_is_primary"] = np.rint(out["parent_is_primary"]).astype(np.int8)
        has_trk, has_shw = X[rows, slots, NAMES.index("has_trk")] > 0.5, X[rows, slots, NAMES.index("has_shw")] > 0.5
        for a in "xyz":   # shower start = the same generated start point
            out[f"shw_start_{a}"] = np.where(has_shw, out[f"trk_start_{a}"], np.nan)
            out[f"trk_start_{a}"] = np.where(has_trk, out[f"trk_start_{a}"], np.nan)
            bend = X[rows, slots, NAMES.index(f"bend_{a}")] if f"bend_{a}" in NAMES and X.shape[-1] == len(NAMES) else 0.0
            out[f"trk_end_{a}"] = np.where(has_trk, out[f"trk_start_{a}"] + out["trk_len"] * out[f"trk_dir_{a}"] + bend, np.nan)   # straight line + generated bend
        out["surrogate_slot"] = slots
        idx = pd.MultiIndex.from_arrays([slc.index.get_level_values(i)[rows] for i in range(slc.index.nlevels)] + [slots], names=list(slc.index.names) + ["rec.slc.reco.pfp..index"])
        return pd.DataFrame(out, index=idx), pd.Series(ne, index=slc.index, name="n_empty_pfp")
