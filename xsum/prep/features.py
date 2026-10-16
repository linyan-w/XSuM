"""Fixed-size, detector-causal event features from the truth tables (M1 baselines).

Inputs are only what enters the detector: the true vertex and the post-FSI primary particles (species,
momentum). Generator labels (genie_mode, nuint_categ, W, Q2, E_nu) are never used.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from xsum.constants import MCNU_INDEX

MASS = {"mu": 0.1056584, "e": 0.000511, "p": 0.938272, "n": 0.9395654, "pi": 0.13957039, "pi0": 0.1349768,
        "g": 0.0, "k": 0.493677, "k0": 0.497611, "lam": 1.115683, "sig": 1.18937}
SPECIES = {"mu": (13,), "e": (11,), "p": (2212,), "n": (2112,), "pi": (211,), "pi0": (111,), "g": (22,),
           "k": (321,), "k0": (311, 310, 130), "lam": (3122,), "sig": (3112, 3222, 3212)}
# kinetic-energy thresholds (GeV) used for the "visible" counts, as in the cohpi signal definition
KE_THRESH = {"mu": 0.027, "e": 0.0, "p": 0.020, "n": 0.0, "pi": 0.030, "pi0": 0.0, "g": 0.0, "k": 0.0, "k0": 0.0, "lam": 0.0, "sig": 0.0}
# SBND active volume (cm): x in [-200, 200] with the cathode at x = 0, y in [-200, 200], z in [0, 500]
AV = {"x": 200.0, "y": 200.0, "zmin": 0.0, "zmax": 500.0}


def primary_table(mcprim: pd.DataFrame) -> pd.DataFrame:
    """Per-primary derived quantities: species class, momentum, KE, cos theta."""
    pdg = mcprim["pdg"].to_numpy()
    apdg = np.abs(pdg)
    cls = np.full(len(mcprim), "other", dtype=object)
    mass = np.full(len(mcprim), np.nan)
    for name, codes in SPECIES.items():
        m = np.isin(apdg, codes)
        cls[m] = name
        mass[m] = MASS[name]
    p = np.sqrt(mcprim.genp_x**2 + mcprim.genp_y**2 + mcprim.genp_z**2).to_numpy()
    e = mcprim["genE"].to_numpy()
    ke = np.where(np.isfinite(mass), e - mass, np.nan)
    out = pd.DataFrame({"cls": cls, "p": p, "ke": ke, "cosz": np.where(p > 0, mcprim.genp_z.to_numpy() / np.where(p > 0, p, 1), 0.0),
                        "pt": np.sqrt(mcprim.genp_x**2 + mcprim.genp_y**2).to_numpy(), "charge": np.sign(pdg) * np.isin(apdg, (13, 11, 211, 321, 2212, 3112, 3222))},
                       index=mcprim.index)
    return out


def event_features(mcnu: pd.DataFrame, mcprim: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """One row per neutrino (index = mcnu.index). Columns are plain float32."""
    nu_levels = list(mcnu.index.names)
    pt = primary_table(mcprim)
    pt = pt[pt.cls != "other"]
    pt_nu = pt.index.droplevel(-1)
    thr = pt.cls.map(KE_THRESH).to_numpy()
    vis = pt.ke.to_numpy() > thr
    feats = {}
    # vertex and geometry
    feats["vtx_x"] = mcnu.position_x; feats["vtx_y"] = mcnu.position_y; feats["vtx_z"] = mcnu.position_z
    feats["dist_cathode"] = np.abs(mcnu.position_x)
    feats["dist_wall_x"] = AV["x"] - np.abs(mcnu.position_x)
    feats["dist_wall_y"] = AV["y"] - np.abs(mcnu.position_y)
    feats["dist_front"] = mcnu.position_z - AV["zmin"]
    feats["dist_back"] = AV["zmax"] - mcnu.position_z
    # per species
    for name in SPECIES:
        s = pt[(pt.cls == name) & vis]
        g = s.groupby(level=nu_levels)
        feats[f"n_{name}"] = g.size()
        feats[f"sumke_{name}"] = g.ke.sum()
        feats[f"maxke_{name}"] = g.ke.max()
        lead = s.sort_values("ke", ascending=False).groupby(level=nu_levels).head(1).droplevel(-1)
        feats[f"lead_p_{name}"] = lead.p
        feats[f"lead_cosz_{name}"] = lead.cosz
        feats[f"lead_pt_{name}"] = lead.pt
    # totals
    chg = pt[vis & (pt.charge != 0)]
    gq = chg.groupby(level=nu_levels)
    feats["n_charged"] = gq.size()
    feats["sumke_charged"] = gq.ke.sum()
    neutral_vis = pt[vis & pt.cls.isin(["pi0", "g"])]
    feats["sumE_em"] = neutral_vis.groupby(level=nu_levels).ke.sum()
    feats["sumke_visible"] = pt[vis & (pt.cls != "n")].groupby(level=nu_levels).ke.sum()
    feats["sumke_all"] = pt.groupby(level=nu_levels).ke.sum()
    pz = (pt.p * pt.cosz).groupby(level=nu_levels).sum()
    feats["sum_pz"] = pz
    df = pd.DataFrame(feats, index=mcnu.index)
    df = df.fillna(0.0).astype(np.float32)
    return df, list(df.columns)
