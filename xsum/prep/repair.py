"""Rebuild the cohpi slice-level two-track block from the `pfp` table (offline repair of the official production).

The second module version counted showers among all PFPs of a slice instead of those within 50 cm of the vertex,
so the two-track topology never occurred and reco_t, opening_angle, containment, muon/cpi start/end points,
range momenta and the pass_* flags are empty in `prod_v10_14_02_03_official`. Everything needed is in `pfp`.
This reproduces make_cohpidf_slc / make_xsumdf step by step (including the muon = longer track convention).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

M_MU, M_PI = 0.105, 0.13957039   # masses used by cafpyana's reco_t (PDG table in makedf.makedf)
PE_CUT, BCFM_CUT, LONG_DIRZ, SHORT_DIRZ, OPEN_ANG, LONG_CHI2P, SHORT_CHI2P = 1500., 0.05, 0.7, 0.5, 0.5, 100., 75.
SPHERE = {"sum_integ_1cm": 4000., "sum_integ_2cm": 7000., "sum_integ_3cm": 10000., "sum_integ_4cm": 13000.}


def in_fv_trk(x, y, z):
    pass_xz = (np.abs(x) < 190.) & (z > 10.) & (z < 450.)
    pass_y = ((z < 250) & (np.abs(y) < 190.)) | ((z > 250) & (y > -190.) & (y < 100.))
    return pass_xz & pass_y


def repair_slc(slc: pd.DataFrame, pfp: pd.DataFrame) -> pd.DataFrame:
    """slc, pfp: one production file's tables (any leading index levels, last levels (entry, slc[, pfp]))."""
    out = slc.copy()
    lv = list(slc.index.names)                       # slice key levels
    p = pfp[pfp.dist_to_vertex < 50.]
    near = (p.dist_to_vertex < 6.) & (p.trk_len > 4.)
    trk = near & (p.trackScore > 0.5)
    g = lambda s: s.groupby(level=lv).sum()
    cnt = pd.DataFrame({"n_prong": g(near), "n_trk": g(trk), "n_proton": g(trk & (p.chi2_mu_new_I2 > 25.)), "n_shower": g(p.trackScore < 0.5)})
    for c in cnt.columns:
        out[c] = cnt[c].reindex(out.index).fillna(0).astype(int)
    two_slc = out.index[(out.n_prong == 2) & (out.n_trk == 2) & (out.n_proton == 0) & (out.n_shower == 0) & out.is_fv.astype(bool)]
    t = p[trk & p.index.droplevel(-1).isin(two_slc)].copy()          # the two track candidates of each two-track slice
    t["_order"] = t.groupby(level=lv).cumcount()                       # pfp order, as cafpyana's groupby.apply saw it
    first, second = t[t._order == 0], t[t._order == 1]
    first = first.droplevel(-1); second = second.droplevel(-1)
    common = first.index.intersection(second.index); first, second = first.loc[common], second.loc[common]
    # reco |t| exactly as cafpyana: first = muon, second = pion, swapped when range_P_muon[0] > range_P_muon[1]
    swap = first.trk_rangeP_mu > second.trk_rangeP_mu
    m0 = np.where(swap, M_PI, M_MU); m1 = np.where(swap, M_MU, M_PI)
    p0 = np.where(swap, first.trk_rangeP_pi, first.trk_rangeP_mu); p1 = np.where(swap, second.trk_rangeP_mu, second.trk_rangeP_pi)
    E0, E1 = np.sqrt(m0**2 + p0**2), np.sqrt(m1**2 + p1**2)
    px = (p0 * first.trk_dir_x + p1 * second.trk_dir_x) ** 2; py = (p0 * first.trk_dir_y + p1 * second.trk_dir_y) ** 2
    pz = (E0 + E1 - p0 * first.trk_dir_z - p1 * second.trk_dir_z) ** 2
    reco_t = pd.Series(px + py + pz, index=common)
    opening = first.trk_dir_x * second.trk_dir_x + first.trk_dir_y * second.trk_dir_y + first.trk_dir_z * second.trk_dir_z
    # muon = longer track candidate, pion = shorter
    longer_first = first.trk_len >= second.trk_len
    mu = first.where(longer_first, second); cpi = second.where(longer_first, first)
    newcols = {"reco_t": reco_t, "opening_angle": opening,
               "is_muon_contained": (in_fv_trk(mu.trk_start_x, mu.trk_start_y, mu.trk_start_z) & in_fv_trk(mu.trk_end_x, mu.trk_end_y, mu.trk_end_z)).astype(float),
               "is_cpi_contained": (in_fv_trk(cpi.trk_start_x, cpi.trk_start_y, cpi.trk_start_z) & in_fv_trk(cpi.trk_end_x, cpi.trk_end_y, cpi.trk_end_z)).astype(float),
               "range_p_mu": mu.trk_rangeP_mu, "range_p_cpi": cpi.trk_rangeP_mu, "range_p_cpi_pion": cpi.trk_rangeP_pi}
    for tag, d in (("muon", mu), ("cpi", cpi)):
        for e in ("start", "end"):
            for a in ("x", "y", "z"):
                newcols[f"{tag}_{e}_{a}"] = d[f"trk_{e}_{a}"]
    for c, v in newcols.items():
        out[c] = np.nan; out.loc[common, c] = v.to_numpy()
    # selection flags as in make_xsumdf
    out["pass_fv_cosmic"] = (out.is_fv.astype(bool) & out.is_not_clear_cosmic.astype(bool))
    out["pass_pe_bcfm"] = ((out.total_pe > PE_CUT) & (out.bcfm_score > BCFM_CUT)).fillna(False)
    out["pass_2trk"] = (out.n_prong == 2) & (out.n_trk == 2) & (out.n_proton == 0) & (out.n_shower == 0)
    out["pass_dir"] = ((out.long_dirz > LONG_DIRZ) & (out.short_dirz > SHORT_DIRZ) & (out.opening_angle > OPEN_ANG)).fillna(False)
    out["pass_chi2"] = ((out.long_trk_chi2_pro_new > LONG_CHI2P) & (out.short_trk_chi2_pro_new > SHORT_CHI2P)).fillna(False)
    sph = pd.Series(True, index=out.index)
    for c, cut in SPHERE.items(): sph &= out[c].fillna(0.) < cut
    sph &= out.sphere_3_over_4m3.fillna(0.) < 6.
    out["pass_sphere"] = sph
    mu_ok = (out.muon_start_x.abs() > 10.) & (out.muon_end_x.abs() > 10.) & (out.muon_start_x * out.muon_end_x > 0.)
    pi_ok = (out.cpi_start_x.abs() > 10.) & (out.cpi_end_x.abs() > 10.) & (out.cpi_start_x * out.cpi_end_x > 0.)
    out["pass_cathode"] = (mu_ok & pi_ok).fillna(False)
    out["pass_cohpi"] = out.pass_fv_cosmic & out.pass_pe_bcfm & out.pass_2trk & out.pass_dir & out.pass_chi2 & out.pass_sphere & out.pass_cathode
    out["in_sr"] = (out.pass_cohpi & (out.reco_t < 0.04)).fillna(False)
    for c in ["pass_fv_cosmic", "pass_pe_bcfm", "pass_2trk", "pass_dir", "pass_chi2", "pass_sphere", "pass_cathode", "pass_cohpi", "in_sr"]:
        out[c] = out[c].astype(bool)
    return out
