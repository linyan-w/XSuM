"""Selection-free dataframes for the XSuM sim-to-reco surrogate.

Reuses the cohpi definitions (fiducial volume, signal, two-track variables, chi2 re-calibration) from
analysis_village/cohpi/makedf/make_cohpidf_all_weight_signal.py, but writes every neutrino, every primary,
every slice and every PFP, with the cohpi selection stages stored as boolean columns instead of applied.
"""
import numpy as np
import pandas as pd

from makedf.makedf import (make_mcdf, make_mcprimdf, make_mcprimdaughtersdf, make_mchdrdf, make_pandora_df, make_opflashdf,
                           make_trkhitdf_plane2, PDG)
from makedf.branches import barycenterFMbranches, shwbranches, trueparticlebranches
from makedf.util import dmagdf, magdf
from pyanalib.pandas_helpers import loadbranches, multicol_merge, multicol_add
from analysis_village.cohpi.makedf.make_cohpidf_all_weight_signal import (
    InFV_nohiyz, Signal, is_contained, measure_reco_t, measure_opening_angle, get_true_t, make_slc_var)

# cohpi selection constants (make_cohpidf_slc step 9 and ana_syst_dfs.ipynb)
PE_CUT = 1500.
BCFM_SCORE_CUT = 0.05
LONG_DIRZ_CUT = 0.7
SHORT_DIRZ_CUT = 0.5
OPEN_ANG_CUT = 0.5
LONG_CHI2_P_CUT = 100.
SHORT_CHI2_P_CUT = 75.
SPHERE_CUTS = {"sum_integ_1cm": 4000., "sum_integ_2cm": 7000., "sum_integ_3cm": 10000., "sum_integ_4cm": 13000.}
SPHERE_RATIO_CUT = 6.
RECO_T_SR = 0.04   # signal region reco_t < 0.04, sideband > 0.06


def flatten(df):
    """Join non-empty MultiIndex column levels with '_' -> plain string columns."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["_".join(str(l) for l in c if str(l) != "") for c in out.columns]
    else:
        out.columns = [str(c) for c in out.columns]
    return out


def _pad(col, n):
    col = tuple(col) if isinstance(col, (tuple, list)) else (col,)
    return col + ("",) * (n - len(col))


def pick(df, spec):
    """spec: {new_name: column tuple}. Missing columns are reported once and skipped."""
    n = df.columns.nlevels
    out = {}
    missing = []
    for name, col in spec.items():
        key = _pad(col, n)
        if key in df.columns:
            out[name] = df[key]
        else:
            missing.append(name)
    if missing:
        print("make_xsumdf: columns not found, skipped:", missing)
    return pd.DataFrame(out, index=df.index)


def _bool(s):
    return s.fillna(False).astype(bool)


# ----------------------------------------------------------------------------------------------- truth
def make_xsum_nudf(f):
    mcdf = make_mcdf(f)
    if mcdf.empty:
        return pd.DataFrame()
    is_fv = InFV_nohiyz(mcdf.position)
    is_signal = Signal(mcdf)
    is_cc = mcdf.iscc
    genie_mode = mcdf.genie_mode
    w = mcdf.w
    categ = pd.Series(8, index=mcdf.index)
    categ[~is_fv] = -1
    categ[is_fv & is_signal] = 1
    categ[is_fv & ~is_cc & ~is_signal] = 0
    categ[is_fv & is_cc & ~is_signal & (genie_mode == 3)] = 2
    categ[is_fv & is_cc & ~is_signal & (genie_mode == 0)] = 3
    categ[is_fv & is_cc & ~is_signal & (genie_mode == 10)] = 4
    categ[is_fv & is_cc & ~is_signal & (genie_mode != 0) & (genie_mode != 3) & (genie_mode != 10) & ((w < 1.4) | (genie_mode == 1))] = 5
    categ[is_fv & is_cc & ~is_signal & (genie_mode != 0) & (genie_mode != 3) & (genie_mode != 10) & ((w > 2.0) | (genie_mode == 2))] = 6

    out = flatten(mcdf)
    p_mu = magdf(mcdf.mu.genp)
    p_pi = magdf(mcdf.cpi.genp)
    out["true_E_nu"] = mcdf.E
    out["true_p_mu"] = p_mu
    out["true_p_pi"] = p_pi
    out["true_cos_theta_mu"] = mcdf.mu.genp.z / p_mu
    out["true_cos_theta_pi"] = mcdf.cpi.genp.z / p_pi
    out["true_t"] = get_true_t(mcdf).fillna(999999)
    out["nuint_categ"] = categ
    out["is_fv"] = is_fv
    out["is_signal_cohpi"] = is_signal
    out["ind"] = out.index.get_level_values(1)
    return out


def make_xsum_primdf(f):
    df = make_mcprimdf(f)
    while df.columns.nlevels > 2:
        df.columns = df.columns.droplevel(0)
    df = flatten(df)
    if not df.empty:
        df["ke"] = np.nan
        for name, (pdg, _, m) in {k: v for k, v in PDG.items()}.items():
            sel = np.abs(df.pdg) == pdg
            df.loc[sel, "ke"] = df.genE[sel] - m
    return df


# ------------------------------------------------------------------------------------------------ reco
_PANDORA_CACHE = {}


def _pandora(f):
    """Calo-updated pandora df of every PFP of every non-clear-cosmic slice (no vertex-distance cut).

    cafpyana applies all df functions to the same open file in one process, so the expensive calo update is
    computed once per file and shared between the slc and pfp tables.
    """
    key = getattr(f, "file_path", None) or id(f)
    if _PANDORA_CACHE.get("key") == key:
        return _PANDORA_CACHE["df"]
    pandora_df = make_pandora_df(f, trkScoreCut=False, trkDistCut=0, cutClearCosmic=True, requireFiducial=False, updatecalo=True)
    if not pandora_df.empty:
        bcfm = loadbranches(f["recTree"], barycenterFMbranches).rec
        pandora_df = multicol_merge(bcfm, pandora_df, left_index=True, right_index=True, how="right", validate="one_to_many")
        pandora_df[("pfp", "trk", "chi2pid", "I2", "mu_over_p", "")] = pandora_df.pfp.trk.chi2pid.I2.chi2_muon / pandora_df.pfp.trk.chi2pid.I2.chi2_proton
        pandora_df[("pfp", "trk", "chi2pid", "I2", "arctan_mu_over_p", "")] = ((2 / np.pi) * np.arctan(pandora_df.pfp.trk.chi2pid.I2.mu_over_p / 0.8)).fillna(-0.2)
        near = (pandora_df.pfp.dist_to_vertex < 6.) & (pandora_df.pfp.trk.len > 4.)
        pandora_df[("pfp", "is_prong_candidate", "", "", "", "")] = near
        pandora_df[("pfp", "is_trk_candidate", "", "", "", "")] = near & (pandora_df.pfp.trackScore > 0.5)
        pandora_df[("pfp", "is_proton_candidate", "", "", "", "")] = near & (pandora_df.pfp.trackScore > 0.5) & (pandora_df.pfp.trk.chi2pid.I2.chi2_muon_new > 25.)
        pandora_df[("pfp", "is_shower_candidate", "", "", "", "")] = pandora_df.pfp.trackScore < 0.50
    _PANDORA_CACHE.clear()
    _PANDORA_CACHE.update({"key": key, "df": pandora_df})
    return pandora_df


def make_xsum_slcdf(f):
    all_df = _pandora(f)
    if all_df.empty:
        return pd.DataFrame()
    g = lambda s: make_slc_var(s)

    slcdf = pd.DataFrame({
        "is_fv": g(InFV_nohiyz(all_df.slc.vertex)),
        "bcfm_score": g(all_df.slc.barycenterFM.score),
        "nu_score": g(all_df.slc.nu_score),
        "is_not_clear_cosmic": g(all_df.slc.is_clear_cosmic == 0),
        "n_pfp": all_df.groupby(level=[0, 1]).size(),
        "n_pfp_50cm": (all_df.pfp.dist_to_vertex < 50.).groupby(level=[0, 1]).sum(),
        "n_prong": all_df.pfp.is_prong_candidate.groupby(level=[0, 1]).sum(),
        "n_trk": all_df.pfp.is_trk_candidate.groupby(level=[0, 1]).sum(),
        "n_proton": all_df.pfp.is_proton_candidate.groupby(level=[0, 1]).sum(),
        "n_shower": all_df.pfp.is_shower_candidate.groupby(level=[0, 1]).sum(),
        "tmatch_idx": g(all_df.slc.tmatch.idx),
        "tmatch_eff": g(all_df.slc.tmatch.eff),
        "tmatch_purity": g(all_df.slc.tmatch.pur),
        "vtx_x": g(all_df.slc.vertex.x),
        "vtx_y": g(all_df.slc.vertex.y),
        "vtx_z": g(all_df.slc.vertex.z),
    })
    # the cohpi two-track block is defined on PFPs within 50 cm of the vertex (make_pandora_df_calo_update)
    pandora_df = all_df[all_df.pfp.dist_to_vertex < 50.]

    # two longest track candidates, ordered by length (cohpi: muon = longer)
    order = pandora_df.sort_values(by=("pfp", "trk", "len", "", "", ""), ascending=False)
    longdf = order.groupby(level=[0, 1]).nth(0)
    shortdf = order.groupby(level=[0, 1]).nth(1)
    for tag, d in (("long", longdf), ("short", shortdf)):
        spec = {
            f"{tag}_dirx": ("pfp", "trk", "dir", "x"), f"{tag}_diry": ("pfp", "trk", "dir", "y"), f"{tag}_dirz": ("pfp", "trk", "dir", "z"),
            f"{tag}_trk_len": ("pfp", "trk", "len"),
            f"{tag}_trk_pid": ("pfp", "trk", "chi2pid", "I2", "arctan_mu_over_p"),
            f"{tag}_trk_chi2_mu": ("pfp", "trk", "chi2pid", "I2", "chi2_muon"),
            f"{tag}_trk_chi2_pro": ("pfp", "trk", "chi2pid", "I2", "chi2_proton"),
            f"{tag}_trk_chi2_mu_new": ("pfp", "trk", "chi2pid", "I2", "chi2_muon_new"),
            f"{tag}_trk_chi2_pro_new": ("pfp", "trk", "chi2pid", "I2", "chi2_proton_new"),
            f"{tag}_pfp_index": ("pfp", "tindex"),
        }
        slcdf = slcdf.join(pick(d, spec).groupby(level=[0, 1]).first())

    # two-track variables on the cohpi two-track topology (NaN elsewhere)
    two = pandora_df[(slcdf.reindex(pandora_df.index.droplevel(2)).eval("n_prong == 2 and n_trk == 2 and n_proton == 0 and n_shower == 0").values)
                     & pandora_df.pfp.is_trk_candidate.values]
    if not two.empty:
        slcdf["reco_t"] = two.groupby(level=[0, 1]).apply(measure_reco_t)
        slcdf["opening_angle"] = two.groupby(level=[0, 1]).apply(measure_opening_angle)
        two_order = two.sort_values(by=("pfp", "trk", "len", "", "", ""), ascending=False)
        muondf = two_order.groupby(level=[0, 1]).nth(0)
        cpidf = two_order.groupby(level=[0, 1]).nth(1)
        slcdf["is_muon_contained"] = g(is_contained(muondf)).astype(float)
        slcdf["is_cpi_contained"] = g(is_contained(cpidf)).astype(float)
        for tag, d in (("muon", muondf), ("cpi", cpidf)):
            spec = {f"{tag}_{e}_{a}": ("pfp", "trk", e, a) for e in ("start", "end") for a in ("x", "y", "z")}
            slcdf = slcdf.join(pick(d, spec).groupby(level=[0, 1]).first())
        slcdf["range_p_mu"] = g(muondf.pfp.trk.rangeP.p_muon)
        slcdf["range_p_cpi"] = g(cpidf.pfp.trk.rangeP.p_muon)
        slcdf["range_p_cpi_pion"] = g(cpidf.pfp.trk.rangeP.p_pion)
    # columns that only exist when a two-track slice was found: make sure they are always present
    for c in (["reco_t", "opening_angle", "is_muon_contained", "is_cpi_contained", "range_p_mu", "range_p_cpi", "range_p_cpi_pion"]
              + [f"{t}_{e}_{a}" for t in ("muon", "cpi") for e in ("start", "end") for a in ("x", "y", "z")]
              + [f"{t}_{c}" for t in ("long", "short") for c in ("dirx", "diry", "dirz", "trk_len", "trk_pid", "trk_chi2_mu", "trk_chi2_pro", "trk_chi2_mu_new", "trk_chi2_pro_new", "pfp_index")]):
        if c not in slcdf.columns:
            slcdf[c] = np.nan

    # light: total PE of flashes within +-5 us of the beam
    opflash = make_opflashdf(f)
    opflash = opflash[(opflash.firsttime > -5.) & (opflash.firsttime < 5.)]
    totalpe = opflash.groupby(level=0)["totalpe"].sum()
    slcdf["total_pe"] = slcdf.index.get_level_values("entry").map(totalpe)

    # charge in spheres around the vertex (collection plane hits of PFPs within 50 cm)
    hitdf = make_trkhitdf_plane2(f)
    hitdf.columns = pd.MultiIndex.from_tuples([("pfp", "trk", "hit", col, "") for col in hitdf.columns])
    near = pandora_df[(pandora_df.pfp.dist_to_vertex < 50.) & (pandora_df.slc.is_clear_cosmic == 0)]
    hm = multicol_merge(hitdf.reset_index(), near.reset_index(),
                        left_on=[("entry", "", "", "", "", ""), ("rec.slc..index", "", "", "", "", ""), ("rec.slc.reco.pfp..index", "", "", "", "", "")],
                        right_on=[("entry", "", "", "", "", ""), ("rec.slc..index", "", "", "", "", ""), ("rec.slc.reco.pfp..index", "", "", "", "", "")],
                        how="right")
    hm = hm.set_index(["entry", "rec.slc..index", "rec.slc.reco.pfp..index", "rec.slc.reco.pfp.trk.calo.2.points..index"], verify_integrity=True)
    hm = multicol_add(hm, dmagdf(hm.slc.vertex, hm.pfp.trk.hit).rename(("pfp", "trk", "hit", "dist_to_vertex", "", "")))
    for r in (1, 2, 3, 4):
        slcdf[f"sum_integ_{r}cm"] = hm[hm.pfp.trk.hit.dist_to_vertex < r].pfp.trk.hit.integral.groupby(level=[0, 1]).sum()
    slcdf["sphere_3_over_4m3"] = slcdf.sum_integ_3cm / (slcdf.sum_integ_4cm - slcdf.sum_integ_3cm)

    # cohpi selection stages as flags (make_cohpidf_slc step 9 + notebook cuts)
    slcdf["pass_fv_cosmic"] = _bool(slcdf.is_fv & slcdf.is_not_clear_cosmic)
    slcdf["pass_pe_bcfm"] = _bool((slcdf.total_pe > PE_CUT) & (slcdf.bcfm_score > BCFM_SCORE_CUT))
    slcdf["pass_2trk"] = _bool((slcdf.n_prong == 2) & (slcdf.n_trk == 2) & (slcdf.n_proton == 0) & (slcdf.n_shower == 0))
    slcdf["pass_dir"] = _bool((slcdf.long_dirz > LONG_DIRZ_CUT) & (slcdf.short_dirz > SHORT_DIRZ_CUT) & (slcdf.opening_angle > OPEN_ANG_CUT))
    slcdf["pass_chi2"] = _bool((slcdf.long_trk_chi2_pro_new > LONG_CHI2_P_CUT) & (slcdf.short_trk_chi2_pro_new > SHORT_CHI2_P_CUT))
    sph = pd.Series(True, index=slcdf.index)
    for c, cut in SPHERE_CUTS.items():
        sph &= slcdf[c].fillna(0.) < cut
    sph &= slcdf.sphere_3_over_4m3.fillna(0.) < SPHERE_RATIO_CUT
    slcdf["pass_sphere"] = _bool(sph)
    mu_ok = (np.abs(slcdf.muon_start_x) > 10.) & (np.abs(slcdf.muon_end_x) > 10.) & (slcdf.muon_start_x * slcdf.muon_end_x > 0.)
    pi_ok = (np.abs(slcdf.cpi_start_x) > 10.) & (np.abs(slcdf.cpi_end_x) > 10.) & (slcdf.cpi_start_x * slcdf.cpi_end_x > 0.)
    slcdf["pass_cathode"] = _bool(mu_ok & pi_ok)
    slcdf["pass_cohpi"] = (slcdf.pass_fv_cosmic & slcdf.pass_pe_bcfm & slcdf.pass_2trk & slcdf.pass_dir & slcdf.pass_chi2
                           & slcdf.pass_sphere & slcdf.pass_cathode)
    slcdf["in_sr"] = _bool(slcdf.pass_cohpi & (slcdf.reco_t < RECO_T_SR))
    return slcdf


def make_xsum_pfpdf(f):
    pandora_df = _pandora(f)
    if pandora_df.empty:
        return pd.DataFrame()
    spec = {
        "trackScore": ("pfp", "trackScore"), "dist_to_vertex": ("pfp", "dist_to_vertex"),
        "parent_is_primary": ("pfp", "parent_is_primary"), "parent": ("pfp", "parent"), "pfp_id": ("pfp", "id"), "t0": ("pfp", "t0"),
        "is_prong_candidate": ("pfp", "is_prong_candidate"), "is_trk_candidate": ("pfp", "is_trk_candidate"),
        "is_proton_candidate": ("pfp", "is_proton_candidate"), "is_shower_candidate": ("pfp", "is_shower_candidate"),
        "trk_len": ("pfp", "trk", "len"), "trk_bestplane": ("pfp", "trk", "bestplane"),
        "trk_rangeP_mu": ("pfp", "trk", "rangeP", "p_muon"), "trk_rangeP_pi": ("pfp", "trk", "rangeP", "p_pion"), "trk_rangeP_p": ("pfp", "trk", "rangeP", "p_proton"),
        "trk_mcsP_mu": ("pfp", "trk", "mcsP", "fwdP_muon"), "trk_mcsP_pi": ("pfp", "trk", "mcsP", "fwdP_pion"), "trk_mcsP_p": ("pfp", "trk", "mcsP", "fwdP_proton"),
        "chi2_mu": ("pfp", "trk", "chi2pid", "I2", "chi2_muon"), "chi2_pro": ("pfp", "trk", "chi2pid", "I2", "chi2_proton"),
        "chi2_ndof": ("pfp", "trk", "chi2pid", "I2", "pid_ndof"), "pida": ("pfp", "trk", "chi2pid", "I2", "pida"),
        "pid_arctan_mu_over_p": ("pfp", "trk", "chi2pid", "I2", "arctan_mu_over_p"),
        "crt_dist": ("pfp", "trk", "crthit", "distance"),
    }
    for a in ("x", "y", "z"):
        for e in ("start", "end", "dir"):
            spec[f"trk_{e}_{a}"] = ("pfp", "trk", e, a)
    for pl in (0, 1, 2):
        spec[f"chi2_mu_new_I{pl}"] = ("pfp", "trk", "chi2pid", f"I{pl}", "chi2_muon_new")
        spec[f"chi2_pro_new_I{pl}"] = ("pfp", "trk", "chi2pid", f"I{pl}", "chi2_proton_new")
        spec[f"ndof_mu_new_I{pl}"] = ("pfp", "trk", "chi2pid", f"I{pl}", "ndof_muon_new")
    out = pick(pandora_df, spec)
    out["slc_is_fv"] = InFV_nohiyz(pandora_df.slc.vertex).values
    out["slc_tmatch_idx"] = pandora_df.slc.tmatch.idx.values

    # kaon-analysis PID: kaon and pion chi2 on all planes, muon and proton on the induction planes, kaon momenta
    kb = "rec.slc.reco.pfp.trk."
    try:
        kdf = loadbranches(f["recTree"], [kb + f"chi2pid.{p}.chi2_{h}" for p in (0, 1, 2) for h in ("kaon", "pion")]
                           + [kb + f"chi2pid.{p}.chi2_{h}" for p in (0, 1) for h in ("muon", "proton")]
                           + [kb + "mcsP.fwdP_kaon"]).rec.slc.reco
        kspec = {f"chi2_{h}_I{p}": ("pfp", "trk", "chi2pid", f"I{p}", f"chi2_{h}") for p in (0, 1, 2) for h in ("kaon", "pion")}
        kspec.update({f"chi2_{h[:3] if h == 'proton' else h[:2]}_I{p}": ("pfp", "trk", "chi2pid", f"I{p}", f"chi2_{h}") for p in (0, 1) for h in ("muon", "proton")})
        kspec["trk_mcsP_k"] = ("pfp", "trk", "mcsP", "fwdP_kaon")
        out = out.join(pick(kdf, kspec), how="left")
    except Exception as e:
        print("make_xsum_pfpdf: no kaon PID branches:", e)
    # (no rangeP.p_kaon branch in the v10_14 flat CAFs; twester derives it offline)

    # showers
    try:
        shw = loadbranches(f["recTree"], shwbranches).rec.slc.reco
        sspec = {"shw_len": ("pfp", "shw", "len"), "shw_open_angle": ("pfp", "shw", "open_angle"), "shw_density": ("pfp", "shw", "density"),
                 "shw_conversion_gap": ("pfp", "shw", "conversion_gap"), "shw_bestplane_energy": ("pfp", "shw", "bestplane_energy"),
                 "shw_bestplane_dEdx": ("pfp", "shw", "bestplane_dEdx")}
        for a in ("x", "y", "z"):
            sspec[f"shw_start_{a}"] = ("pfp", "shw", "start", a)
        out = out.join(pick(shw, sspec), how="left")
    except Exception as e:
        print("make_xsum_pfpdf: no shower branches:", e)

    # truth match of the PFP's track
    tb = "rec.slc.reco.pfp.trk.truth."
    try:
        tru = loadbranches(f["recTree"], [tb + s for s in ("p.pdg", "p.G4ID", "p.parent", "p.interaction_id", "p.genE", "p.length",
                                                           "p.start.x", "p.start.y", "p.start.z", "p.end.x", "p.end.y", "p.end.z",
                                                           "p.contained", "bestmatch.energy", "bestmatch.hit_completeness", "bestmatch.hit_purity")]).rec.slc.reco.pfp.trk.truth
        tru = flatten(tru)
        tru.columns = ["truth_" + c.replace("p_", "", 1) if c.startswith("p_") else "truth_" + c for c in tru.columns]
        out = out.join(tru, how="left")
    except Exception as e:
        print("make_xsum_pfpdf: no track truth branches:", e)
    return out


# ----------------------------------------------------------------------------------------------- Geant4 daughters
def make_xsum_daughtdf(f):
    """Geant4 daughters of every neutrino primary (one generation), from rec.true_particles.

    One row per (neutrino, primary, daughter): pdg, G4ID, parent, energies, momentum, start/end, length,
    containment, start/end process. Lets kaon and pion decay modes, pi0 photons and secondary interactions
    be derived later without rerunning cafpyana.
    """
    dd = make_mcprimdaughtersdf(f)
    if dd.empty:
        return pd.DataFrame()
    while dd.columns.nlevels > 1:
        dd.columns = dd.columns.droplevel(0)
    dd = flatten(dd)
    dd.columns = ["G4ID"]
    tp = loadbranches(f["recTree"], trueparticlebranches).rec.true_particles
    tp = flatten(tp)
    tp = tp.reset_index(level=list(range(1, tp.index.nlevels)), drop=True)   # keep 'entry' only
    out = dd.reset_index().merge(tp.reset_index(), on=["entry", "G4ID"], how="left")
    out = out.set_index(list(dd.index.names))
    return out
