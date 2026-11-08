#!/usr/bin/env python3
"""Acceptance test: the collaborator's CC1p0pi cut flow (M. Fernandez, xsum_cc1p0pi_start.ipynb, run_cc1p0pi),
real (Pandora) vs XSuM surrogate, on the held-out test files, with efficiency, purity and the real/surrogate overlap
of selected neutrinos per stage.

Two signal definitions are counted. Primary (XSuM acceptance): the notebook's topology, momentum caps and true
vertex in the FV (outside the cathode band), without the truth containment of the muon and proton end points. A truth-containment boundary is sharp where
the end point is set by stochastic processes, so events just inside and just outside it are interchangeable; any
end-point imperfection then moves efficiency against one definition while the reco containment cut still selects
the same mixture (reports/cc1p_containment/README.md). Secondary: the notebook's definition with truth containment.

Differences to the notebook, by necessity: the notebook's private FV helpers (InFV det="SBND_Gen1",
PER_TPC_INCATHODE_CM) are not in any cafpyana branch; this script uses cafpyana's SBND FV (|x|,|y| < 190,
10 < z < 450) and a 10 cm per-TPC cathode margin, identically for both samples. Restricted to test files.

usage: python3 check_cc1p0pi_stages.py <surrogate dir> <first job> <last job + 1>   -> prints one CSV-like block
(stage, real sel, real sig, XSuM sel, XSuM sig, both, real sig_topo, XSuM sig_topo, both sig_topo)
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "/pnfs/sbnd/persistent/users/lwan/dataframe")
import xsum_testsplit as ts

CATHODE = 10.0


def fv(x, y, z):
    x, y, z = (np.asarray(v, dtype=float) for v in (x, y, z))
    return (x > -190) & (x < 190) & (y > -190) & (y < 190) & (z > 10) & (z < 450)


def per_tpc(points):
    """points: list of (x,y,z) arrays -> all in FV and all on one side of the cathode with margin."""
    allfv = np.ones(len(points[0][0]), bool); t1 = allfv.copy(); t2 = allfv.copy()
    for x, y, z in points:
        x = np.asarray(x, float); allfv &= fv(x, y, z); t1 &= x < -CATHODE; t2 &= x > CATHODE
    return allfv & (t1 | t2)


def truth_signal(mcnu, mcprim, contained=True):
    prim = mcprim.copy(); prim["p"] = np.sqrt(prim.genp_x**2 + prim.genp_y**2 + prim.genp_z**2)
    lv = list(mcnu.index.names); g = lambda m: m.groupby(level=lv).sum().reindex(mcnu.index).fillna(0)
    nmu = g((prim.pdg.abs() == 13) & (prim.p > 0.22)); npr = g((prim.pdg == 2212) & (prim.p > 0.30)); npi = g((prim.pdg.abs() == 211) & (prim.p > 0.07))
    topo = (mcnu.pdg == 14) & (mcnu.iscc == 1) & (nmu == 1) & (npr == 1) & (npi == 0) & (mcnu.npi0 == 0) & (mcnu.mu_totp < 1.0) & (mcnu.p_totp < 1.0)
    if not contained: return topo.to_numpy(bool) & per_tpc([(mcnu.position_x, mcnu.position_y, mcnu.position_z)])
    cont = per_tpc([(mcnu.position_x, mcnu.position_y, mcnu.position_z), (mcnu.mu_end_x, mcnu.mu_end_y, mcnu.mu_end_z), (mcnu.p_end_x, mcnu.p_end_y, mcnu.p_end_z)])
    return topo.to_numpy(bool) & cont


def stages(slc, pfp):
    """slc indexed (job, __ntuple, entry, rec.slc..index); pfp the same + pfp index. Returns dict stage -> set of neutrino ids."""
    s = slc.copy(); lv = list(s.index.names)
    good = (pfp.trk_len > 0) & (pfp.dist_to_vertex < 100); p = pfp[good].copy()
    s["n_good"] = p.groupby(level=lv).size().reindex(s.index).fillna(0).astype(int)
    p = p.sort_values("trk_len", ascending=False); p["rank"] = p.groupby(level=lv).cumcount()
    cols = ["trk_len", "trk_start_x", "trk_start_y", "trk_start_z", "trk_end_x", "trk_end_y", "trk_end_z", "trackScore", "dist_to_vertex", "chi2_mu", "chi2_pro", "trk_rangeP_mu", "trk_mcsP_mu"]
    for r in (0, 1):
        t = p[p["rank"] == r][cols]; t.index = t.index.droplevel(-1); t.columns = [f"t{r+1}_{c}" for c in cols]
        s = s.join(t, how="left")
    st = {}
    st["allreco"] = np.ones(len(s), bool)
    st["vertex_in_fv"] = st["allreco"] & fv(s.vtx_x, s.vtx_y, s.vtx_z) & s.is_not_clear_cosmic.fillna(False).astype(bool).to_numpy()
    st["nu_score"] = st["vertex_in_fv"] & (s.nu_score > 0.45).to_numpy()
    st["2prong"] = st["nu_score"] & (s.n_good == 2).to_numpy()
    pts = [(s.vtx_x, s.vtx_y, s.vtx_z)] + [(s[f"t{i}_trk_{e}_x"], s[f"t{i}_trk_{e}_y"], s[f"t{i}_trk_{e}_z"]) for i in (1, 2) for e in ("start", "end")]
    st["2prong-contained"] = st["2prong"] & per_tpc(pts)
    st["2prong-trackscore"] = st["2prong-contained"] & ((s.t1_trackScore > 0.5) & (s.t2_trackScore > 0.5)).to_numpy()
    st["2prong-vtxdist"] = st["2prong-trackscore"] & ((s.t1_dist_to_vertex < 1.2) & (s.t2_dist_to_vertex < 1.2)).to_numpy()
    mu, pr = {}, {}
    for i in (1, 2):
        q = np.abs((s[f"t{i}_trk_rangeP_mu"] - s[f"t{i}_trk_mcsP_mu"]) / s[f"t{i}_trk_rangeP_mu"])
        mu[i] = ((s[f"t{i}_chi2_mu"] > 0) & (s[f"t{i}_chi2_mu"] < 30) & (s[f"t{i}_chi2_pro"] > 100) & (s[f"t{i}_trk_len"] > 50) & (q < 0.2)
                 & (s[f"t{i}_trk_rangeP_mu"] > 0.22) & (s[f"t{i}_trk_rangeP_mu"] < 1.0)).to_numpy()
        pr[i] = ((s[f"t{i}_chi2_pro"] > 0) & (s[f"t{i}_chi2_pro"] < 90) & (s[f"t{i}_trk_len"] > 0)).to_numpy()
    st["2prong-muX"] = st["2prong-vtxdist"] & (mu[1] | mu[2])
    st["2prong-mup-PID"] = st["2prong-muX"] & ((mu[1] & pr[2]) | (mu[2] & pr[1]))
    nu = pd.MultiIndex.from_arrays([s.index.get_level_values("job"), s.index.get_level_values("__ntuple"), s.index.get_level_values("entry"), s.tmatch_idx.fillna(-1).astype(int).to_numpy()])
    return {k: set(nu[m]) for k, m in st.items()}


def best_matched(slc, mcnu):
    """Real side: slices truth-matched to an mcnu neutrino, keep the highest tmatch_eff per neutrino (as the notebook)."""
    s = slc[slc.tmatch_idx.notna() & (slc.tmatch_idx >= 0) & slc.tmatch_eff.notna()].copy()
    key = pd.MultiIndex.from_arrays([s.index.get_level_values("job"), s.index.get_level_values("__ntuple"), s.index.get_level_values("entry"), s.tmatch_idx.astype(int).to_numpy()])
    s = s[key.isin(mcnu.index)]
    s["_k"] = list(zip(s.index.get_level_values("job"), s.index.get_level_values("__ntuple"), s.index.get_level_values("entry"), s.tmatch_idx.astype(int)))
    s = s.sort_values("tmatch_eff", ascending=False); s = s[~s["_k"].duplicated()].drop(columns="_k").sort_index()
    return s


if __name__ == "__main__":
    surr_dir, j0, j1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    ts.SURR = f"/pnfs/sbnd/persistent/users/lwan/dataframe/{surr_dir}/xsum_surrogate_{{n}}.df"
    jobs = range(j0, j1)
    mcnu = ts.load_test("real", "mcnu", jobs); mcprim = ts.load_test("real", "mcprim", jobs)
    sig = truth_signal(mcnu, mcprim); sig_ids = set(mcnu.index[sig]); all_ids = set(mcnu.index)
    topo_ids = set(mcnu.index[truth_signal(mcnu, mcprim, contained=False)])
    rslc = best_matched(ts.load_test("real", "slc", jobs, matched_only=False), mcnu)
    rpfp = ts.load_test("real", "pfp", jobs, matched_only=False)
    rpfp = rpfp[rpfp.index.droplevel(-1).isin(rslc.index) & (rpfp.trk_len.notna() | ((rpfp.shw_bestplane_energy > -4) & (rpfp.shw_len > 0)))]
    sslc = ts.load_test("surrogate", "slc", jobs); spfp = ts.load_test("surrogate", "pfp", jobs)
    R, S = stages(rslc, rpfp), stages(sslc, spfp)
    print(f"#jobs {j0}-{j1-1} n_nu {len(all_ids)} n_signal {len(sig_ids)} n_signal_topo {len(topo_ids)}")
    for k in R:
        r, s_ = R[k] & all_ids, S[k] & all_ids
        print(f"{k},{len(r)},{len(r & sig_ids)},{len(s_)},{len(s_ & sig_ids)},{len(r & s_)},{len(r & topo_ids)},{len(s_ & topo_ids)},{len(r & s_ & topo_ids)}", flush=True)
