#!/usr/bin/env python3
"""Containment diagnostic for the CC1p0pi acceptance test: per neutrino, the two longest good tracks of the real
(best-matched) slice and of the surrogate slice, each containment condition separately, and a truth category.
Writes one parquet per job range; aggregate with scripts/diag_cc1p_summary.py.

usage: python3 diag_cc1p_containment.py <surrogate dataset name> <first job> <last job + 1> <out.parquet>
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "/pnfs/sbnd/persistent/users/lwan/dataframe")
import xsum_testsplit as ts
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from check_cc1p0pi_stages import fv, per_tpc, truth_signal, best_matched, CATHODE  # noqa: E402

COLS = ["trk_len", "trk_start_x", "trk_start_y", "trk_start_z", "trk_end_x", "trk_end_y", "trk_end_z", "trk_dir_x", "trk_dir_y", "trk_dir_z",
        "trackScore", "dist_to_vertex", "chi2_mu", "chi2_pro"]


def tracks(slc, pfp):
    s = slc.copy(); lv = list(s.index.names)
    p = pfp[(pfp.trk_len > 0) & (pfp.dist_to_vertex < 100)].copy()
    s["n_good"] = p.groupby(level=lv).size().reindex(s.index).fillna(0).astype(int)
    p = p.sort_values("trk_len", ascending=False); p["rank"] = p.groupby(level=lv).cumcount()
    for r in (0, 1):
        t = p[p["rank"] == r][[c for c in COLS if c in p.columns]]; t.index = t.index.droplevel(-1); t.columns = [f"t{r+1}_{c}" for c in t.columns]
        s = s.join(t, how="left")
    out = pd.DataFrame(index=pd.MultiIndex.from_arrays([s.index.get_level_values("job"), s.index.get_level_values("__ntuple"), s.index.get_level_values("entry"),
                                                        s.tmatch_idx.fillna(-1).astype(int).to_numpy()], names=["job", "__ntuple", "entry", "nu"]))
    pre = fv(s.vtx_x, s.vtx_y, s.vtx_z) & s.is_not_clear_cosmic.fillna(False).astype(bool).to_numpy() & (s.nu_score > 0.45).to_numpy()
    out["pre"] = pre; out["two"] = pre & (s.n_good == 2).to_numpy()
    for c in ("vtx_x", "vtx_y", "vtx_z", "n_good"): out[c] = s[c].to_numpy()
    for c in ("n_pfp", "n_prong", "n_trk", "n_proton", "n_shower"):   # slice-level topology counts (M2's topology class in the surrogate)
        if c in s.columns: out[c] = s[c].to_numpy()
    for c in s.columns:
        if c.startswith("t1_") or c.startswith("t2_"): out[c] = s[c].to_numpy()
    out["c_vtx"] = fv(s.vtx_x, s.vtx_y, s.vtx_z)
    for i in (1, 2):
        for e in ("start", "end"):
            out[f"c_t{i}_{e}"] = fv(s[f"t{i}_trk_{e}_x"], s[f"t{i}_trk_{e}_y"], s[f"t{i}_trk_{e}_z"])
    pts = [(s.vtx_x, s.vtx_y, s.vtx_z)] + [(s[f"t{i}_trk_{e}_x"], s[f"t{i}_trk_{e}_y"], s[f"t{i}_trk_{e}_z"]) for i in (1, 2) for e in ("start", "end")]
    xs = np.stack([np.asarray(p[0], float) for p in pts], 1)
    out["c_side"] = (xs < -CATHODE).all(1) | (xs > CATHODE).all(1)
    out["contained"] = out["two"] & per_tpc(pts)
    return out[~out.index.duplicated()]


def category(mcnu, mcprim):
    prim = mcprim.copy(); prim["p"] = np.sqrt(prim.genp_x**2 + prim.genp_y**2 + prim.genp_z**2)
    lv = list(mcnu.index.names); g = lambda m: m.groupby(level=lv).sum().reindex(mcnu.index).fillna(0).to_numpy()
    npi = g((prim.pdg.abs() == 211) & (prim.p > 0.07)); npr = g((prim.pdg == 2212) & (prim.p > 0.30)); nmu = g((prim.pdg.abs() == 13) & (prim.p > 0.22))
    cc = (mcnu.iscc == 1).to_numpy() & (np.abs(mcnu.pdg.to_numpy()) == 14)
    cat = np.where(~((np.abs(mcnu.pdg.to_numpy()) == 14) | (np.abs(mcnu.pdg.to_numpy()) == 12)), "other", "")
    cat = np.where((mcnu.iscc == 0).to_numpy(), "NC", cat)
    cat = np.where((np.abs(mcnu.pdg.to_numpy()) == 12) & (mcnu.iscc == 1).to_numpy(), "nue CC", cat)
    cat = np.where(cc & (npi > 0), "numu CC Npi", cat)
    cat = np.where(cc & (npi == 0) & (npr == 1) & (nmu == 1) & (mcnu.npi0.to_numpy() == 0), "numu CC 1p0pi (any cont.)", cat)
    cat = np.where(cc & (npi == 0) & (cat == ""), "numu CC other 0pi", cat)
    return np.where(cat == "", "other", cat)


if __name__ == "__main__":
    surr, j0, j1, outp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    ts.SURR = f"/pnfs/sbnd/persistent/users/lwan/dataframe/{surr}/xsum_surrogate_{{n}}.df"
    jobs = range(j0, j1)
    mcnu = ts.load_test("real", "mcnu", jobs); mcprim = ts.load_test("real", "mcprim", jobs)
    sig = truth_signal(mcnu, mcprim); cat = category(mcnu, mcprim)
    truth = pd.DataFrame({"signal": sig, "cat": cat, "mu_end_x": mcnu.mu_end_x.to_numpy(), "p_end_x": mcnu.p_end_x.to_numpy(),
                          "true_vtx_x": mcnu.position_x.to_numpy()}, index=mcnu.index)
    truth.index.names = ["job", "__ntuple", "entry", "nu"]
    rslc = best_matched(ts.load_test("real", "slc", jobs, matched_only=False), mcnu)
    rpfp = ts.load_test("real", "pfp", jobs, matched_only=False)
    rpfp = rpfp[rpfp.index.droplevel(-1).isin(rslc.index) & (rpfp.trk_len.notna() | ((rpfp.shw_bestplane_energy > -4) & (rpfp.shw_len > 0)))]
    R = tracks(rslc, rpfp); S = tracks(ts.load_test("surrogate", "slc", jobs), ts.load_test("surrogate", "pfp", jobs))
    R = R[R.pre | R.two]; S = S[S.pre | S.two]
    D = truth.join(R.add_prefix("r_"), how="left").join(S.add_prefix("s_"), how="left")
    D = D[D.r_pre.fillna(False).astype(bool) | D.s_pre.fillna(False).astype(bool)]
    D.reset_index().to_parquet(outp, index=False)
    print(f"jobs {j0}-{j1-1}: {len(D)} neutrinos at nu_score in either sample -> {outp}")
