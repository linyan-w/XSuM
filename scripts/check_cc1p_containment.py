#!/usr/bin/env python3
"""CC1p0pi containment closure on the test files: real (Pandora) vs XSuM surrogate.

Selection (PFP-level, cohpi PFP definitions): 2 prongs within 6 cm of the vertex and longer than 4 cm, both with
trackScore > 0.5, exactly one proton candidate (chi2_mu_new_I2 > 25), no shower within 50 cm. Containment: both
track ends inside the cohpi FV box. Truth signal (assumed definition, adjust to the collaborator's):
true vertex in the cohpi FV, numu CC, exactly 1 mu (KE > 27 MeV), 1 p (KE > 20 MeV), 0 pi+- (KE > 30 MeV), 0 pi0.
Reports, per sample: inclusive containment fractions, and efficiency / purity of 2-prong and 2-prong-contained.

usage (on sbndbuild03, with xsum_env.sh):  python3 check_cc1p_containment.py <first job> <last job + 1>
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "/pnfs/sbnd/persistent/users/lwan/dataframe")
from xsum_testsplit import load_test


def in_fv(x, y, z):
    xz = (np.abs(x) < 190.) & (z > 10.) & (z < 450.)
    yy = ((z < 250) & (np.abs(y) < 190.)) | ((z > 250) & (y > -190.) & (y < 100.))
    return xz & yy


def in_fv_nu(x, y, z):
    return in_fv(x, y, z) & (np.abs(x) > 10.)


def select(pfp, end_from="real"):
    """-> DataFrame per selected slice (index job, __ntuple, entry, slc): tmatch, contained flags."""
    p = pfp[(pfp.dist_to_vertex < 50) & pfp.trk_len.notna() | (pfp.dist_to_vertex < 50) & (pfp.trackScore < 0.5)]
    near = (p.dist_to_vertex < 6) & (p.trk_len > 4)
    trk = near & (p.trackScore > 0.5)
    prot = trk & (p.chi2_mu_new_I2 > 25)
    lv = list(p.index.names[:-1])
    g = lambda s: s.groupby(level=lv).sum()
    ok = (g(near) == 2) & (g(trk) == 2) & (g(prot) == 1) & (g(p.trackScore < 0.5) == 0)
    ok = ok[ok].index
    t = p[trk & p.index.droplevel(-1).isin(ok)].copy()
    if end_from == "straight":
        for a in "xyz":
            t[f"trk_end_{a}"] = t[f"trk_start_{a}"] + t.trk_len * t[f"trk_dir_{a}"]
    t["cont"] = in_fv(t.trk_start_x, t.trk_start_y, t.trk_start_z) & in_fv(t.trk_end_x, t.trk_end_y, t.trk_end_z)
    t["is_p"] = t.chi2_mu_new_I2 > 25
    out = pd.DataFrame(index=ok)
    out["mu_cont"] = t[~t.is_p].groupby(level=lv).cont.first().reindex(ok).fillna(False).astype(bool)
    out["p_cont"] = t[t.is_p].groupby(level=lv).cont.first().reindex(ok).fillna(False).astype(bool)
    out["tmatch"] = t.groupby(level=lv).slc_tmatch_idx.first().reindex(ok)
    return out


def signal(mcnu):
    return (in_fv_nu(mcnu.position_x, mcnu.position_y, mcnu.position_z) & (mcnu.iscc == 1) & (np.abs(mcnu.pdg) == 14)
            & (mcnu.nmu_27MeV == 1) & (mcnu.np_20MeV == 1) & (mcnu.npi_30MeV == 0) & (mcnu.npi0 == 0))


def report(label, sel, sig_keys, n_sig):
    key = pd.MultiIndex.from_arrays([sel.index.get_level_values("job"), sel.index.get_level_values("__ntuple"),
                                     sel.index.get_level_values("entry"), sel.tmatch.fillna(-1).astype(int).to_numpy()])
    is_sig = key.isin(sig_keys)
    rows = []
    for name, m in (("2-prong", np.ones(len(sel), bool)), ("2-prong, mu contained", sel.mu_cont.to_numpy()),
                    ("2-prong, p contained", sel.p_cont.to_numpy()), ("2-prong, both contained", (sel.mu_cont & sel.p_cont).to_numpy())):
        s = m.sum(); ss = (m & is_sig).sum()
        rows.append(f"{label:30s} {name:26s} selected {s:6d}  signal {ss:5d}  eff {ss / max(n_sig, 1):.4f}  purity {ss / max(s, 1):.3f}")
    print("\n".join(rows), flush=True)


jobs = range(int(sys.argv[1]), int(sys.argv[2]))
mcnu = load_test("real", "mcnu", jobs)          # identical in real and surrogate files
sig = mcnu[signal(mcnu)]
sig_keys = pd.MultiIndex.from_arrays([sig.index.get_level_values(i) for i in range(4)])
print(f"jobs {jobs.start}-{jobs.stop - 1}: test neutrinos {len(mcnu)}, true CC1p0pi signal {len(sig)}", flush=True)
real = load_test("real", "pfp", jobs)
surr = load_test("surrogate", "pfp", jobs)
report("Pandora (real end points)", select(real, "real"), sig_keys, len(sig))
report("Pandora (straight-line ends)", select(real, "straight"), sig_keys, len(sig))
report("XSuM generated", select(surr, "real"), sig_keys, len(sig))
