import sys, numpy as np, pandas as pd
sys.path.insert(0, "/pnfs/sbnd/persistent/users/lwan/dataframe")
from xsum_testsplit import load_test

def in_fv(x, y, z):
    xz = (np.abs(x) < 190.) & (z > 10.) & (z < 450.)
    yy = ((z < 250) & (np.abs(y) < 190.)) | ((z > 250) & (y > -190.) & (y < 100.))
    return xz & yy

def analyse(pfp, label, end_from):
    p = pfp[pfp.dist_to_vertex < 50]
    near = (p.dist_to_vertex < 6) & (p.trk_len > 4)
    trk = near & (p.trackScore > 0.5)
    prot = trk & (p.chi2_mu_new_I2 > 25)
    lv = list(p.index.names[:-1])
    g = lambda s: s.groupby(level=lv).sum()
    n_prong, n_trk, n_prot, n_shw = g(near), g(trk), g(prot), g(p.trackScore < 0.5)
    ok = (n_prong == 2) & (n_trk == 2) & (n_prot == 1) & (n_shw == 0)
    t = p[trk & p.index.droplevel(-1).isin(ok[ok].index)].copy()
    if end_from == "straight":
        for a in "xyz": t[f"trk_end_{a}"] = t[f"trk_start_{a}"] + t.trk_len * t[f"trk_dir_{a}"]
    t["cont"] = in_fv(t.trk_start_x, t.trk_start_y, t.trk_start_z) & in_fv(t.trk_end_x, t.trk_end_y, t.trk_end_z)
    t["is_p"] = t.chi2_mu_new_I2 > 25
    mu, pr = t[~t.is_p], t[t.is_p]
    cm = mu.groupby(level=lv).cont.first(); cp = pr.groupby(level=lv).cont.first()
    both = (cm.reindex(ok[ok].index).fillna(False).astype(bool) & cp.reindex(ok[ok].index).fillna(False).astype(bool))
    endout = ~in_fv(mu.trk_end_x, mu.trk_end_y, mu.trk_end_z)
    av = (np.abs(mu.trk_end_x) > 201) | (np.abs(mu.trk_end_y) > 201) | (mu.trk_end_z < -1) | (mu.trk_end_z > 510)
    print(f"{label:34s} 1mu1p 2-prong slices {int(ok.sum()):6d} | mu contained {cm.mean():.3f}  p contained {cp.mean():.3f}  both {both.mean():.3f} "
          f"| mu end outside active volume {av.mean():.3f}", flush=True)
    return ok, both

jobs = range(int(sys.argv[1]), int(sys.argv[2]))
real = load_test("real", "pfp", jobs); surr = load_test("surrogate", "pfp", jobs)
real = real[real.trk_len.notna()]
analyse(real, "Pandora (real end points)", "real")
analyse(real, "Pandora (straight-line end points)", "straight")
analyse(surr, "XSuM generated", "real")
