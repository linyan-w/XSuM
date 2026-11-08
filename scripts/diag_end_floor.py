"""Intrinsic end-point floor: how well can the Geant4 end of a contained muon / proton be predicted from generator
truth alone (vertex, momentum vector)?  Best straight-line predictor: end = start + Lmed(p) * p_hat, Lmed = median
chord |end - start| in fine momentum bins. Decomposed into a length part (true chord direction, predicted length)
and a direction part (generator direction, true chord length)."""
import numpy as np
d = "data/prod_v10_14_02_03_official/primsets/mmap"
n = 2_000_000
cls = np.load(f"{d}/cls.npy", mmap_mode="r")[:n]; X = np.load(f"{d}/X.npy", mmap_mode="r")[:n, :, :3]; G = np.load(f"{d}/G.npy", mmap_mode="r")[:n]
cls, X, G = np.asarray(cls), np.asarray(X), np.asarray(G)
for name, c, pmax in (("muon", 0, 1.0), ("proton", 2, 1.0)):
    m = (cls == c) & (G[..., 7] == 1) & (G[..., 3] != -32768)
    p3 = X[m].astype(np.float64); g = G[m].astype(np.float64) / 10.0      # cm
    p = np.linalg.norm(p3, axis=1); keep = (p > 0.05) & (p < pmax); p3, g, p = p3[keep], g[keep], p[keep]
    s, e = g[:, :3], g[:, 3:6]; ch = e - s; L = np.linalg.norm(ch, axis=1); u = p3 / p[:, None]
    bins = np.quantile(p, np.linspace(0, 1, 201)); b = np.clip(np.searchsorted(bins, p) - 1, 0, 199)
    Lmed = np.array([np.median(L[b == k]) for k in range(200)])[b]
    full = np.linalg.norm(s + Lmed[:, None] * u - e, axis=1)
    lenpart = np.abs(Lmed - L)
    dirpart = np.linalg.norm(s + L[:, None] * u - e, axis=1)
    q = lambda v: f"median {np.median(v):5.1f}  p90 {np.percentile(v, 90):5.1f}  >10cm {np.mean(v > 10):.2f}"
    print(f"{name}: {len(p)} contained, p < {pmax} GeV, median chord {np.median(L):.0f} cm")
    print(f"   best straight-line predictor |pred - G4 end|: {q(full)}")
    print(f"     length part (straggling, interactions):   {q(lenpart)}")
    print(f"     direction part (multiple scattering):     {q(dirpart)}")
    # two independent draws from the conditional distribution (what a perfect surrogate vs reality gives), approx:
    rng = np.random.default_rng(0); res = (s + Lmed[:, None] * u - e)
    # pair each event with the residual of another event of similar momentum (sorted by p, neighbour)
    o = np.argsort(p); r2 = np.empty_like(res); r2[o] = np.roll(res[o], 1, axis=0)
    print(f"   perfect surrogate vs reality (two draws):    {q(np.linalg.norm(res - r2, axis=1))}")
