import glob, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
def dw(x, y, z):
    return np.minimum.reduce([201.3 - np.abs(x), 203.7 - np.abs(y), z, 500.8 - z])
bins = [-1e9, -3, 3, 10, 20, 1e9]
for tag in sys.argv[1:]:
    D = pd.concat([pd.read_parquet(f) for f in glob.glob(f"reports/cc1p_containment/{tag}/g*.parquet")])
    for c in ("r_two", "s_two", "r_contained", "s_contained"): D[c] = D[c].fillna(False).astype(bool)
    print(f"== {tag}")
    for cat in ["numu CC Npi", "numu CC other 0pi", "NC", "numu CC 1p0pi (any cont.)"]:
        m = (D.cat == cat) & ~D.signal
        for p in "rs":
            if tag != sys.argv[1] and p == "r": continue
            E = D[m & D[f"{p}_two"]]
            out = []
            for t in (1, 2):
                d = dw(E[f"{p}_t{t}_trk_end_x"], E[f"{p}_t{t}_trk_end_y"], E[f"{p}_t{t}_trk_end_z"])
                out.append(np.round(np.histogram(d, bins)[0] / len(E), 3))
            print(f"  {cat:26s} {'real' if p=='r' else 'XSuM'} n={len(E):5d} t1 {out[0]} t2 {out[1]}")
