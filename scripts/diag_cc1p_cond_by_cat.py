import glob, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
conds = ["c_vtx", "c_t1_start", "c_t1_end", "c_t2_start", "c_t2_end", "c_side", "contained"]
for tag in sys.argv[1:]:
    D = pd.concat([pd.read_parquet(f) for f in glob.glob(f"reports/cc1p_containment/{tag}/g*.parquet")])
    for c in D.columns:
        if c[2:] in conds + ["two"]: D[c] = D[c].fillna(False).astype(bool)
    print(f"== {tag}   pass rate on own 2prong population: " + " ".join(f"{c:>10s}" for c in conds))
    for cat in ["numu CC Npi", "numu CC other 0pi", "NC", "numu CC 1p0pi (any cont.)"]:
        m = (D.cat == cat) & ~D.signal
        for p in "rs":
            E = D[m & D[f"{p}_two"]]
            print(f"  {cat:26s} {'real' if p=='r' else 'XSuM'} " + " ".join(f"{E[f'{p}_{c}'].mean():10.3f}" for c in conds))
    # end-in-FV given end not on wall: where are interior ends
