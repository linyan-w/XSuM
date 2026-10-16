#!/usr/bin/env python
"""Build the per-neutrino training table (features + labels + targets) from XSuM production chunk files.

usage: python scripts/build_training_table.py 'data/prod/xsum_*_chunk*.df' data/prod/table
Writes <out>/features.parquet, <out>/labels.parquet (truth labels kept for holdouts, never model inputs),
<out>/targets.parquet, one row per neutrino, same index, processed chunk by chunk to bound memory.
"""
import argparse
import glob
import os
import time

import pandas as pd

from xsum.io.xsum_df import chunk_id, read_table
from xsum.prep.features import event_features
from xsum.prep.table import production_table

LABEL_COLS = ["E", "pdg", "iscc", "genie_mode", "w", "Q2", "nuint_categ", "is_fv", "is_signal_cohpi",
              "true_E_nu", "true_p_mu", "true_p_pi", "true_cos_theta_mu", "true_cos_theta_pi", "true_t",
              "nmu_27MeV", "npi_30MeV", "np_20MeV", "npi0", "nn_0MeV", "max_proton_ke"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(args.pattern), key=chunk_id)
    F, L, T = [], [], []
    for f in files:
        t = time.time()
        mcnu, mcprim, slc = read_table(f, "mcnu"), read_table(f, "mcprim"), read_table(f, "slc")
        feats, names = event_features(mcnu, mcprim)
        tab = production_table(mcnu, slc)
        lab = mcnu[[c for c in LABEL_COLS if c in mcnu.columns]].copy()
        F.append(feats); L.append(lab); T.append(tab)
        print(f"{os.path.basename(f)}: {len(mcnu)} neutrinos, {len(mcprim)} primaries, {len(slc)} slices, "
              f"matched {int(tab.matched.sum())}, selected {int(tab.selected.sum())}  ({time.time()-t:.0f} s)", flush=True)
        del mcnu, mcprim, slc
    for name, parts in (("features", F), ("labels", L), ("targets", T)):
        df = pd.concat(parts)
        df.to_parquet(os.path.join(args.out, f"{name}.parquet"))
        print(name, df.shape)


if __name__ == "__main__":
    main()
