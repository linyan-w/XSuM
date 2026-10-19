#!/usr/bin/env python
"""Concatenate grouped training tables (group_NN/{features,labels,targets}.parquet) into one table directory.

usage: python scripts/merge_table_groups.py data/prod_v10_14_02_03_official/groups data/prod_v10_14_02_03_official/table
Rows keep their (chunk=job number, __ntuple, entry, rec.mc.nu..index) index, so file-level splits work unchanged.
"""
import argparse
import glob
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("groups"); ap.add_argument("out"); ap.add_argument("--force", action="store_true"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dirs = sorted(d for d in glob.glob(f"{a.groups}/group_*") if os.path.exists(f"{d}/targets.parquet"))
    import numpy as np
    for name in ("features", "labels", "targets"):
        if os.path.exists(f"{a.out}/{name}.parquet") and not a.force:
            print(name, "exists, skipped"); continue
        parts = []
        for d in dirs:
            df = pd.read_parquet(f"{d}/{name}.parquet")
            if name == "targets":                      # float64 -> float32 column by column (half the memory)
                for c in df.columns:
                    if df[c].dtype == np.float64: df[c] = df[c].astype(np.float32)
            parts.append(df)
        df = pd.concat(parts, copy=False); del parts   # rows are unique by construction (job number in the index); no sort
        df.to_parquet(f"{a.out}/{name}.parquet"); print(name, df.shape, f"from {len(dirs)} groups", flush=True); del df


if __name__ == "__main__":
    main()
