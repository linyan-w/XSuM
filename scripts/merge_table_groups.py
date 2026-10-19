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
    ap = argparse.ArgumentParser(); ap.add_argument("groups"); ap.add_argument("out"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dirs = sorted(d for d in glob.glob(f"{a.groups}/group_*") if os.path.exists(f"{d}/targets.parquet"))
    for name in ("features", "labels", "targets"):
        df = pd.concat([pd.read_parquet(f"{d}/{name}.parquet") for d in dirs])
        df = df[~df.index.duplicated(keep="first")].sort_index()
        df.to_parquet(f"{a.out}/{name}.parquet"); print(name, df.shape, f"from {len(dirs)} groups", flush=True); del df


if __name__ == "__main__":
    main()
