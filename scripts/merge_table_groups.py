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
    import pyarrow as pa
    import pyarrow.parquet as pq
    for name in ("features", "labels", "targets"):
        out = f"{a.out}/{name}.parquet"
        if os.path.exists(out) and not a.force:
            print(name, "exists, skipped"); continue
        # stream group by group into one Parquet file (one row group per input group): memory = one group
        writer, n = None, 0
        for d in dirs:
            df = pd.read_parquet(f"{d}/{name}.parquet")
            if name == "targets":
                for c in df.columns:
                    if df[c].dtype == np.float64: df[c] = df[c].astype(np.float32)
            t = pa.Table.from_pandas(df, preserve_index=True); del df
            if writer is None: writer = pq.ParquetWriter(out, t.schema)
            writer.write_table(t); n += t.num_rows; del t
        writer.close(); print(name, n, "rows from", len(dirs), "groups", flush=True)

if __name__ == "__main__":
    main()
