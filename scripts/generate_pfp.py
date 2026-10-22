#!/usr/bin/env python
"""Generate PFP tables for a generated slice file (output of generate_slc.py) with a trained M4 model.

usage: python scripts/generate_pfp.py reports/m4_official_v2 <table dir> <generated slc parquet> <out parquet> [--max-rows N]
The table dir provides the features of the generating neutrinos (features.parquet); rows are matched by neutrino index.
"""
import argparse
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from xsum.gen.pfp import PfpSurrogate


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model_dir"); ap.add_argument("table_dir"); ap.add_argument("slc"); ap.add_argument("out")
    ap.add_argument("--max-rows", type=int, default=None); ap.add_argument("--batch", type=int, default=200000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(); t = time.time()
    sur = PfpSurrogate(a.model_dir, device="cpu" if a.cpu else "cuda")
    G = pd.read_parquet(a.slc)
    if a.max_rows and len(G) > a.max_rows: G = G.iloc[np.sort(np.random.default_rng(a.seed).choice(len(G), a.max_rows, replace=False))]
    nu_levels = [n for n in G.index.names if n != "draw"]
    F = pd.read_parquet(f"{a.table_dir}/features.parquet")
    writer, n = None, 0
    for i in range(0, len(G), a.batch):
        g = G.iloc[i:i + a.batch]
        f = F.loc[pd.MultiIndex.from_arrays([g.index.get_level_values(l) for l in nu_levels])]
        pfp, ne = sur.generate(f, g, seed=a.seed + i)
        tbl = pa.Table.from_pandas(pfp, preserve_index=True)
        if writer is None: writer = pq.ParquetWriter(a.out, tbl.schema)
        writer.write_table(tbl); n += len(pfp)
        print(f"{min(i + a.batch, len(G))}/{len(G)} slices -> {n} PFPs ({time.time()-t:.0f} s)", flush=True)
    if writer is not None: writer.close()
    print(f"done: {len(G)} slices -> {n} PFPs in {time.time()-t:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
