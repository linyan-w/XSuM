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

    def nukey(idx):   # (chunk, __ntuple, entry, rec.mc.nu..index) -> one int64
        g = lambda n: idx.get_level_values(n).to_numpy().astype(np.int64)
        return ((g("chunk") * 100000 + g("__ntuple")) * 10000000 + g("entry")) * 16 + g("rec.mc.nu..index")
    F = pd.read_parquet(f"{a.table_dir}/features.parquet"); fkey = nukey(F.index); order = np.argsort(fkey); fkey = fkey[order]
    Fx = np.ascontiguousarray(F.to_numpy(np.float32))[order]; fcols = list(F.columns); del F
    pf = pq.ParquetFile(a.slc); writer, n, done = None, 0, 0
    for rg in range(pf.num_row_groups):
        G = pf.read_row_group(rg).to_pandas()
        if a.max_rows and done >= a.max_rows: break
        if a.max_rows and done + len(G) > a.max_rows: G = G.iloc[:a.max_rows - done]
        for i in range(0, len(G), a.batch):
            g = G.iloc[i:i + a.batch]
            pos = np.searchsorted(fkey, nukey(g.index)); assert np.all(fkey[np.minimum(pos, len(fkey) - 1)] == nukey(g.index)), "neutrino not found in the features table"
            f = pd.DataFrame(Fx[pos], columns=fcols)
            pfp, ne = sur.generate(f, g, seed=a.seed + done + i)
            tbl = pa.Table.from_pandas(pfp, preserve_index=True)
            if writer is None: writer = pq.ParquetWriter(a.out, tbl.schema)
            writer.write_table(tbl); n += len(pfp); del pfp, tbl, f
        done += len(G); del G
        print(f"row group {rg + 1}/{pf.num_row_groups}: {done} slices -> {n} PFPs ({time.time()-t:.0f} s)", flush=True)
    if writer is not None: writer.close()
    print(f"done: {done} slices -> {n} PFPs in {time.time()-t:.0f} s -> {a.out}")

if __name__ == "__main__":
    main()
