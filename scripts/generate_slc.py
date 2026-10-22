#!/usr/bin/env python
"""Generate surrogated slice records for a truth table.

usage: python scripts/generate_slc.py reports/m2_scratch1633_v3 data/prod/table out.parquet [--split test] [--seed 0]
Reads <table>/features.parquet (and labels/targets for the optional split selection), writes one Parquet file
with the production slice columns plus the recomputed cohpi flags, one row per generated slice.
"""
import argparse
import time

import numpy as np
import pandas as pd

from xsum.data.splits import split_by_level
from xsum.gen.slc import Surrogate, cohpi_flags, generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("table_dir"); ap.add_argument("out")
    ap.add_argument("--split", default="all", choices=["all", "train", "val", "test"]); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=None); ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--batch", type=int, default=400000)
    a = ap.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = time.time()
    sur = Surrogate(a.model_dir, device="cpu" if a.cpu else "cuda")
    pf = pq.ParquetFile(f"{a.table_dir}/features.parquet")
    # file-level split decided from the index alone (cheap), the features stream row group by row group
    idx = pd.read_parquet(f"{a.table_dir}/features.parquet", columns=[]).index
    fid = idx.get_level_values("chunk").to_numpy() * 100000 + idx.get_level_values("__ntuple").to_numpy()
    keep = np.ones(len(idx), dtype=bool) if a.split == "all" else split_by_level(pd.MultiIndex.from_arrays([fid], names=["file"]), level="file", seed=0)[a.split]
    if a.max_rows and keep.sum() > a.max_rows:
        sel = np.random.default_rng(a.seed).choice(np.flatnonzero(keep), a.max_rows, replace=False); keep[:] = False; keep[sel] = True
    del idx, fid
    writer, n_in, n_out, off = None, 0, 0, 0
    stages = None
    for rg in range(pf.num_row_groups):
        F = pf.read_row_group(rg).to_pandas(); m = keep[off:off + len(F)]; off += len(F)
        F = F[m]
        if F.empty: continue
        for i in range(0, len(F), a.batch):
            g = generate(sur, F.iloc[i:i + a.batch], seed=a.seed + off + i)
            if g.empty: continue
            g = g.join(cohpi_flags(g))
            st = {c: int(g[c].sum()) for c in g.columns if c.startswith("pass_") or c == "in_sr"}
            stages = st if stages is None else {k: stages[k] + st[k] for k in st}
            tbl = pa.Table.from_pandas(g, preserve_index=True)
            if writer is None: writer = pq.ParquetWriter(a.out, tbl.schema)
            writer.write_table(tbl); n_in += len(F.iloc[i:i + a.batch]); n_out += len(g); del g, tbl
        del F
        print(f"row group {rg + 1}/{pf.num_row_groups}: {n_in} neutrinos -> {n_out} slices ({time.time()-t:.0f} s)", flush=True)
    if writer is not None: writer.close()
    print(f"{n_in} neutrinos -> {n_out} generated slices in {time.time()-t:.0f} s -> {a.out}")
    print("cohpi stages:", stages)

if __name__ == "__main__":
    main()
