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
    a = ap.parse_args()
    t = time.time()
    F = pd.read_parquet(f"{a.table_dir}/features.parquet")
    if a.split != "all":
        fid = F.index.get_level_values("chunk").to_numpy() * 100000 + F.index.get_level_values("__ntuple").to_numpy()
        F = F[split_by_level(pd.MultiIndex.from_arrays([fid], names=["file"]), level="file", seed=0)[a.split]]
    if a.max_rows and len(F) > a.max_rows:
        F = F.iloc[np.sort(np.random.default_rng(a.seed).choice(len(F), a.max_rows, replace=False))]
    sur = Surrogate(a.model_dir, device="cpu" if a.cpu else "cuda")
    parts = []
    for i in range(0, len(F), 500000):
        parts.append(generate(sur, F.iloc[i:i + 500000], seed=a.seed + i))
    gen = pd.concat(parts)
    gen = gen.join(cohpi_flags(gen))
    gen.to_parquet(a.out)
    print(f"{len(F)} neutrinos -> {len(gen)} generated slices in {time.time()-t:.0f} s -> {a.out}")
    print("cohpi stages:", {c: int(gen[c].sum()) for c in gen.columns if c.startswith("pass_") or c == "in_sr"})


if __name__ == "__main__":
    main()
