#!/usr/bin/env python
"""Rebuild the two-track block and selection flags of the `slc` tables of a production from its `pfp` tables.

usage: python scripts/repair_official_slc.py <production dir> <sidecar out dir> [--workers N]
Writes <out>/<basename>.df with a single key slc_0 (same index as the original). Point XSUM_SLC_DIR at <out>
when building tables or loading the production.
"""
import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from xsum.io.xsum_df import chunk_id
from xsum.prep.repair import repair_slc


def one(args):
    f, out = args
    dst = os.path.join(out, os.path.basename(f))
    if os.path.exists(dst):
        return f, "exists"
    with pd.HDFStore(f, mode="r") as st:
        keys = [k for k in st.keys() if k.startswith("/slc_")]
        parts = []
        for k in keys:
            i = k.rsplit("_", 1)[1]
            slc, pfp = st[k], st[f"/pfp_{i}"]
            parts.append((k, repair_slc(slc, pfp) if len(slc) else slc))
    tmp = dst + ".tmp"
    with pd.HDFStore(tmp, mode="w") as st:
        for k, r in parts:
            st.put(k.lstrip("/"), r, format="fixed")
    os.replace(tmp, dst)
    return f, f"{sum(len(r) for _, r in parts)} slices, {sum(int(r.pass_cohpi.sum()) for _, r in parts if 'pass_cohpi' in r)} cohpi"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("prod_dir"); ap.add_argument("out"); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(f"{a.prod_dir}/*.df"), key=chunk_id)
    t = time.time(); n = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for f, msg in ex.map(one, [(f, a.out) for f in files]):
            n += 1
            if n % 100 == 0 or msg == "exists" and n < 3: print(f"{n}/{len(files)} {os.path.basename(f)}: {msg}  ({time.time()-t:.0f} s)", flush=True)
    print(f"done: {n} files in {time.time()-t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
