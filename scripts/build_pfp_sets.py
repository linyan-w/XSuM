#!/usr/bin/env python
"""Build PFP-set arrays for the matched slices of production files (one .npz per input file, then merge per group).

usage: python scripts/build_pfp_sets.py <prod dir or glob> <out dir> [--workers N] [--groups 20]
Honours XSUM_SLC_DIR for repaired slc sidecars. Output: <out>/group_NN.npz with key, X, n_fit, n_empty, truth_pdg.
"""
import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from xsum.io.xsum_df import chunk_id, read_table
from xsum.prep.pfpset import build_sets


def one(f):
    mcnu = read_table(f, "mcnu"); slc = read_table(f, "slc"); pfp = read_table(f, "pfp")
    if len(mcnu) == 0 or len(slc) == 0: return None
    return build_sets(mcnu, slc, pfp)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--groups", type=int, default=20)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(a.src if ("*" in a.src or a.src.endswith(".df")) else f"{a.src}/*.df"), key=chunk_id)
    print(f"{len(files)} input files", flush=True)
    t = time.time()
    for g in range(a.groups):
        out = f"{a.out}/group_{g:02d}.npz"
        if os.path.exists(out): continue
        grp = [f for f in files if chunk_id(f) % a.groups == g]
        parts = []
        with ProcessPoolExecutor(a.workers) as ex:
            for r in ex.map(one, grp):
                if r is not None: parts.append(r)
        if not parts: continue
        np.savez(out, **{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
        n = sum(len(p["key"]) for p in parts)
        print(f"group {g}: {len(grp)} files, {n} slices, n_fit>12: {np.mean(np.concatenate([p['n_fit'] for p in parts]) > 12):.4f}  ({time.time()-t:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
