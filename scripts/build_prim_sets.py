#!/usr/bin/env python
"""Build primary-particle token arrays for every true neutrino of a production (one npz per group of job files).

usage: python scripts/build_prim_sets.py <prod dir> <out dir> [--workers N] [--groups 20] [--geo-only]
--geo-only writes only key and G (Geant4 start/end/length/flags, int16) compressed, to be added to an existing
particle-set mmap with xsum.train.m3.add_geo.
Output <out>/group_NN.npz: key [n,4] (chunk=job, __ntuple, entry, nu index), cls [n,32], X [n,32,4], n_prim, n_dropped, ke_dropped.
"""
import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from xsum.io.xsum_df import chunk_id, read_table
from xsum.prep.primset import build_primsets


def one(f):
    mcnu = read_table(f, "mcnu"); mcprim = read_table(f, "mcprim")
    return build_primsets(mcnu, mcprim) if len(mcnu) else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--groups", type=int, default=20); ap.add_argument("--geo-only", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(a.src if ("*" in a.src or a.src.endswith(".df")) else f"{a.src}/*.df"), key=chunk_id)
    print(f"{len(files)} input files", flush=True); t = time.time()
    for g in range(a.groups):
        out = f"{a.out}/group_{g:02d}.npz"
        if os.path.exists(out): continue
        grp = [f for f in files if chunk_id(f) % a.groups == g]
        with ProcessPoolExecutor(a.workers) as ex:
            parts = [r for r in ex.map(one, grp) if r is not None]
        if not parts: continue
        if a.geo_only: np.savez_compressed(out + ".tmp.npz", key=np.concatenate([p["key"] for p in parts]), cls=np.concatenate([p["cls"] for p in parts]), G=np.concatenate([p["G"] for p in parts]))
        else: np.savez(out + ".tmp.npz", **{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
        os.replace(out + ".tmp.npz", out)
        nd = np.concatenate([p["n_dropped"] for p in parts])
        print(f"group {g}: {len(grp)} files, {sum(len(p['key']) for p in parts)} neutrinos, truncated {np.mean(nd > 0):.4f}  ({time.time()-t:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
