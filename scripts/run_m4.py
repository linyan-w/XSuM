#!/usr/bin/env python
"""Train the M4 PFP-set surrogate. Usage: scripts/run_m4.py <matched table dir> <pfpsets dir> reports/m4 [--epochs N] [--max-slices N]"""
import argparse

from xsum.train.m4 import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("table_dir"); ap.add_argument("pfpsets_dir"); ap.add_argument("out_dir"); ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max-slices", type=int, default=3_000_000); ap.add_argument("--d-model", type=int, default=192); ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--cpu", action="store_true"); ap.add_argument("--primsets", default=None, help="particle tokens for PFP-to-particle cross-attention")
    ap.add_argument("--part-layers", type=int, default=2); ap.add_argument("--geo", type=int, nargs="?", const=1, default=0, help="particle tokens carry Geant4 start/end/length/flags (1) + TPC exit point (2)")
    a = ap.parse_args()
    run(a.table_dir, a.pfpsets_dir, a.out_dir, device="cpu" if a.cpu else "cuda", epochs=a.epochs, max_slices=a.max_slices, d_model=a.d_model, n_layers=a.layers, primsets_dir=a.primsets, n_part_layers=a.part_layers, geo=a.geo)
