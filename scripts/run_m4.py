#!/usr/bin/env python
"""Train the M4 PFP-set surrogate. Usage: scripts/run_m4.py <matched table dir> <pfpsets dir> reports/m4 [--epochs N] [--max-slices N]

Default (the shipped configuration, M3w0, 2026-11-08): layout v4 (generated track end + per-axis wall flag and offset),
no particle cross-attention, generator truth only; 3 M slices, 40 epochs, d = 256.
--primsets / --part-layers turn on PFP-to-particle cross-attention (ablation: 1 cm on the end point, no per-class
gain, 2x parameters). --geo puts Geant4 truth on the particle tokens: diagnostic only, never for a shipped model."""
import argparse

from xsum.train.m4 import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("table_dir"); ap.add_argument("pfpsets_dir"); ap.add_argument("out_dir"); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max-slices", type=int, default=3_000_000); ap.add_argument("--d-model", type=int, default=256); ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--cpu", action="store_true"); ap.add_argument("--primsets", default=None, help="particle tokens for PFP-to-particle cross-attention (off by default)")
    ap.add_argument("--part-layers", type=int, default=2); ap.add_argument("--geo", type=int, nargs="?", const=1, default=0, help="DIAGNOSTIC ONLY (Geant4 inputs): particle tokens carry Geant4 start/end/length/flags (1) + TPC exit point (2)")
    a = ap.parse_args()
    run(a.table_dir, a.pfpsets_dir, a.out_dir, device="cpu" if a.cpu else "cuda", epochs=a.epochs, max_slices=a.max_slices, d_model=a.d_model, n_layers=a.layers, primsets_dir=a.primsets, n_part_layers=a.part_layers, geo=a.geo)
