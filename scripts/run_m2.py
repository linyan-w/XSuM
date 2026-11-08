#!/usr/bin/env python
"""Train the M2 slice-block flow surrogate. Usage: scripts/run_m2.py data/prod/table reports/m2 [--epochs N] [--max-rows N]"""
import argparse

from xsum.train.m2 import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("table_dir"); ap.add_argument("out_dir"); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max-rows", type=int, default=None); ap.add_argument("--hidden", type=int, default=512); ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--cpu", action="store_true"); ap.add_argument("--card-table", default=None)
    ap.add_argument("--topo-first", action="store_true", help="old order: topology sampled first and fed to the flow")
    ap.add_argument("--no-range-curve", action="store_true")
    a = ap.parse_args()
    run(a.table_dir, a.out_dir, device="cpu" if a.cpu else "cuda", epochs=a.epochs, max_rows=a.max_rows, hidden=a.hidden, n_layers=a.layers, card_table=a.card_table,
        topo_last=not a.topo_first, range_curve=not a.no_range_curve)
