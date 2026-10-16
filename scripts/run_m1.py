#!/usr/bin/env python
"""Run the M1 baselines. Usage: scripts/run_m1.py data/prod/table reports/m1 [--epochs N] [--cpu] [--max-rows N]"""
import argparse

from xsum.train.baselines import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("table_dir"); ap.add_argument("out_dir"); ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--cpu", action="store_true"); ap.add_argument("--max-rows", type=int, default=None)
    a = ap.parse_args()
    run(a.table_dir, a.out_dir, device="cpu" if a.cpu else "cuda", epochs=a.epochs, max_rows=a.max_rows)
