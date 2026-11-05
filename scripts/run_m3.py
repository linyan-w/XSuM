#!/usr/bin/env python
"""M3: train the particle-set encoder (stage A) and/or export embeddings into new table dirs.

usage: python scripts/run_m3.py train <table_matched> <primsets> reports/m3 [--epochs N] [--max-rows N]
       python scripts/run_m3.py export reports/m3 <primsets> <table dir> <out table dir>
"""
import argparse

from xsum.train.m3 import export, run, to_mmap

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["train", "export", "mmap"]); ap.add_argument("args", nargs="+")
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--max-rows", type=int, default=3_000_000)
    a = ap.parse_args()
    if a.mode == "train": run(*a.args, epochs=a.epochs, max_rows=a.max_rows)
    elif a.mode == "mmap": print(to_mmap(a.args[0]), "neutrinos")
    else: export(*a.args)
