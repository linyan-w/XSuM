#!/usr/bin/env python
"""Train the M2 slice-block flow surrogate. Usage: scripts/run_m2.py data/prod/table reports/m2 [--epochs N] [--max-rows N]"""
import argparse

from xsum.train.m2 import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("table_dir"); ap.add_argument("out_dir"); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max-rows", type=int, default=None); ap.add_argument("--hidden", type=int, default=512); ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--cpu", action="store_true"); ap.add_argument("--card-table", default=None)
    ap.add_argument("--walls", action="store_true", help="study M2w: wall flag / offset targets")
    ap.add_argument("--topo-last", action="store_true", help="study M2t: topology sampled after the flow")
    ap.add_argument("--range-curve", action="store_true", help="study M2t: range momenta flagged on the length curve")
    ap.add_argument("--vtx-resid", action="store_true", help="reco vertex as the residual to the true vertex, asinh(d / 1 cm)")
    ap.add_argument("--topo-cls", action="store_true", help="with --two-stage: no counts in stage 1; topology class from a classifier on truth + generated stage 1")
    ap.add_argument("--topo-onehot", action="store_true", help="with --two-stage: counts as a one-hot topology class in stage 1")
    ap.add_argument("--two-stage", action="store_true", help="stage 1: block + PFP counts + group presence; stage 2: optional groups")
    a = ap.parse_args()
    run(a.table_dir, a.out_dir, device="cpu" if a.cpu else "cuda", epochs=a.epochs, max_rows=a.max_rows, hidden=a.hidden, n_layers=a.layers, card_table=a.card_table,
        walls=a.walls, topo_last=a.topo_last, range_curve=a.range_curve, two_stage=a.two_stage, topo_onehot=a.topo_onehot, topo_cls=a.topo_cls, vtx_resid=a.vtx_resid)
