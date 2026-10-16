#!/usr/bin/env python
"""Summarise a slimmed cohpi cache: rows, matching, selection efficiency, response quantiles, splits.

usage: python scripts/dataset_summary.py data/cache [--out reports/dataset_summary.md]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from xsum.constants import NUINT_CATEG
from xsum.data.splits import split_by_level
from xsum.io.cohpi_df import load_cache
from xsum.prep.table import build_table, reco_columns, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dfs = load_cache(args.cache, ["mcnu", "evt", "hdr"])
    mcnu, evt, hdr = dfs["mcnu"], dfs["evt"], dfs["hdr"]
    lines = ["# Dataset summary", ""]
    lines += [f"- mcnu rows: {len(mcnu)} (signal neutrinos); evt rows: {len(evt)} (selected slices); hdr rows: {len(hdr)} (triggers)",
              f"- files (file_idx, __ntuple): {hdr.reset_index()[['file_idx', '__ntuple']].drop_duplicates().shape[0]}; total POT: {hdr.pot.sum():.3e}",
              f"- reco columns modelled: {len(reco_columns(evt))}"]
    cats = evt.nuint_categ.fillna(-99).astype(int).value_counts()
    lines += ["", "| nuint_categ of selected slices | rows |", "|---|---|"]
    for k, v in cats.items():
        lines.append(f"| {k} {NUINT_CATEG.get(k, 'cosmic / unmatched')} | {v} |")
    table = build_table(mcnu, evt)
    s = summary(table)
    lines += ["", "## Truth -> reco table", "", "```", json.dumps(s, indent=1), "```"]
    masks = split_by_level(table.index)
    lines += ["", "| split (by file_idx) | neutrinos | selected |", "|---|---|---|"]
    for k, m in masks.items():
        lines.append(f"| {k} | {int(m.sum())} | {int(table.selected[m].sum())} |")
    sel = table[table.selected == 1]
    lines += ["", "## Tier 1 columns on selected signal (mean, std, NaN fraction)", "", "| column | mean | std | NaN |", "|---|---|---|---|"]
    for c in reco_columns(evt):
        x = sel[c].astype(float)
        lines.append(f"| {c} | {np.nanmean(x):.4g} | {np.nanstd(x):.4g} | {x.isna().mean():.3f} |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
