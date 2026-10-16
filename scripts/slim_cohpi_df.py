#!/usr/bin/env python
"""Slim a cafpyana cohpi HDF5 dataframe to its physics columns and write one Parquet file per key.

usage: python scripts/slim_cohpi_df.py <file.df> <out_dir> [--keys mcnu evt hdr ...]
"""
import argparse
import json
import time

from xsum.io.cohpi_df import DEFAULT_KEYS, hdf_keys, load_key, slim_to_parquet, weight_groups
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("df")
    ap.add_argument("out_dir")
    ap.add_argument("--keys", nargs="*", default=None)
    args = ap.parse_args()
    keys = args.keys or [k for k in DEFAULT_KEYS if k in hdf_keys(args.df)]
    t = time.time()
    written = slim_to_parquet(args.df, args.out_dir, keys)
    for k, f in written.items():
        print(f"{k:14s} -> {f}")
    # record the weight groups once, for the documentation
    groups = weight_groups(pd.read_hdf(args.df, key="mcnu")) if "mcnu" in keys else {}
    with open(f"{args.out_dir}/weight_groups_mcnu.json", "w") as fh:
        json.dump(groups, fh, indent=1)
    print(f"{len(groups)} weight groups recorded; {time.time() - t:.1f} s")


if __name__ == "__main__":
    main()
