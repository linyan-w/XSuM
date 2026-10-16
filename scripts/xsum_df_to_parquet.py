#!/usr/bin/env python
"""Merge the chunked XSuM production .df files into one Parquet file per table.

usage: python scripts/xsum_df_to_parquet.py 'data/prod_v10_14_02_03/xsum_v10_14_02_03_chunk*.df' data/prod_v10_14_02_03/parquet
"""
import argparse
import time

from xsum.io.xsum_df import TABLES, to_parquet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("out_dir")
    ap.add_argument("--tables", nargs="*", default=list(TABLES))
    args = ap.parse_args()
    t = time.time()
    for f in to_parquet(args.pattern, args.out_dir, args.tables):
        print(f)
    print(f"{time.time() - t:.1f} s")


if __name__ == "__main__":
    main()
