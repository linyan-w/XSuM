#!/usr/bin/env python
"""Split a training table into (a) a matched-only table for the slice models and (b) a row subsample with the
cardinality labels, streaming row group by row group so that memory stays at one row group.

usage: python scripts/split_official_table.py data/prod/table data/prod/table_matched data/prod/table_card [--card-rows 4000000]
"""
import argparse
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def stream_filter(src, dst, masks, columns=None):
    pf = pq.ParquetFile(src); writer = None; n = 0; off = 0
    if columns is not None:                       # keep the pandas index columns so that the file still reads as a DataFrame
        import json
        meta = json.loads(pf.schema_arrow.metadata[b"pandas"]); idx = [c for c in meta.get("index_columns", []) if isinstance(c, str)]
        columns = list(columns) + [c for c in idx if c not in columns]
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=columns); m = masks[off:off + t.num_rows]; off += t.num_rows
        t = t.filter(pa.array(m))
        if writer is None: writer = pq.ParquetWriter(dst, t.schema)
        writer.write_table(t); n += t.num_rows
    writer.close(); return n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("table"); ap.add_argument("out_matched"); ap.add_argument("out_card")
    ap.add_argument("--card-rows", type=int, default=4_000_000); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    os.makedirs(a.out_matched, exist_ok=True); os.makedirs(a.out_card, exist_ok=True)
    matched = pq.read_table(f"{a.table}/targets.parquet", columns=["matched"]).column("matched").to_numpy().astype(bool)
    n = len(matched); print("rows", n, "matched", int(matched.sum()), flush=True)
    for name in ("features", "labels", "targets"):
        k = stream_filter(f"{a.table}/{name}.parquet", f"{a.out_matched}/{name}.parquet", matched); print("matched", name, k, flush=True)
    rng = np.random.default_rng(a.seed); sub = np.zeros(n, dtype=bool); sub[rng.choice(n, min(a.card_rows, n), replace=False)] = True
    k = stream_filter(f"{a.table}/features.parquet", f"{a.out_card}/features.parquet", sub)
    k2 = stream_filter(f"{a.table}/targets.parquet", f"{a.out_card}/targets.parquet", sub, columns=["matched", "n_matched_slices"])
    print("card", k, k2, flush=True)


if __name__ == "__main__":
    main()
