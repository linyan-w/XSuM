#!/usr/bin/env python3
"""cohpi cut flow on true cohpi-signal neutrinos, real (matched slices) vs generated slices of one or more models,
on the test files (same file split as training). Usage: cohpi_signal_cutflow.py <out.json> name=gen.parquet ..."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from xsum.train.m2 import cohpi_cutflow
from xsum.data.splits import split_by_level

T = "data/prod_v10_14_02_03_official/table_matched"
K = ["chunk", "__ntuple", "entry", "rec.mc.nu..index"]
COLS = ["total_pe", "bcfm_score", "n_prong", "n_trk", "n_proton", "n_shower", "long_dirz", "short_dirz", "opening_angle",
        "long_trk_chi2_pro_new", "short_trk_chi2_pro_new", "sum_integ_1cm", "sum_integ_2cm", "sum_integ_3cm", "sum_integ_4cm",
        "muon_start_x", "muon_end_x", "cpi_start_x", "cpi_end_x", "reco_t"]


def keyof(df):
    return ((df.chunk.to_numpy(np.int64) * 100000 + df.__ntuple.to_numpy(np.int64)) * 10000000 + df.entry.to_numpy(np.int64)) * 16 + df["rec.mc.nu..index"].to_numpy(np.int64)


def test_mask(df, test_files):
    if os.environ.get("ALL"): return np.ones(len(df), bool)
    return np.isin(df.chunk.to_numpy(np.int64) * 100000 + df.__ntuple.to_numpy(np.int64), test_files)


L = pd.read_parquet("data/prod_v10_14_02_03_official/table/labels.parquet", columns=["is_signal_cohpi"]).reset_index()
fid = L.chunk.to_numpy(np.int64) * 100000 + L.__ntuple.to_numpy(np.int64)
sp = split_by_level(pd.MultiIndex.from_arrays([fid], names=["file"]), level="file", seed=0)
test_files = np.unique(fid[np.asarray(sp["test"])])
sig = L[L.is_signal_cohpi.astype(bool)]
out = {"n_signal_all": len(sig), "n_signal_test": int(test_mask(sig, test_files).sum())}
sigkeys = np.sort(keyof(sig)); del L
R = pd.read_parquet(f"{T}/targets.parquet", columns=COLS + K)
R = R.reset_index() if "chunk" not in R.columns else R
R = R[np.isin(keyof(R), sigkeys) & test_mask(R, test_files)]
out["real"] = cohpi_cutflow(R)
for a in sys.argv[2:]:
    name, f = a.split("=")
    G = pd.read_parquet(f, columns=COLS + K)
    G = G.reset_index() if "chunk" not in G.columns else G
    G = G[np.isin(keyof(G), sigkeys) & test_mask(G, test_files)]
    out[name] = cohpi_cutflow(G)
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out))
