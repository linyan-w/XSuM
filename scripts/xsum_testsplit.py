#!/usr/bin/env python3
"""Load XSuM real and surrogate dataframes restricted to the TEST files (input flat CAFs never used in training).

The train/val/test split is by input flat CAF: a row belongs to the test sample if its (job number, __ntuple)
pair is labelled "test" in file_splits.parquet (one copy at the top of the dataframe area, written by XSuM's
scripts/make_file_splits.py from the generators' training table: the test files are the files no generator saw;
the per-dataset copies are identical). Job number N is the number in the file name
(real: xsum_v10_14_02_03_N.df, surrogate: xsum_surrogate_N.df); __ntuple is the first index level of every table.

Usage as a library:
    from xsum_testsplit import load_test
    real = load_test("real", "slc", jobs=range(10))          # repaired slc tables, neutrino-matched slices only (default)
    surr = load_test("surrogate", "slc", jobs=range(10))     # generated slc tables
    real_pfp = load_test("real", "pfp", jobs=range(10)); surr_pfp = load_test("surrogate", "pfp", jobs=range(10))
Usage from the shell (prints how many test rows per table for a few jobs):
    python3 xsum_testsplit.py 0 1 2
Every returned DataFrame gets a leading index level "job". Keys: mcnu, mcprim, mcdaught, slc, pfp, hdr.

Like-for-like: the surrogate generates only slices truth-matched to a true neutrino (tmatch_idx points to an mcnu
row). The real slc/pfp tables also contain cosmic-only and unmatched slices; by default (matched_only=True) those
are dropped from the real tables so both sides describe the same population. Pass matched_only=False to keep them.
"""
import os
import sys

import pandas as pd

BASE = "/pnfs/sbnd/persistent/users/lwan/dataframe"
REAL = f"{BASE}/prod_v10_14_02_03_official/xsum_v10_14_02_03_{{n}}.df"
REAL_SLC = f"{BASE}/prod_v10_14_02_03_official_slcfix/xsum_v10_14_02_03_{{n}}.df"   # slc must come from here (repaired)
SURR = f"{BASE}/surrogate_official_m2vw_pfp/xsum_surrogate_{{n}}.df"   # shipped chain; scripts override via ts.SURR
SPLITS = f"{BASE}/file_splits.parquet"   # one copy at the top level: the split is identical for every surrogate dataset

_splits = None


def test_ntuples(job, split="test"):
    """Set of __ntuple values of job `job` that belong to `split` (default: test)."""
    global _splits
    if _splits is None:
        _splits = pd.read_parquet(SPLITS)
    s = _splits[(_splits["chunk"] == job) & (_splits["split"] == split)]
    return set(s["__ntuple"].tolist())


def _read(path, key):
    with pd.HDFStore(path, mode="r") as st:
        keys = sorted((k for k in st.keys() if k.startswith(f"/{key}_")), key=lambda k: int(k.rsplit("_", 1)[1]))
        return pd.concat([st[k] for k in keys]) if keys else pd.DataFrame()


def _matched(df, mcnu, slc_tmatch_col):
    key = pd.MultiIndex.from_arrays([df.index.get_level_values("__ntuple"), df.index.get_level_values("entry"), df[slc_tmatch_col].astype(int).to_numpy()])
    nu = pd.MultiIndex.from_arrays([mcnu.index.get_level_values("__ntuple"), mcnu.index.get_level_values("entry"), mcnu.index.get_level_values(-1)])
    return df[key.isin(nu)]


def load_test(which, key, jobs, split="test", matched_only=True):
    """which: 'real' or 'surrogate'; key: table name; jobs: iterable of job numbers (0..1999)."""
    out = {}
    for n in jobs:
        path = (REAL_SLC if (which == "real" and key == "slc") else REAL if which == "real" else SURR).format(n=n)
        if not os.path.exists(path):
            continue
        df = _read(path, key)
        if df.empty:
            continue
        keep = df.index.get_level_values("__ntuple").isin(test_ntuples(n, split))
        df = df[keep]
        if which == "real" and matched_only and key in ("slc", "pfp"):
            mcnu = _read(REAL.format(n=n), "mcnu")
            df = _matched(df, mcnu, "tmatch_idx" if key == "slc" else "slc_tmatch_idx")
        out[n] = df
    return pd.concat(out, names=["job"]) if out else pd.DataFrame()


if __name__ == "__main__":
    jobs = [int(x) for x in sys.argv[1:]] or [0]
    for key in ("mcnu", "slc", "pfp"):
        r, s = load_test("real", key, jobs), load_test("surrogate", key, jobs)
        print(f"{key:5s}  test rows: real {len(r):8d}   surrogate {len(s):8d}")
    r, s = load_test("real", "slc", jobs), load_test("surrogate", "slc", jobs)
    for c in ("pass_2trk", "pass_cohpi", "in_sr"):
        print(f"{c:10s}  real {int(r[c].sum()):6d}   surrogate {int(s[c].sum()):6d}")
