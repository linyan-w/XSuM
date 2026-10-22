#!/usr/bin/env python
"""Write a surrogated dataset in the cafpyana production format.

usage: python scripts/write_surrogate_df.py <generated slc parquet> <production chunk dir> <out dir> [--chunks 0 1 ...]
For every production chunk file <dir>/xsum_*_chunk<N>.df it writes <out dir>/xsum_surrogate_chunk<N>.df with the
truth tables (mcnu, mcprim, hdr, histpotdf, histgenevtdf, pot) copied from the production and the `slc` table
replaced by the generated slices (index __ntuple, entry, rec.slc..index; tmatch_idx = generating neutrino,
tmatch_eff/purity NaN, selection flags recomputed). It also writes <out dir>/file_splits.parquet with the
train/val/test assignment of every input file, so comparisons with the real chain can be restricted to test files.
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

from xsum.data.splits import split_by_level
from xsum.io.xsum_df import chunk_id, read_table

SLC_ORDER = ["is_fv", "bcfm_score", "nu_score", "is_not_clear_cosmic", "n_pfp", "n_prong", "n_trk", "n_proton", "n_shower",
             "tmatch_idx", "tmatch_eff", "tmatch_purity", "long_dirx", "long_diry", "long_dirz", "long_trk_len", "long_trk_pid",
             "long_trk_chi2_mu", "long_trk_chi2_pro", "long_trk_chi2_mu_new", "long_trk_chi2_pro_new", "short_dirx", "short_diry",
             "short_dirz", "short_trk_len", "short_trk_pid", "short_trk_chi2_mu", "short_trk_chi2_pro", "short_trk_chi2_mu_new",
             "short_trk_chi2_pro_new", "vtx_x", "vtx_y", "vtx_z", "reco_t", "opening_angle", "is_muon_contained", "is_cpi_contained",
             "muon_start_x", "muon_start_y", "muon_start_z", "muon_end_x", "muon_end_y", "muon_end_z",
             "cpi_start_x", "cpi_start_y", "cpi_start_z", "cpi_end_x", "cpi_end_y", "cpi_end_z",
             "range_p_mu", "range_p_cpi", "total_pe", "sum_integ_1cm", "sum_integ_2cm", "sum_integ_3cm", "sum_integ_4cm", "sphere_3_over_4m3",
             "pass_fv_cosmic", "pass_pe_bcfm", "pass_2trk", "pass_dir", "pass_chi2", "pass_sphere", "pass_cathode", "pass_cohpi", "in_sr"]


def in_fv_nohiyz(x, y, z):
    pass_xz = (np.abs(x) > 10.) & (np.abs(x) < 190.) & (z > 10.) & (z < 450.)
    pass_y = ((z < 250) & (np.abs(y) < 190.)) | ((z > 250) & (y > -190.) & (y < 100.))
    return pass_xz & pass_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("generated"); ap.add_argument("prod_dir"); ap.add_argument("out_dir"); ap.add_argument("--chunks", nargs="*", type=int, default=None)
    ap.add_argument("--features", default=None, help="features.parquet of the full table, for the file-split table (default: derived from the generated index)")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    G = pd.read_parquet(a.generated)
    files = sorted(glob.glob(f"{a.prod_dir}/xsum_*.df"), key=chunk_id)
    files = [f for f in files if a.chunks is None or chunk_id(f) in a.chunks]
    # file-level split table (same rule and seed as training)
    if a.features:
        idx = pd.read_parquet(a.features, columns=[]).index
    else:
        idx = G.index
    fid = idx.get_level_values("chunk").to_numpy() * 100000 + idx.get_level_values("__ntuple").to_numpy()
    sp = split_by_level(pd.MultiIndex.from_arrays([fid], names=["file"]), level="file", seed=0)
    lab = np.where(sp["train"], "train", np.where(sp["val"], "val", "test"))
    splits = pd.DataFrame({"chunk": idx.get_level_values("chunk"), "__ntuple": idx.get_level_values("__ntuple"), "split": lab}).drop_duplicates()
    splits.to_parquet(f"{a.out_dir}/file_splits.parquet", index=False)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.workers) as ex:
        for msg in ex.map(write_one, [(f, G[G.index.get_level_values("chunk") == chunk_id(f)], a.out_dir, a.generated) for f in files]):
            print(msg, flush=True)


def write_one(args):
    f, g, out_dir, generated = args
    k = chunk_id(f)
    if True:
        g = g.copy()
        if g.empty:
            return f"chunk {k}: no generated rows, skipped"
        g = g.reset_index()
        g["rec.slc..index"] = g.groupby(["__ntuple", "entry"]).cumcount()
        g["tmatch_idx"] = g["rec.mc.nu..index"].astype(np.int32)
        g["tmatch_eff"] = np.nan; g["tmatch_purity"] = np.nan
        g["is_fv"] = in_fv_nohiyz(g.vtx_x.to_numpy(), g.vtx_y.to_numpy(), g.vtx_z.to_numpy())
        g["is_not_clear_cosmic"] = True
        g["sphere_3_over_4m3"] = g.sum_integ_3cm / (g.sum_integ_4cm - g.sum_integ_3cm)
        g["pass_fv_cosmic"] = g.is_fv
        g["pass_cohpi"] = g.pass_cohpi & g.pass_fv_cosmic; g["in_sr"] = g.in_sr & g.pass_fv_cosmic
        g["surrogate_topology_class"] = g["topology_class"]
        slc = g.set_index(["__ntuple", "entry", "rec.slc..index"])[[c for c in SLC_ORDER if c in g.columns] + ["surrogate_topology_class"]].sort_index()
        out = f"{out_dir}/xsum_surrogate_{k}.df"
        if os.path.exists(out):
            return f"chunk {k}: exists"
        tmp = out + ".tmp"
        with pd.HDFStore(tmp, mode="w") as st:
            for t in ("mcnu", "mcprim", "mcdaught", "pfp", "hdr", "histpotdf", "histgenevtdf", "pot"):
                try:
                    df = read_table(f, t)
                except Exception:
                    continue
                if len(df): df = df.droplevel("chunk")
                st.put(f"{t}_0", df, format="fixed")
            st.put("slc_0", slc, format="fixed")
            st.put("split", pd.DataFrame({"n_split": [1]}), format="fixed")
            st.put("provenance", pd.DataFrame({"generated_from": [os.path.basename(generated)], "production_chunk": [os.path.basename(f)]}), format="fixed")
        os.replace(tmp, out)
        return f"chunk {k}: {len(slc)} surrogated slices -> {out}"


if __name__ == "__main__":
    main()
