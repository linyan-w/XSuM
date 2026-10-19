"""Read the XSuM production dataframes written by cafpyana's run_df_maker (pool mode).

Each chunk file <prefix>_chunkN.df holds, per table, keys <table>_<k> for k in range(n_split) (one key per
~2 GB of accumulated input) and a 'split' key with n_split. Row indices start with (__ntuple, entry, ...)
where __ntuple numbers the input flat CAFs within that run; the chunk id is prepended here so that rows are
unique across the production.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Dict, Iterable, List, Optional

import pandas as pd

TABLES = ("mcnu", "mcprim", "slc", "pfp", "hdr", "pot", "histpotdf", "histgenevtdf")


def chunk_id(path: str) -> int:
    """Chunk number from '<prefix>_chunkN.df' (pool production) or '<prefix>_N.df' (grid job N)."""
    b = os.path.basename(path)
    m = re.search(r"chunk(\d+)\.df$", b) or re.search(r"_(\d+)\.df$", b)
    return int(m.group(1)) if m else 0


def read_table(path: str, table: str, slc_dir: Optional[str] = None) -> pd.DataFrame:
    """Concatenate all splits of one table in one chunk file; adds a leading 'chunk' index level.

    slc_dir (or env XSUM_SLC_DIR): directory of repaired `slc` sidecar files with the same basenames
    (see xsum.prep.repair); when given, the slc table is read from there instead of the production file."""
    slc_dir = slc_dir or os.environ.get("XSUM_SLC_DIR")
    if table == "slc" and slc_dir and os.path.exists(os.path.join(slc_dir, os.path.basename(path))):
        path, cid = os.path.join(slc_dir, os.path.basename(path)), chunk_id(path)
    with pd.HDFStore(path, mode="r") as st:
        keys = sorted((k for k in st.keys() if re.fullmatch(rf"/{table}_\d+", k)), key=lambda k: int(k.rsplit("_", 1)[1]))
        parts = [st[k] for k in keys]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts)
    return pd.concat({chunk_id(path): df}, names=["chunk"])


def read_production(pattern: str, tables: Iterable[str] = ("mcnu", "mcprim", "slc", "pfp", "hdr")) -> Dict[str, pd.DataFrame]:
    """pattern: glob of chunk files, e.g. 'data/prod/xsum_v10_14_02_03_chunk*.df'."""
    files = sorted(glob.glob(pattern), key=chunk_id)
    if not files:
        raise FileNotFoundError(pattern)
    out = {}
    for t in tables:
        parts = [read_table(f, t) for f in files]
        parts = [p for p in parts if len(p)]
        out[t] = pd.concat(parts) if parts else pd.DataFrame()
    return out


def to_parquet(pattern: str, out_dir: str, tables: Iterable[str] = TABLES) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for t in tables:
        df = read_production(pattern, [t])[t]
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        f = os.path.join(out_dir, f"{t}.parquet")
        df.to_parquet(f)
        written.append(f)
    return written
