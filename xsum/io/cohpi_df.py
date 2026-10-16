"""Read the cafpyana cohpi HDF5 dataframes and slim them to their physics columns.

The analysis df stores every key in pandas 'fixed' format with a two-level column MultiIndex. Physics
columns have an empty second level; systematic weights (about 22k columns) have 'univ_i', 'ps1'.. or 'morph'
as second level. Loading a key reads the whole frame (about 1 s each for the example file).
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from xsum.constants import WEIGHT_PREFIXES, WEIGHT_SUFFIXES

DEFAULT_KEYS = ("mcnu", "evt", "hdr", "histpotdf", "histgenevtdf", "pot")


def hdf_keys(path: str) -> list:
    with pd.HDFStore(path, mode="r") as st:
        return [k.lstrip("/") for k in st.keys()]


def is_weight_column(col) -> bool:
    name = col[0] if isinstance(col, tuple) else col
    sub = col[1] if isinstance(col, tuple) and len(col) > 1 else ""
    if sub != "":
        return True
    return name.startswith(WEIGHT_PREFIXES) or name.endswith(WEIGHT_SUFFIXES)


def physics_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop systematic-weight columns and flatten the column MultiIndex to plain names."""
    if isinstance(df.columns, pd.MultiIndex):
        keep = [c for c in df.columns if not is_weight_column(c)]
        out = df[keep].copy()
        out.columns = [c[0] for c in keep]
    else:
        out = df[[c for c in df.columns if not is_weight_column(c)]].copy()
    # bool columns stored as object (NaN-able) -> float so Parquet and torch can take them
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(float)
    return out


def load_key(path: str, key: str, slim: bool = True) -> pd.DataFrame:
    df = pd.read_hdf(path, key=key)
    return physics_columns(df) if slim else df


def weight_groups(df: pd.DataFrame) -> Dict[str, int]:
    """Name -> number of columns for each systematic-weight group (for documentation / audits)."""
    out: Dict[str, int] = {}
    for c in df.columns:
        if is_weight_column(c):
            name = c[0] if isinstance(c, tuple) else c
            out[name] = out.get(name, 0) + 1
    return dict(sorted(out.items()))


def slim_to_parquet(path: str, out_dir: str, keys: Iterable[str] = DEFAULT_KEYS) -> Dict[str, str]:
    """Write <out_dir>/<key>.parquet with physics columns only; returns key -> file."""
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    for key in keys:
        df = load_key(path, key, slim=True)
        if df.empty and len(df.columns) and df.index.nlevels == 1:
            # the MC 'pot' key is an empty frame with a RangeIndex; keep it for completeness
            pass
        f = os.path.join(out_dir, f"{key}.parquet")
        df.to_parquet(f)
        written[key] = f
    return written


def load_cache(cache_dir: str, keys: Optional[Iterable[str]] = None) -> Dict[str, pd.DataFrame]:
    keys = list(keys) if keys is not None else [k[:-8] for k in sorted(os.listdir(cache_dir)) if k.endswith(".parquet")]
    return {k: pd.read_parquet(os.path.join(cache_dir, f"{k}.parquet")) for k in keys}
