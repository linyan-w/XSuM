import numpy as np
import pandas as pd

from xsum.prep.pfpset import WALL_BAND, WALL_REF
from xsum.train.m2 import apply_walls


def _frame(end, flag, off):
    d = {"long_wall": [0.0] * len(end), "long_woff": [0.0] * len(end), "long_trk_len": [50.0] * len(end),
         "vtx_x": [0.0] * len(end), "vtx_y": [0.0] * len(end), "vtx_z": [100.0] * len(end),
         "long_dirx": [0.0] * len(end), "long_diry": [0.0] * len(end), "long_dirz": [1.0] * len(end)}
    for p in ("muon", "cpi"):
        for a in "xyz":
            d[f"{p}_end_{a}"] = [10.0] * len(end); d[f"{p}_end_wall_{a}"] = [0.0] * len(end); d[f"{p}_end_woff_{a}"] = [0.0] * len(end)
    d["muon_end_x"] = end; d["muon_end_wall_x"] = flag; d["muon_end_woff_x"] = off
    return pd.DataFrame(d)


def test_apply_walls_cohpi_ends():
    lo, hi = WALL_REF["x"]; bl, bh = WALL_BAND["x"]
    #          inside      flagged ok   flagged off>bh  flagged off<bl  unflagged beyond +   unflagged beyond -   unflagged in band  nan
    end = [50.0, 180.0, 180.0, 180.0, hi + 30.0, lo - 12.0, hi + 0.5 * bh, np.nan]
    flag = [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    off = [0.0, 0.5, bh + 5.0, bl - 3.0, 0.0, 0.0, 0.0, 0.0]
    out = apply_walls(_frame(end, flag, off))
    e = out["muon_end_x"].to_numpy()
    assert e[0] == 50.0                                  # inside, unflagged: untouched
    assert np.isclose(e[1], hi + 0.5)                    # flagged, offset within the band, snapped onto the upper wall
    assert np.isclose(e[2], hi + bh)                     # offset clipped to the band
    assert np.isclose(e[3], hi + bl)
    assert np.isclose(e[4], hi - 30.0)                   # mirrored inside about the upper wall
    assert np.isclose(e[5], lo + 12.0)                   # mirrored inside about the lower wall
    assert np.isclose(e[6], hi + 0.5 * bh)               # within the band: left alone
    assert np.isnan(e[7])
    # the other columns are untouched
    assert (out["cpi_end_x"] == 10.0).all() and (out["long_trk_len"] == 50.0).all()
