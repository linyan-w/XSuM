import numpy as np
import pandas as pd

from xsum.constants import MCNU_INDEX, TRUTH_KIN
from xsum.data.splits import split_by_level
from xsum.io.cohpi_df import is_weight_column, physics_columns
from xsum.prep.table import build_table, reco_columns


def test_weight_column_detection():
    assert is_weight_column(("GENIEReWeight_SBN_v1_multisim_NormCCMEC", "univ_0"))
    assert is_weight_column(("expskin_Flux", "univ_3"))
    assert is_weight_column(("MINERvAE2p2h_SBN_v1_E2p2h_A_nu", "cv"))
    assert not is_weight_column(("true_p_mu", ""))
    assert not is_weight_column(("vtx_x", ""))


def test_physics_columns_flatten():
    cols = pd.MultiIndex.from_tuples([("true_p_mu", ""), ("knob", "univ_0"), ("flag", "")])
    df = pd.DataFrame([[1.0, 2.0, True], [2.0, 3.0, None]], columns=cols)
    out = physics_columns(df)
    assert list(out.columns) == ["true_p_mu", "flag"]
    assert out["flag"].dtype == float


def test_table_matches_one_to_one(cache):
    mcnu, evt = cache["mcnu"], cache["evt"]
    assert list(mcnu.index.names) == MCNU_INDEX
    table = build_table(mcnu, evt)
    assert len(table) == len(mcnu)
    n_sel = int(table.selected.sum())
    assert n_sel == int((evt.nuint_categ == 1).sum())          # every selected signal slice matched exactly once
    sel = table[table.selected == 1]
    for c in TRUTH_KIN:
        assert sel[c].notna().all()
    assert sel["range_p_mu"].notna().all()
    assert table.loc[table.selected == 0, reco_columns(evt)].isna().all().all()
    # truth copied into evt agrees with mcnu truth for matched rows
    key = pd.MultiIndex.from_arrays([evt.index.get_level_values(n) for n in MCNU_INDEX[:3]] + [evt.tmatch_idx.astype(int).values], names=MCNU_INDEX)
    m = key.isin(mcnu.index)
    np.testing.assert_allclose(evt.true_p_mu[m].values, mcnu.true_p_mu.reindex(key[m]).values, rtol=1e-6)


def test_split_by_file(cache):
    table = build_table(cache["mcnu"], cache["evt"])
    masks = split_by_level(table.index)
    total = sum(int(m.sum()) for m in masks.values())
    assert total == len(table)
    files = {k: set(table.index.get_level_values("file_idx")[m]) for k, m in masks.items()}
    assert not (files["train"] & files["test"]) and not (files["train"] & files["val"])
