"""Draft cafpyana df-maker config for the full XSuM surrogate dataset (not runnable here: needs cafpyana
and the flat CAFs on dCache). Place it in cafpyana/analysis_village/cohpi/configs/ and run
    python run_df_maker.py -c analysis_village/cohpi/configs/xsum_surrogate.py -l <sample list> -o xsum_surrogate

Compared with cohpi_all_weight_signal.py it keeps every neutrino (not only signal) with its true vertex and
GENIE labels, the primary particle table, and every slice with the selection stages as columns.
See docs/PROJECT.md section 8.
"""
# from makedf.makedf import make_hdrdf, make_mcdf, make_mcprimdf, make_potdf_bnb
# from analysis_village.cohpi.makedf.make_cohpidf_all_weight_signal import make_cohpi_nudf, make_cohpidf_slc


def make_xsum_nudf(f):
    """All neutrinos: cohpi kinematics + vertex + labels (labels are for holdouts, never model inputs)."""
    from makedf.makedf import make_mcdf
    from analysis_village.cohpi.makedf.make_cohpidf_all_weight_signal import make_cohpi_nudf
    nudf = make_cohpi_nudf(f)
    mcdf = make_mcdf(f)
    for col in [("position", "x"), ("position", "y"), ("position", "z"), ("momentum", "x"), ("momentum", "y"),
                ("momentum", "z"), ("pdg", ""), ("iscc", ""), ("genie_mode", ""), ("w", ""), ("Q2", ""),
                ("nmu_27MeV", ""), ("npi_30MeV", ""), ("np_20MeV", ""), ("npi0", ""), ("nn_0MeV", "")]:
        name = "true_" + "_".join(c for c in col if c)
        nudf[(name, "")] = mcdf[col] if col[1] else mcdf[col[0]]
    return nudf


def make_xsum_primdf(f):
    from makedf.makedf import make_mcprimdf
    return make_mcprimdf(f)


def make_xsum_slcdf(f):
    """Every non-clear-cosmic slice with the selection stages as boolean columns.

    TODO: refactor make_cohpidf_slc so that steps (9) a-d set flags instead of filtering, then merge truth
    for every slice through tmatch_idx as it does now.
    """
    from analysis_village.cohpi.makedf.make_cohpidf_all_weight_signal import make_cohpidf_slc
    return make_cohpidf_slc(f)


def _hdr(f):
    from makedf.makedf import make_hdrdf
    return make_hdrdf(f)


def _pot(f):
    from makedf.makedf import make_potdf_bnb
    return make_potdf_bnb(f)


DFS = [make_xsum_nudf, make_xsum_primdf, make_xsum_slcdf, _hdr, _pot]
NAMES = ["mcnu", "mcprim", "evt", "hdr", "pot"]
