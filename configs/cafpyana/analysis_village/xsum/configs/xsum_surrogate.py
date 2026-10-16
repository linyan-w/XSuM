"""cafpyana df-maker config for the XSuM surrogate dataset.

Copy analysis_village/xsum/ into a cafpyana checkout (branch feature/sungbin_technote_plots or later) and run
    python run_df_maker.py -c analysis_village/xsum/configs/xsum_surrogate.py -l <sample list> -o <out prefix> -ncpu N

Tables (one row is):
  mcnu   every true neutrino: full make_mcdf truth (vertex, 4-momentum, GENIE labels, particle counts,
         leading mu/pi/p/e) + the cohpi kinematics and nuint_categ
  mcprim every GENIE primary (post-FSI) of every neutrino: pdg, genE, genp, start, end, length, contained
  slc    every non-clear-cosmic slice: cohpi slice/two-track variables and the selection stages as flags
  pfp    every PFP of those slices: track/shower reco, chi2 PID (incl. re-calibrated), truth match
  hdr, pot, histpotdf, histgenevtdf as in the cohpi config
"""
from makedf.makedf import make_hdrdf, make_potdf_bnb
from analysis_village.xsum.makedf.make_xsumdf import make_xsum_nudf, make_xsum_primdf, make_xsum_slcdf, make_xsum_pfpdf

DFS = [make_xsum_nudf, make_xsum_primdf, make_xsum_slcdf, make_xsum_pfpdf, make_hdrdf, make_potdf_bnb]
NAMES = ["mcnu", "mcprim", "slc", "pfp", "hdr", "pot"]
