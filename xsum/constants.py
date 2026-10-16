"""Shared constants: index names, column groups, PDG masses (GeV)."""

MCNU_INDEX = ["file_idx", "__ntuple", "entry", "rec.mc.nu..index"]
EVT_INDEX = ["file_idx", "__ntuple", "entry", "rec.slc..index"]
ENTRY_INDEX = ["file_idx", "__ntuple", "entry"]

TRUTH_KIN = ["true_E_nu", "true_p_mu", "true_p_pi", "true_cos_theta_mu", "true_cos_theta_pi", "true_t"]
TRUTH_ALL = TRUTH_KIN + ["nuint_categ", "ind"]

# nuint_categ codes from cafpyana make_cohpi_nudf
NUINT_CATEG = {-1: "out of FV", 0: "NC", 1: "signal", 2: "CC COH (non-signal)", 3: "CCQE",
               4: "2p2h", 5: "RES", 6: "DIS", 8: "other"}

# second-level-name prefixes that identify systematic-weight column groups in the cohpi df
WEIGHT_PREFIXES = ("GENIEReWeight", "MINERvAE2p2h", "MiscInteractionSysts", "NOvAStyleNonResPionNorm")
WEIGHT_SUFFIXES = ("_Flux", "_Geant4")

MASS = {"mu": 0.1056584, "pi": 0.13957039, "p": 0.938272, "n": 0.9395654, "pi0": 0.1349768}
