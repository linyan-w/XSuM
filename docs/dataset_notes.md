# What is in the example dataset

File: `cohpi_all_weight_signal_mc_no_dirz_cont_var.df` (2.4 GB, pandas HDF5, `fixed` format, six keys).
Produced by cafpyana `analysis_village/cohpi/configs/cohpi_all_weight_signal.py` →
`make_cohpidf_all_weight_signal.py` from the SBND 2026 gen2 BNB + cosmics CV flat CAFs (v10_14_02_03).
Reference analysis notebook: `analysis_village/cohpi/plotting/syst/ana_syst_dfs.ipynb`.

| key | rows | columns | one row is |
|---|---|---|---|
| `mcnu` | 3,962 | 22,573 | a true **signal** neutrino (CC coherent pi+, `nuint_categ == 1`) |
| `evt` | 18,650 | 22,632 | a Pandora slice that passed the cohpi selection (all categories, incl. cosmic) |
| `hdr` | 7,500,530 | 11 | a trigger (`run, subrun, evt, pot, ngenevt, ...`) |
| `histpotdf` | 565,463 | 1 | a subrun-file (`TotalPOT`) |
| `histgenevtdf` | 565,463 | 1 | a subrun-file (`TotalGenEvents`, 50 per file) |
| `pot` | 0 | 2 | empty for MC (`TOR860, TOR875` are data toroids) |

Columns are a two-level MultiIndex. The 8 (`mcnu`) / 67 (`evt`) physics columns have an empty second
level; the rest are systematic weights (`GENIEReWeight_SBN_v1_multisigma_*` with `ps1..ms3, cv` or `morph`,
`GENIEReWeight_SBN_v1_multisim_*` with 100 universes, `*_Flux` and `reinteractions_*_Geant4` with 1000,
`MINERvAE2p2h_*`, `MiscInteractionSysts_*`, `NOvAStyleNonResPionNorm_*`). Loading a key takes about 1 s.
`scripts/slim_cohpi_df.py` drops the weights and writes Parquet.

## Indices and matching

- `mcnu` index: `(file_idx, __ntuple, entry, rec.mc.nu..index)`.
- `evt` index: `(file_idx, __ntuple, entry, rec.slc..index)`; `tmatch_idx` is the `rec.mc.nu..index` of the
  truth-matched neutrino (−999 = no match, 624 rows = cosmic slices).
- A selected slice maps to a signal neutrino through `(file_idx, __ntuple, entry, tmatch_idx)`:
  1,463 of 18,650 slices, exactly one slice per matched neutrino, one slice per entry.
- Selection efficiency for signal: 1,463 / 3,962 = 36.9 %.
- `(run, subrun, evt)` for a row come from `hdr` through the first three index levels.

## `mcnu` physics columns

`true_E_nu, true_p_mu, true_p_pi` (GeV), `true_cos_theta_mu, true_cos_theta_pi`, `true_t` (GeV², |t|
from E and the mu/pi 4-momenta; 999999 when undefined), `nuint_categ`, `ind` (= `rec.mc.nu..index`).
No vertex, no primary particle list: `make_cohpi_nudf` computes these from `make_mcdf` and keeps only them.

`nuint_categ`: −1 out of FV, 0 NC, 1 signal, 2 non-signal CC COH, 3 CCQE, 4 2p2h, 5 RES, 6 DIS,
8 other. Signal = in FV (`InFV_nohiyz`: |x| in (10, 190), z in (10, 450), |y| < 190 for z < 250 and
−190 < y < 100 for z > 250 cm) and exactly 1 mu (KE > 27 MeV), 1 pi± (KE > 30 MeV), 0 p (KE > 20 MeV),
0 pi0, 0 n.

## `evt` physics columns

Slice: `is_fv, bcfm_score, nu_score, is_not_clear_cosmic, n_prong, n_trk, n_proton, n_shower` (after the
selection all slices have `n_prong = n_trk = 2`, `n_proton = n_shower = 0`), `tmatch_*`, `vtx_{x,y,z}`,
`total_pe` (sum of flash PE within ±5 µs), `sum_integ_{1,2,3,4}cm` (collection-plane hit integral within r
of the vertex; NaN when no hit, 23 % for 1 cm), `sphere_3_over_4m3`.
Tracks, ordered by length: `long_*`/`short_*` direction, length, `*_pid` (arctan-scaled chi2 mu/p ratio),
`*_chi2_mu, *_chi2_pro` and `*_new` (re-calibrated dE/dx).
Two-track variables with the muon = longer track assumption: `reco_t, opening_angle, is_muon_contained,
is_cpi_contained, muon_{start,end}_{x,y,z}, cpi_{start,end}_{x,y,z}, range_p_mu, range_p_cpi` (both from
the muon range table).
Truth copied from the matched neutrino: `true_*`, `nuint_categ, ind` (NaN for cosmic slices;
`true_p_pi` NaN also when the matched neutrino has no pi±).

`nuint_categ` of selected slices: RES 9,456; out-of-FV 1,926; DIS 1,702; signal 1,463; CCQE 1,227;
NC 1,131; 2p2h 1,121; cosmic 624.

## Measured response (selected signal)

| quantity | 16 % | 50 % | 84 % |
|---|---|---|---|
| `range_p_mu / true_p_mu` | 0.44 | 0.89 | 1.02 |
| `range_p_cpi / true_p_pi` | 0.33 | 0.57 | 0.85 |
| `long_dirz − true_cos_theta_mu` | −0.010 | −0.001 | 0.006 |

Muon contained 33 %, pion contained 76 %; correlation of `reco_t` with `true_t` is 0.23.
