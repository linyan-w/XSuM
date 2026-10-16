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

# The kaon analysis dataframes (`kcv_reco`)

`/pnfs/sbnd/scratch/users/twester/cafpyana_out/dfs/2026_09_23_192946__kcv_reco/`, 6,000 files, 9.1 GB,
mounted on sbndbuild03; same v10_14_02_03 BNB + cosmics CV flat CAFs. Produced by
`/exp/sbnd/app/users/twester/analysis/kaon/make_kaon_df.py` plus a second-stage converter. Per file (file 0):

| key | one row is | rows |
|---|---|---|
| `true_type_0` | every true neutrino: `true_type` (1 K+→mu signal, 2 K+→pi, 3/4 other K+, 98 no K+ in FV, 99 out of FV), `is_cc, genie_mode, is_true_fv, true_E_nu, W, Q2, x, y, genie_inttype, target_pdg`, particle counts | 5,830 |
| `slice_0` | every non-clear-cosmic FV slice: `nu_score, barycenterFM_score, tmatch_idx, vertex` | 6,676 |
| `track_0` | every PFP of those slices: start/end, `trackScore`, length, chi2 (K, mu, p, pi; 3 planes), range P, truth match (pdg, G4ID, parent, interaction_id, genE, start/end) | 19,711 |
| `pair_0`, `true_kaon_0` | kaon candidate pairs, true kaons | 2,578 / 123 |
| `hdr_0`, `file_0`, `histpotdf_0`, `histgenevtdf_0` | triggers, provenance, POT | |

Selection-free on the reco side (this is the Tier 1 / Tier 2 schema XSuM adopts), but the truth side keeps
only labels and counts: the producer loads `rec.mc.nu.prim` and the vertex and drops `position`, `momentum`,
`genp`, `start`, `end` before writing. 35 % of PFP rows have no track fit (`trackScore` = −5).
In file 0: 1,882 / 5,830 neutrinos have a matched slice (1,057 / 1,267 in the FV).

# The XSuM production (`configs/cafpyana/analysis_village/xsum/`)

Runs on sbndbuild03 from local disk `/scratch/lwan/xsum/` (shallow clone of `sungbinoh/cafpyana` branch
`feature/sungbin_technote_plots`, environment `xsum_env.sh`: cvmfs spack hdf5/xrootd/root + an existing
py3.10 cafpyana venv, no new venv; dCache refuses in-place rewrites, so code, logs and HDF5 outputs never go
to `/pnfs` directly). Finished dataframes are copied to `/pnfs/sbnd/persistent/users/lwan/dataframe/`.
`run_chunks.sh` runs 6 `run_df_maker.py` processes with 5 workers each; one merged flat CAF (2.5 GB,
3,300 events) takes 105 s on an idle core, about 5 min under the 30-worker load; output is 12 MB per merged
file (3.7 kB per event). Output files `<prefix>_chunkN.df` hold keys `<table>_<k>` per 2 GB split;
`xsum.io.xsum_df.read_production` concatenates them with a leading `chunk` index level.

Input samples:

- cafpyana sample list `prod_2026/v10_14_02_03/..._flatcaf_sbnd_merged.txt`: 2,996 merged flat CAFs on
  sungbino's dCache scratch; by 2026-10-16 only 1,633 still existed (scratch is purged). First production
  uses those (about 5.4 M events, 55 % of the sample).
- SAM definition `aurora_SBND2026A_gen2_BNBLight_prodgenie_corsika_proton_rockbox0p1_sbnd_CV_v10_14_02_03_flatcaf_sbnd`
  (sbndpro, 2026-03-02): 749,262 unmerged flat CAFs, 7.9 TB, 9,935,524 events, on
  `/pnfs/fnal.gov/usr/sbn/data_add/sbn_nd/aurora/mc/v10_14_02_03/...`, xrootd URLs via
  `samweb list-file-locations --defname=... --schema=root` (SAM host is `samsbnd.fnal.gov`; set
  `SAM_WEB_HOST`/`SAM_WEB_BASE_URL`, the default `samweb.fnal.gov` does not resolve on sbndbuild03). This is
  the full official sample for the second production; about 13 events per file, so the per-file overhead
  dominates (about 2 s per file per worker measured with 5 workers). Tables: `mcnu` (every neutrino, full
`make_mcdf` truth + cohpi kinematics + `nuint_categ`), `mcprim` (every primary), `mcdaught` (every Geant4
daughter of every primary, from `rec.true_particles`: pdg, G4ID, parent, energies, momentum, start/end,
length, containment, start/end process; kaon and pion decay modes and pi0 photons are derived from it),
`slc` (every non-clear-cosmic slice with the cohpi variables and selection stages `pass_*` as flags,
`pass_cohpi` = full selection incl. sphere-charge and cathode-crosser cuts from the notebook, `in_sr` =
signal region; the cohpi two-track block is defined on PFPs within 50 cm of the vertex as in the cohpi code),
`pfp` (every PFP of those slices, no vertex-distance cut: track, shower, chi2 for mu/p/pi/K on all planes
and re-calibrated mu/p, range and MCS momenta incl. kaon, PFP parent/id links, truth match), `hdr`, `pot`.
The kaon-analysis candidate pairs are rebuilt from `pfp` (parent/id, kaon chi2, endpoints) and the kaon
truth labels from `mcprim` + `mcdaught`.
