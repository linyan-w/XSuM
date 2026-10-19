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
  sungbino's dCache scratch; by 2026-10-16 only 1,633 still existed (scratch is purged), and `flatcaf_587.root`
  is truncated (82 MB, unreadable). First production (`prod_v10_14_02_03_scratch1633`, six chunk files,
  18.6 GB, 1,632 files, about 5.4 M triggers, 9.8 M neutrinos) is in
  `/pnfs/sbnd/persistent/users/lwan/dataframe/prod_v10_14_02_03_scratch1633/`. It was made with the module
  version of commit a62f032 (kept in the clone as `analysis_village/xsum_v1`): no `mcdaught`, PFPs within 50 cm
  of the vertex only, no kaon PID branches. Chunk 0 had to be rerun: a pool worker died and cafpyana's pool
  never returns when a task is lost.
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

## Grid production of the official sample

From `/scratch/lwan/xsum/cafpyana` on sbndbuild03 (module committed in the clone, since grid mode ships
`git archive HEAD`; `envs/xrootd-5.6.9` is a link to an existing XRootD python build):

```
source /scratch/lwan/xsum/xsum_env.sh            # also sources /etc/profile.d/jobsub_lite.sh (JOBSUB_DROPBOX_SERVER_LIST)
export CAFPYANA_GRID_OUT_DIR=/pnfs/sbnd/scratch/users/lwan/cafpyana_out
htgettoken -a htvaultprod.fnal.gov -i sbnd
python run_df_maker.py -c analysis_village/xsum/configs/xsum_surrogate.py -l lists/sam_flatcaf_urls.txt -o xsum_v10_14_02_03 -ngrid 2000
```

Each job copies its ~375 flat CAFs with xrdcp, builds the cafpyana venv on the node, runs single-threaded and
writes `<out>_<job>.df` to `$CAFPYANA_GRID_OUT_DIR/dfs/<timestamp>__<out>/`. Pilot: one job over 375 files
(cluster 72523085, 2026-10-16 09:25): about 10 min of setup on the node, then 21.5 s per file single-threaded
(2 h 14 min for 375 files, 41.5 MB output, 4,863 events); so the full definition is about 4,500 CPU-hours.
Full submission: 2,000 jobs over the 748,887 remaining files (the pilot's 375 excluded; its output is part of
the dataset), 2026-10-16 12:05. Memory note: the merged-file test of the extended module peaked at
19 GB RSS (true-particle table + calo update of every PFP of 3,300 events); official files (13 events) need
well under 1 GB.

# Surrogated datasets

`scripts/generate_slc.py` samples slice records for a truth table with a trained surrogate;
`scripts/write_surrogate_df.py` writes them in the production format: one `xsum_surrogate_chunkN.df` per
production chunk with `mcnu`, `mcprim`, `hdr`, `histpotdf`, `histgenevtdf`, `pot` copied from the production
and `slc_0` replaced by the generated slices (index `(__ntuple, entry, rec.slc..index)`, `tmatch_idx` = the
generating neutrino, `tmatch_eff` / `tmatch_purity` NaN, `is_fv` from the generated vertex, selection flags
recomputed, `surrogate_topology_class` added), plus `file_splits.parquet` (train / val / test per input file,
same rule and seed as training). Comparisons with the real chain should use the test files only.
First dataset: `surrogate_v3_6ch` (M2 v3 trained on all six scratch chunks, applied to the same 9.8 M
neutrinos), in `/pnfs/sbnd/persistent/users/lwan/dataframe/surrogate_v3_6ch/` once written. No `pfp` table yet.

# The official production (`prod_v10_14_02_03_official`)

2,000 grid jobs over the SAM definition (375 flat CAFs each, pilot excluded), submitted 2026-10-16 12:05,
all finished by 2026-10-17; archived with their logs in
`/pnfs/sbnd/persistent/users/lwan/dataframe/prod_v10_14_02_03_official/` (2,000 `xsum_v10_14_02_03_N.df`,
79 GB, world-readable; the pilot's 375 files are in `.../cafpyana_out/dfs/2026_10_16_092446__xsum_pilot/` on
scratch). 3,284 of the 749,262 input files (0.44 %) were skipped by 461 jobs: about 3,000 failed to copy to
the worker (`No such file` after xrdcp), about 300 are unreadable (`expected Chunk of length ...`); both
lists are recoverable from the job logs for a small follow-up submission. Each file holds the full schema
(`mcnu, mcprim, mcdaught, slc, pfp, hdr, pot, histpotdf, histgenevtdf`; about 42 MB, 46 % of it `mcdaught`).
Training tables: built on sbndbuild03 with `scripts/postgrid_official.sh` in 20 groups (job number modulo 20),
4.2 GB of Parquet, merged here with `scripts/merge_table_groups.py` into
`data/prod_v10_14_02_03_official/table/` (index level `chunk` = job number). The raw files are not kept on
this machine.

**Known defect of the official production and its repair.** The module version used on the grid (commits
after the all-PFP change) counted `n_shower` over all PFPs of a slice instead of those within 50 cm of the
vertex, so `pass_2trk` never fired and the slice two-track block (`reco_t`, `opening_angle`, containment,
`muon_*`/`cpi_*` points, `range_p_*`) and the `pass_*` flags are empty in `prod_v10_14_02_03_official`. The
`pfp`, `mcnu`, `mcprim`, `mcdaught` and the rest of `slc` are correct. `xsum.prep.repair.repair_slc` rebuilds
the block from `pfp` exactly as cafpyana does; `scripts/repair_official_slc.py` writes repaired `slc` sidecar
files (`prod_v10_14_02_03_official_slcfix/` on persistent) which the loader uses when `XSUM_SLC_DIR` points
at them. The scratch production (module v1) is not affected. The module is fixed for future productions.
