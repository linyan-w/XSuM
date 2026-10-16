# SBND analysis notes for the XSuM surrogate

Reference extracted from two SBND technotes. Page numbers refer to the technote PDFs.

- **cohpi**: S. Oh, *Technical Note for Cross-Section Measurement of Charged-Current Coherent Charged Pion Production from νμ–Ar Interactions at SBND*, 5 June 2026, 45 pp. Target: Neutrino 2026 event-rate distributions.
- **kaon**: T. Wester, *Measurement of Charged Kaon Production in Charged-Current and Neutral-Current Neutrino Interactions with SBND*, November 2026 draft, 50 pp. Sections 3.2, 3.3.2, 3.4.1, 3.4.2, 4.1 and App. C are marked "planned but not yet written"; eight open items are listed on p. 3.

Coordinate system (cohpi p. 3): x = drift (x = 0 cathode), y = vertical, z = beam (z = 0 TPC front face). Flux mean energies in the cohpi FV (p. 3): νμ 0.76 GeV, ν̄μ 0.48 GeV, νe 0.90 GeV, ν̄e 1.02 GeV.

---

## 1. CC coherent π± (cohpi)

### 1.1 Signal definition (pp. 3–4)

Measured process: νμ Ar → μ− π+ Ar and ν̄μ Ar → μ+ π− Ar combined; lepton/pion charge is not distinguished.

| Requirement | Value |
|---|---|
| True vertex x | 10 cm < \|x\| < 190 cm |
| True vertex z | 10 cm < z < 450 cm |
| True vertex y, z < 250 cm | \|y\| < 190 cm |
| True vertex y, z > 250 cm | −190 cm < y < 100 cm |
| μ± | exactly 1, KE > 27 MeV (≈ 4 cm track) |
| π± | exactly 1, KE > 30 MeV (≈ 4 cm track) |
| p | 0 with KE > 20 MeV (≈ 0.45 cm track) |
| n, π0 | none |
| other final-state particles | allowed |

The \|x\| > 10 cm cut removes the cathode-material interaction peak at x = 0 (p. 9, Fig. 7). The y/z asymmetry removes a high-y, high-z region with E-field distortions (p. 3). Muon PFP-reconstruction efficiency vs true KE from a CCQE sample is shown in Fig. 3 (p. 4); values are not readable in the extraction.

### 1.2 Samples (pp. 4–5, 18)

| Sample | Definition | Exposure |
|---|---|---|
| Data, BNB + light, fixed development (runs 18255, 18259) | `data_SBND2026A_SBND2026A_gen2_run1_BNBLight_Data_FixedDev_Respin_v10_14_02_04_flatcaf_sbnd` | 4.62e18 POT |
| Data, off-beam + light | `data_SBND2026A_gen2_InTime-Run1_v10_14_02_02_flatcaf_sbnd` | gate-ratio normalised |
| MC, BNB + cosmics (GENIE CV) | `aurora_SBND2026A_gen2_BNBLight_prodgenie_corsika_proton_rockbox0p1_sbnd_CV_v10_14_02_03_flatcaf_sbnd` | cut-flow table normalised to ≈ 9.0e19 POT (p. 19) |
| Detector-variation MC | pending; flat 10 % used instead | — |
| Full Run 1 (publication) | — | 2e20 POT; ≈ 2,600 signal events in active volume (p. 3) |

Generator label on plots: GENIE v3.0.6, tune AR25_20i_00_000 (p. 35). Software: cafpyana, `analysis_village/cohpi/` (p. 5). Off-beam data are scaled by the BNB/off-beam trigger-gate ratio and then by (1 − P_beam), P_beam ≈ 0.075 (p. 5, Eq. 4). Blinding: data hidden in bins with > 10 % expected signal during development (p. 6).

### 1.3 Reconstruction objects and cut flow (pp. 6–19, 34)

Objects: Pandora slices (one slice = one event), PFPs with a vertex distance, length, track score and χ² PID; OpFlashes; barycenter flash-match (BCFM) score. χ² PID recomputed in cafpyana with the ellipsoidal modified box recombination model (p. 12):

| Parameter | MC | Data |
|---|---|---|
| α | 0.904 | 0.9366111 |
| β90 [kV/MeV·g/cm³] | 0.204 | 0.1835451 |
| R | 1.25 | 1.5676855 |
| collection-plane calibration [A.U./electron] | 0.0200727 | 0.02 |

χ²μ peaks (p. 12): ≈ 2 stopping μ/π (Bragg peak), ≈ 10 MIP-only (exiting μ, inelastic π), ≈ 42 stopping protons.

Definitions (pp. 6, 11–13):
- prong: PFP with start–vertex distance < 6 cm and length > 4 cm;
- track: prong with track score > 0.5;
- proton candidate: track with χ²μ ≥ 25 (cut requires both tracks χ²μ < 25);
- shower candidate: PFP with vertex distance < 50 cm and track score < 0.5 (50 cm allows for the photon conversion gap);
- muon candidate = longer track, pion candidate = shorter track. Correct for 82 % of signal (p. 6; "correct" = track whose cos θ is closer to the true muon's);
- angles are with respect to the detector z axis, not the neutrino direction (p. 14; BNB target centre at (−73.38, 0, −11000) cm);
- momenta from CSDA range of the full track length (p. 6); \|t\| from Eq. 5 (p. 6), infinite-target-mass approximation:
  \|t\| = (Σ_{μ,π} p⊥)² + (Σ_{μ,π} (E − p∥))².

Cut flow (Table 1, p. 19; events normalised to ≈ 9.0e19 POT MC; "SR" rows are the subset with \|t\|_reco < 0.04 (GeV/c)²):

| # | Cut | Events | Purity | Efficiency | SR events | SR purity | SR eff. |
|---|---|---|---|---|---|---|---|
| 1 | not clear cosmic & reco vertex in FV | 1,315,285 | 0.0006 | 0.9634 | | | |
| 2 | total PE > 1500 (OpFlashes, first time in [−5, 5] µs) | 1,127,170 | 0.0007 | 0.9622 | | | |
| 3 | BCFM score > 0.05 | 429,899 | 0.0017 | 0.8768 | | | |
| 4 | N(prongs) = 2 | 191,886 | 0.0027 | 0.6244 | | | |
| 5 | N(tracks) = 2 | 162,854 | 0.0029 | 0.5780 | | | |
| 6 | N(protons) = 0 (both χ²μ < 25) | 57,309 | 0.0074 | 0.5183 | | | |
| 7 | N(showers) = 0 | 44,608 | 0.0069 | 0.3768 | | | |
| 8 | cos θμ > 0.7 & cos θπ > 0.5 | 8,553 | 0.0325 | 0.3390 | 2,308 | 0.1057 | 0.2976 |
| 9 | cos θ_open(μ, π) > 0.5 | 3,631 | 0.0730 | 0.3232 | 1,324 | 0.1798 | 0.2902 |
| 10 | χ²p(μ) > 100 & χ²p(π) > 75 | 2,170 | 0.1147 | 0.3037 | 813 | 0.2768 | 0.2744 |
| 11 | charge-sphere cuts: Q1cm < 4000, Q2cm < 7000, Q3cm < 10000, Q4cm < 13000 A.U. | 1,229 | 0.1831 | 0.2744 | 519 | 0.4027 | 0.2549 |
| 12 | Q3cm / (Q4cm − Q3cm) < 6 | 1,204 | 0.1852 | 0.2720 | 513 | 0.4035 | 0.2524 |
| 13a | muon contained | 413 | 0.1913 | 0.0963 | 168 | 0.4345 | 0.0890 |
| 13b | muon and pion contained | 367 | 0.1880 | 0.0841 | 148 | 0.4324 | 0.0780 |
| 13c | muon and pion contained per TPC | 312 | 0.1987 | 0.0756 | 133 | 0.4361 | 0.0707 |

Purity = selected signal slices / selected neutrino slices; efficiency = selected signal slices / true signal interactions (p. 18). The charge spheres are the sum of hit integrals of PFP-associated hits within radius r of the reconstructed vertex (p. 16); Fig. 22 cut sequence is 1 cm, then 2 cm, 3 cm, 4 cm (p. 18).

Containment is **not** used in the final selection (p. 34). Instead, after cut 12, each of the two tracks must satisfy (DENT / cathode track-stitching veto, p. 34):
- start \|x\| > 10 cm, end \|x\| > 10 cm, start_x × end_x > 0.
Survival: BNB + cosmic MC 82.8 % (5371 / 6490), off-beam data 81.8 % (18 / 22), BNB data 98.6 % (68 / 69).

Expected signal at 2.5e20 POT (p. 18): ≈ 630 after cut 12, ≈ 185 after both-contained.

Regions (p. 7): signal region \|t\|_reco < 0.04 (GeV/c)², sideband \|t\|_reco > 0.06 (GeV/c)², buffer between. Before the sphere cuts the SR purity was ≈ 28 % (p. 16).

### 1.4 Observables, binning, response (pp. 20–25, 30)

Cross-section variables: p_μ (reco, range), cos θμ, cos θπ, all after cut 12 and the cathode veto, in the SR. \|t\|_reco is used only to define SR / sideband (p. 35).

Binning criteria (p. 24): completeness ≥ 60 % per bin (fraction of selected SR signal with true and reco value in the same bin) and smearing-matrix diagonal ≥ 68 % of events; 100,000 random binnings scanned per variable.

| Variable | Bin edges | Bins |
|---|---|---|
| p_μ [GeV/c] | 0, 0.34, 0.51, 3.0 | 3 |
| cos θμ | −1.0, 0.964, 0.988, 1.0 (θ ≈ 15.4°, 8.9°) | 3 |
| cos θπ | −1.0, 0.688, 0.885, 0.95, 1.0 (θ ≈ 46.5°, 27.7°, 18.2°) | 4 |

Smearing matrices (Fig. 27, p. 25; column = true bin, column sum 1; rows = reco bins in increasing order):

| p_μ | true bin 1 | 2 | 3 |
|---|---|---|---|
| reco 1 | 0.72 | 0.18 | 0.11 |
| reco 2 | 0.22 | 0.77 | 0.20 |
| reco 3 | 0.06 | 0.05 | 0.69 |

| cos θμ | true 1 | 2 | 3 |
|---|---|---|---|
| reco 1 | 0.77 | 0.14 | 0.03 |
| reco 2 | 0.14 | 0.78 | 0.18 |
| reco 3 | 0.09 | 0.08 | 0.79 |

| cos θπ | true 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| reco 1 | 0.78 | 0.05 | 0.01 | 0.01 |
| reco 2 | 0.22 | 0.84 | 0.12 | 0.08 |
| reco 3 | 0.00 | 0.08 | 0.77 | 0.15 |
| reco 4 | 0.00 | 0.04 | 0.10 | 0.76 |

Response matrices (efficiency × smearing, same layout, p. 25):

| p_μ | true 1 | 2 | 3 |
|---|---|---|---|
| reco 1 | 0.131 | 0.055 | 0.031 |
| reco 2 | 0.039 | 0.233 | 0.060 |
| reco 3 | 0.011 | 0.014 | 0.204 |

| cos θμ | true 1 | 2 | 3 |
|---|---|---|---|
| reco 1 | 0.170 | 0.048 | 0.008 |
| reco 2 | 0.030 | 0.256 | 0.053 |
| reco 3 | 0.020 | 0.026 | 0.230 |

| cos θπ | true 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| reco 1 | 0.018 | 0.013 | 0.005 | 0.003 |
| reco 2 | 0.005 | 0.224 | 0.039 | 0.022 |
| reco 3 | 0.000 | 0.021 | 0.251 | 0.043 |
| reco 4 | 0.000 | 0.009 | 0.032 | 0.223 |

Column sums of the response matrices (≈ 0.18, 0.30, 0.30 for p_μ; 0.22, 0.33, 0.29 for cos θμ; 0.02, 0.27, 0.33, 0.29 for cos θπ) are the SR selection efficiency per true bin. Unfolding method: not stated. In the systematics section the cross section per universe is computed as "the central value event rate multiplied by the response matrix in each multiverse" (p. 30, Fig. 31).

### 1.5 Systematics (pp. 26–33, 35–42)

Framework: SBNEventWeight reweighting; unisim (morph), multisim, multisigma; multisigma reduced to ±1σ and converted to Gaussian multisims; weights capped to [0, 10] (p. 26); cafpyana `makedf/getsyst.py`.

| Source | Treatment | Size / universes |
|---|---|---|
| BNB flux | G4BNB flux reweighted to Gen-1 flux per (Px, Py, Pz, parent, ν type), 200³ bins; then 13 multisim parameters (Table 2, p. 27): π±, K±, K0 production, horn current, skin depth, nucleon/pion inel./QE/total xsec | 1000 universes; mean reweight 1.002 (signal), 1.015 (all selected) |
| GENIE cross section | knobs of Table 3 (p. 29): QE/MEC, RES (MaCCRES, MvCCRES, Δ decay angles), non-RES backgrounds, DIS (Bodek-Yang), COH normalisation, NC elastic, FSI (MFP, CEx, Inel, Abs, PiProd for π and N) | NormCCCOH reduced from 100 % to 50 % (p. 29) |
| Geant4 reinteractions | Geant4Reweight (G4Cascade), π, p, K | 20 % on interaction rates (p. 30) |
| MC statistics | bootstrap, Poisson(λ = 1) weights | 1000 universes |
| Detector | flat on event rate (light yield + calorimetry); no variation samples yet | 10 % (13 % for the \|t\| plot that mixes SR and sideband, p. 35) |
| POT | flat | 2 % |
| Argon targets | flat | 1 % |
| Off-beam statistics | included in budget plots | not stated |

Dominant contributions (top-5 lists in Figs. 37, 39, 41, 43, pp. 36–42; values not readable): GENIE: CC Coh norm, Δ → Nπ angle, MA CC RES, MEC decay angle, FSI mean free path (sideband: Δ angle, MA/MV CC RES, FSI MFP, MEC decay angle). Flux: horn current depth, π+ production, nucleon QE xsec, K+ production, pion inelastic xsec. Geant4: proton, π+, π−, K+, K−. Detector effects actually varied: none (flat 10 %); the cathode track-stitching discrepancy (Fig. 35, p. 33) is handled by the veto of Sec. 1.3.

### 1.6 Numbers for surrogate validation

Composition of selected slices (Figs. 5, 6, 24, 36, 38; MC at 4.62e18 POT):

| Region | Cosmic | Others | Non-FV | DIS | RES | 2p2h | QE | NC | Signal | MC total | Data |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all, after cut 12 (p. 8) | 0.8 % | 1.2 % | 0.5 % | 13.0 % | 47.6 % | 5.1 % | 4.9 % | 8.4 % | 18.4 % | 62 | 69 |
| SR, after cut 12 (p. 8) | 0.1 % | 0.4 % | 0.4 % | 8.8 % | 30.8 % | 4.9 % | 4.3 % | 10.1 % | 40.3 % | 26 | 32 |
| sideband, after cut 12 (p. 8) | 1.7 % | 2.1 % | 0.8 % | 18.1 % | 61.2 % | 5.6 % | 4.9 % | 4.9 % | 0.8 % | 27 | 27 |
| SR, both contained (p. 21) | 0.0 % | 0.7 % | 0.7 % | 4.1 % | 25.7 % | 4.1 % | 0.7 % | 20.9 % | 43.2 % | 8 | 11 |
| all, final with cathode veto (p. 35) | 0.7 % (off-beam) | 0.9 % | 0.7 % | 12.9 % | 46.6 % | 4.2 % | 5.1 % | 8.6 % | 20.2 % | 62.3 | 68 |
| SR, final (p. 37) | 0.1 % | 0.5 % | 0.9 % | 9.7 % | 31.9 % | 3.2 % | 5.1 % | 9.8 % | 38.7 % | 29.7 | 32 |
| sideband, final (p. 37) | 1.6 % | 1.4 % | 0.4 % | 17.5 % | 61.3 % | 5.6 % | 4.5 % | 6.1 % | 1.6 % | 24.4 | 27 |

Composition at intermediate stages (MC signal fraction, pp. 11–17): after BCFM cut 0.1 % (44.5 % "others"); after 2 prongs 0.2 %; after 2 tracks 0.3 %; after 0 showers 0.7 %; after angle cuts 3.2 %; after χ²p cuts 7.2 % (11.4 % in the \|t\| plot, 27.6 % in the SR). Non-signal CC COH is 0.0 % everywhere.

Data/MC χ²/ndf for final event rates (pp. 35–41): \|t\| 2.56/6; p_μ SR 2.15/3, sideband 2.40/3; cos θμ 2.13/3, 2.02/3; cos θπ 2.11/4, 3.05/4.

Resolutions: only the smearing matrices of Sec. 1.4 are given. Muon/pion momentum or angle resolution as a number: not stated. Containment fractions: not stated directly; from Table 1 the efficiency drops from 0.272 to 0.096 when the muon must be contained, i.e. ≈ 35 % of selected signal muons are contained, ≈ 31 % have both tracks contained, ≈ 28 % both contained per TPC. Signal pions are softer than muons (Fig. 14, p. 14).

### 1.7 Reco variables the surrogate must reproduce (cohpi)

Names as in the XSuM `evt`/`slc` tables (`configs/data.yaml`).

| Stage | Variables |
|---|---|
| slice-level cuts | `is_not_clear_cosmic`, `is_fv` (from `vtx_x, vtx_y, vtx_z`), `total_pe`, `bcfm_score` |
| multiplicities | `n_prong`, `n_trk`, `n_proton`, `n_shower` (need per-PFP vertex distance, length, track score, χ²μ) |
| candidate kinematics | `long_dirz`, `short_dirz` (cos θ), `long_dir{x,y,z}`, `short_dir{x,y,z}` → `opening_angle`; `long_trk_len`, `short_trk_len` → `range_p_mu`, `range_p_cpi` → `reco_t` |
| PID | `long_trk_chi2_mu_new`, `short_trk_chi2_mu_new` (< 25), `long_trk_chi2_pro_new` (> 100), `short_trk_chi2_pro_new` (> 75) |
| vertex activity | `sum_integ_1cm`, `sum_integ_2cm`, `sum_integ_3cm`, `sum_integ_4cm`, `sphere_3_over_4m3` |
| cathode veto | `muon_start_x`, `muon_end_x`, `cpi_start_x`, `cpi_end_x` |
| optional (containment studies) | `is_muon_contained`, `is_cpi_contained`, all start/end points |

The muon/pion assignment is by length, so the long/short swap must be modelled jointly with both track lengths.

---

## 2. Charged kaon production (kaon)

### 2.1 Signal definition (pp. 5–7, 25, 29)

Measurements: CC-inclusive ν → K+ differential cross section in K+ momentum, NC ν → K+ rate, and a mono-energetic track-length study.

| Requirement | Value |
|---|---|
| True vertex FV (Table 2, p. 6) | −190 < x < 190 cm, −190 < y < 190 cm, 10 < z < 450 cm |
| K+ | produced at the neutrino vertex (primary); K+ from π± or K0L secondary interactions are not signal (p. 29 footnote, EVENT 7 p. 48) |
| K+ track | contained in the FV ("contained K+") |
| decay channel | K+ → μ+ νμ (BR 63.56 %) or K+ → π+ π0 (BR 20.67 %), daughter track also contained in the FV (p. 5) |
| K+ momentum / KE threshold | not stated (lowest bin edge 0.275 GeV/c set by the 7 cm parent-length cut, p. 26) |
| CC vs NC | separated at reco level by a muon candidate (Sec. 2.3); truth-level CC/NC definition for the NC measurement: not stated |

Fiducial volume may change to exclude the cathode and E-field non-uniform regions (OPEN 3, p. 6). No cathode cut is applied (x range spans both TPCs).

Predicted rates per 1e20 POT in this FV (Fig. 2, p. 6): K+ → μ+ 303.8, K+ → π+ 114.4, other contained K+ 293.9, uncontained K+ 154.6; true K+ momentum peaks near 0.4 GeV/c. Associated production thresholds: 911.1 MeV (NC), 1095.5 MeV (CC) (p. 4).

### 2.2 Samples (pp. 5, 42–50)

| Sample | Definition | Exposure |
|---|---|---|
| MC, BNB ν + cosmics CV (dev sample, `kcv`) | `aurora_SBND2026A_gen2_BNBLight_DevSample_prodgenie_corsika_proton_rockbox0p1_sbnd_CV_v10_14_02_03_flatcaf_sbnd` | 5.40e20 POT |
| MC, exclusive K+ filter (`kex`) | `aurora_SBND2026A_gen2_BNBLight_prodgenie_corsika_proton_rockbox0p1_sbnd_EX_kaon_v10_14_02_05_flat_caf_sbnd` | 2.67e21 POT (normalises only interactions with a kaon at the ν vertex) |
| Data, fixed development | `data_SBND2026A_SBND2026A_gen2_run1_BNBLight_Data_FixedDev_Respin_v10_14_02_04_flatcaf_sbnd` | 4.62e18 POT (run 18255 appears in EVENT 1) |
| Data, rolling development | `..._RollingDev_Respin_v10_14_02_04_flatcaf_sbnd` | not used |

Software (Table 13, p. 50): GENIE v3_06_02_sbn2, tune AR25_20i_00_000; Geant4 v4_10_6_p01g; sbndcode v10_14_02_05 (kex; kcv "TBD"); WireCell v0_32_1; Pandora v04_16_02; G4BNB "TBD". Caveats: single-kaon production MC absent, GENIE DIS/pythia bug, replacement sample pending (OPEN 2, p. 6). Data POT for the measurement is a placeholder ("X POT", p. 5). Sensitivities quoted at 1e20 and 3e20 POT.

### 2.3 Reconstruction objects and cut flow (pp. 6–13, 25)

Objects: Pandora slices with vertex, BCFM score, PFPs with `trackScore`, length, start/end, per-plane χ² PID for p, μ, π, K (computed from dE/dx vs residual range of hits in the last 26 cm of the track, excluding first and last hit, p. 8); track *pairs* (parent, daughter). ν score is not used (App. A, p. 32: mis-modelled, and it removes genuine K0L-induced K+ pairs).

Slice pre-selection and pair selection (Table 3, p. 7; per 1e20 POT; signal = contained K+ → μ+ and K+ → π+ for the pre-selection rows, then the single channel named):

| # | Cut | Value | Slices | Signal slices | Eff. [%] | Purity [%] |
|---|---|---|---|---|---|---|
| 1 | not a clear cosmic | — | 3.75e6 | 494 | 99.5 ± 0.1 | 0.01 |
| 2 | reco vertex in FV | Table 2 | 1.82e6 | 478 | 98.7 ± 0.1 | 0.03 |
| 3 | BCFM score | > 0.02 (unscored slices removed) | 5.89e5 | 395 | 93.7 ± 0.2 | 0.07 |
| 4 | N_PFP | ≥ 2 | 4.92e5 | 391 | 93.1 ± 0.2 | 0.08 |
| 5 | valid track pair | Table 4 | 8.93e4 | 181 | 43.2 ± 0.4 | 0.20 |
| K+ → μ+ | | | | | | |
| 6μ | daughter length, parent length | 30–100 cm, > 7 cm | 6.62e3 | 61.9 | 20.4 ± 0.4 | 0.93 |
| 7μ | BDT score | ε_BDT = 0.6 | 41.5 | 37.2 | 12.2 ± 0.3 | 89.71 |
| K+ → π+ | | | | | | |
| 6π | daughter length, parent length | 20–45 cm, > 7 cm | 7.52e3 | 7.83 | 6.8 ± 0.4 | 0.10 |
| 7π | BDT score | ε_BDT = 0.2 | 2.27 | 1.62 | 1.4 ± 0.2 | 71.30 |

Signal slice counts include multiple slices per interaction before the pair cut. In the text the length cuts are applied *after* the BDT cut (pp. 12–13); the table lists them before.

Valid track pair (Table 4, p. 7; p. 8):

| Criterion | Requirement |
|---|---|
| proximity | parent end to daughter start ≤ 2 cm |
| containment | both tracks' start and end points inside the FV of Table 2 |
| track score | parent > 0.5, daughter > 0.4 |
| parent length | ≥ 5 cm |
| daughter calorimetry | > 6 hits within the last 26 cm of the daughter end on ≥ 2 planes |

No vertex-proximity requirement on the pair (to keep K+ fragments after inelastic scatters, p. 25).

BDT (XGBoost, pp. 8–12, Table 5):

| Item | K+ → μ+ | K+ → π+ |
|---|---|---|
| parent inputs (3 planes each) | log χ²K, log(χ²K/χ²μ), log(χ²K/χ²p), log(χ²K/χ²π) | same |
| daughter inputs | log χ²μ (3 planes), length | log χ²π (3 planes), log(χ²π/χ²μ) (3 planes), length |
| n_features | 16 | 19 |
| n_signal pairs | 1982 | 213 |
| out-of-fold ROC AUC | 0.993 | 0.964 |
| cut (common efficiency across folds) | 0.6 | 0.2 |

χ² values with ndof ≤ 2 are set to NaN (p. 8). Hyperparameters: 400 trees, learning rate 0.05, max depth 5, γ = 2, min child weight 10, α = 0.1, λ = 0.1, subsample 0.8, colsample_bytree 0.7, binary:logistic, hist; 5-fold k-fold on 485,828 pairs; data pairs scored by one model chosen from a hash of run/subrun/event. The highest-scoring pair in the slice is the K+ candidate (Fig. 7, p. 12).

Additional criteria for the cross-section fits (Table 10, p. 25):

| Criterion | Requirement |
|---|---|
| muon candidate | PFP not in the highest-scoring pair, trackScore > 0.5, closer end within 10 cm of the slice vertex, length < 600 cm |
| CC tag | longest muon candidate > 100 cm (40 cm would give the purest NC sample, p. 28) |
| connected K+ | parent track start within 3 cm of the slice vertex; otherwise "disconnected" (rate-only bin) |

Mono-energetic sample (p. 29): K+ → μ+ selection plus daughter trackScore > 0.5.

### 2.4 Observables, binning, fit (pp. 24–28)

| Item | Value |
|---|---|
| CC differential variable | K+ range momentum (connected candidates); KE considered later (OPEN 8) |
| bin edges | 0.275, 0.4, 0.55, 1.2 GeV/c (3 bins, chosen for 3e20 POT so stat. error < syst.; scanned in 25 MeV/c steps) |
| extra rate-only bins | disconnected ν → K+ slices; K+ → π+ slices |
| NC measurement | rate only, via muon-candidate tag; RES sideband (CC/NC π0) planned |
| fit | Gundam v2.2.0 forward folding: signal strengths per true bin (flat priors), nuisance parameters (Gaussian priors), Poisson likelihood, Minuit |
| systematics in fit | Table 9 (p. 25): 35 GENIE multi-σ per-event splines (knots ±1, 2, 3σ; act on selected + truth); 18 GENIE multisims, 13 flux multisims, 5 Geant4 multisims → SVD linear dials; 4 calorimetry unisims → binned PCA dials (selected only) |
| expected precision, single rate parameter α (K+ → μ+) | σα = 0.258 (stat 0.173, syst 0.192) at 1e20 POT; 0.215 (stat 0.0998, syst 0.191) at 3e20 POT (p. 35) |
| toy coverage (3 bins) | 68.3 %: 0.700 / 0.693; 90 %: 0.901 / 0.904 at 1e20 / 3e20 POT (p. 27) |

No explicit smearing matrix is given; migration is handled inside the forward-folded fit with both reco and true bins (p. 24).

### 2.5 Systematics (pp. 14–23, 33–35)

Five groups: flux, GENIE, Geant4 reinteractions, calorimetry (detector), MC statistics. Flux and GENIE lead at low momentum, Geant4 at high momentum; background uncertainty is MC-stat dominated (Fig. 10, p. 14).

Fractional change [%] of the selected rate (Table 11, pp. 33–35; group totals and leading terms):

| Term | Type | K→μ signal | K→μ bkg | K→π signal | K→π bkg |
|---|---|---|---|---|---|
| Flux total | multisim | 11.85 | 8.42 | 10.56 | 9.57 |
| expskin | multisim | 7.64 | 3.38 | 6.98 | 6.64 |
| piplus | multisim | 7.39 | 4.10 | 5.99 | 4.07 |
| kplus | multisim | 4.19 | 5.03 | 4.18 | 4.51 |
| nucleonqexsec | multisim | 2.70 | 2.76 | 2.66 | 2.80 |
| GENIE total | | 11.72 | 8.51 | 13.14 | 18.95 |
| MaCCRES | multi-σ | 8.73 | 0.84 | 9.44 | 8.07 |
| MvCCRES | multi-σ | 6.88 | 0.61 | 7.26 | 6.47 |
| MaNCRES | multi-σ | 2.23 | 1.59 | 1.85 | 2.57 |
| RDecBR1eta | multisim | 2.01 | 0.25 | 2.20 | 2.07 |
| FrInel_pi | multi-σ | 0.26 | 5.19 | 0.42 | 0.03 |
| FrCEx_pi | multi-σ | 0.62 | 3.59 | 3.48 | 0.93 |
| NonRESBGvnCC2pi | multisim | 0.00 | 0.00 | 0.00 | 13.92 |
| Geant4 total | multisim | 6.55 | 9.37 | 6.11 | 8.73 |
| kplus reint. | multisim | 6.50 | 6.68 | 2.59 | 5.15 |
| proton reint. | multisim | 0.62 | 5.78 | 0.74 | 5.84 |
| piplus reint. | multisim | 0.27 | 2.94 | 5.07 | 1.91 |
| Calorimetry total | unisim | 1.61 | 4.39 | 11.54 | 39.81 |
| Gain C_cal | unisim | 1.28 | 0.00 | 6.93 | 39.09 |
| Recomb. α | unisim | 0.82 | 3.11 | 5.44 | 0.00 |
| Recomb. β90 | unisim | 0.47 | 3.11 | 7.32 | 7.55 |
| Recomb. R | unisim | 0.21 | 0.00 | 1.36 | 0.00 |
| Other interaction (2p2h C→Ar scaling, νe ratios, SPP low-Q²) | multi-σ | 0.00 | 0.01 | 0.01 | 0.00 |

Additional treatments: flux rate uncertainty 10–14 % with little momentum dependence (p. 14); extra ±1σ hadronisation uncertainty = (all-pythia − nominal) for DIS 2.3 < W < 3.0 GeV, not yet applied to DIS without a K+ (OPEN 5, pp. 14–16); Geant4 K+ knob is a 20 % normal throw on the reinteraction rate, weight log w = N ln s − (s − 1) L/λ̃ (p. 32), verified against K+–Ca data (Fig. 15, p. 17), no extra uncertainty added (unlike MicroBooNE's 40 %).

Detector effects varied (Sec. 2.3.4, Table 6, p. 17): EMB recombination α = 0.904 ± 0.008, β90 = 0.204 ± 0.008, R = 1.25 ± 0.02, gain C_cal ±2 %; each a unisim in which all track dE/dx and χ² PID scores are recomputed, correlations ignored. Light yield, E-field, wire response: not varied.

Calorimetry Noise Model (Sec. 2.4, pp. 17–23; not yet used in the sensitivities, OPEN 6): MC dE/dx distributions of stopping p and μ are narrower than data. Correction applied to MC hit dQ/dx only: correlated within-track throw (variance 1, correlation length 0.56–0.75 cm), scaled by amplitude a × (dQ/dx / dQ/dx0)^−0.34 with dQ/dx0 = 1.13e5 e/cm, plus a per-track common factor; hits with dQ/dx < 5.6e4 e/cm not smeared.

| TPC | plane | amplitude a | correlation length [cm] |
|---|---|---|---|
| 0 | 0 | 0.044 | 0.75 |
| 1 | 0 | 0.048 | 0.57 |
| 0 | 1 | 0.057 | 0.67 |
| 1 | 1 | 0.060 | 0.56 |
| 0 | 2 | 0.043 | 0.71 |
| 1 | 2 | 0.040 | 0.58 |

Control sample (p. 19): CC νμ 1μ1p by geometry only; pre-cuts, exactly two primary tracks near the vertex, muon candidate (longer) > 140 cm, proton candidate (shorter) contained and 10–50 cm; proton candidates 89 % p / 7 % π / 3 % μ, muon candidates 98.5 % μ; 1680 two-track and 660 three-track slices in 4.6e18 POT (three-track shortest track 17.7 % π vs 7.2 %). Proton-hypothesis χ² shape χ²/ndf default → Noise Model: plane 0 167/59 → 85/59, plane 1 151/59 → 73/59, plane 2 206/57 → 70/58; proton median χ²p MC/data 0.87/0.88/0.77 → 0.97/0.97/0.94 (Table 8, p. 20). Noise-amplitude fit uncertainties 11.5 %, 10.7 %, 9.0 % on planes 0, 1, 2 (p. 20).

### 2.6 Numbers for surrogate validation

Composition after the K+ → μ+ selection (Fig. 8, p. 13; slices per 1e20 POT, parent-length plot after daughter-length cut): K+ → μ+ 38.23, K+ → π+ 0.22, other contained K+ 0.41, uncontained K+ 0.03, K+ from π 1.30, K+ from K0 0.65, ν in FV no K+ 0.93, ν not in FV 1.30, cosmic 0.00. After both length cuts (daughter plot): 37.86 / 0.25 / 0.44 / 0.03 / 1.30 / 0.65 / 0.56 / 1.30 / 0.00. K+ → π+ selection (Fig. 9): K+ → μ+ 0.22, K+ → π+ 1.68 → 1.62, other contained 0.22, from K0 0.03, ν in FV no K+ 0.19.

Muon-candidate tag (Fig. 26, p. 28): K+ → μ+ 37.24, K+ → π+ 0.22, other contained 0.41, uncontained 0.03, from π 1.30, from K0 0.65, ν in FV no K+ 0.37, not in FV 1.30 per 1e20 POT.

K+ momentum (Fig. 23, p. 26): connected candidates 31.7 per 1e20 POT, (p_reco − p_true)/p_true peaked near 0; disconnected 5.6 per 1e20 POT, biased low (plot range −0.30 to +0.10; width not stated). Examples: disconnected fragment 0.39 vs true 0.54 GeV/c (EVENT 4, parent start 16.1 cm from vertex); wrongly disconnected 0.38 vs 0.42 GeV/c (EVENT 5, start 8.6 cm from vertex).

Mono-energetic daughter length (Fig. 27, p. 29): K+ decay-at-rest μ+ CSDA range 53.93 cm; fit peak μ = 53.76 cm, core σ = 1.63 cm, Michel-merge tail fraction 0.39 with λ = 6.4 cm, background fraction 7.0 %; 31.3 decay-at-rest slices per 1e20 POT, purity 96.3 % (includes secondary K+).

Data/MC χ²/dof of pre-cut variables (area normalised, pp. 8–9, 12): BCFM 70.4/17, N_PFP 315.2/17, parent length 49.0/39, parent track score 15.4/17, daughter track score 51.3/20, BDT score 36.0/19 (K→μ), 9.1/19 (K→π), ν score 1118.5/29.

Efficiency vs kinematics: not stated beyond the cut-flow table; angular resolution: not stated.

### 2.7 Reco variables the surrogate must reproduce (kaon)

Per slice: `is_not_clear_cosmic`, vertex (x, y, z) for the FV cut, `barycenterFM_score` (including the "no score" state), N_PFP.

Per PFP (XSuM `pfp` table): `trackScore`, length, start and end points (containment, parent-end–daughter-start distance ≤ 2 cm, start–vertex distance for the 3 cm connected / 10 cm muon-candidate criteria), per-plane χ²K, χ²μ, χ²p, χ²π with their ndof (hits in the last 26 cm; ndof ≤ 2 → NaN; daughter > 6 hits on ≥ 2 planes), range momentum under the kaon hypothesis, and PFP parent/id links to rebuild pairs. Derived: BDT score of the highest-scoring pair, parent/daughter lengths, muon-candidate length (CC/NC tag).

---

## Implications for XSuM

1. Shared by both analyses: not-clear-cosmic, reco vertex in FV, BCFM score (cuts 0.05 vs 0.02), PFP track score (0.5), PFP length, start/end points and containment, χ²μ and χ²p PID. The same `slc` + `pfp` tables serve both if every PFP carries start/end, length, trackScore and per-plane χ².
2. Fiducial volumes differ: cohpi excludes \|x\| < 10 cm and the high-y/high-z corner; kaon uses the full ±190 cm box. Store the vertex, not a flag.
3. cohpi-only inputs: `total_pe` (OpFlash sum, trigger emulation), `n_shower` (PFPs within 50 cm with track score < 0.5), the four `sum_integ_*cm` hit integrals, and the cathode veto on track x endpoints. The sphere charges are hit-level quantities; they are already in `slc` but cannot be re-derived from `pfp`, so the surrogate must generate them directly.
4. kaon-only inputs the cohpi block lacks: χ²K and χ²π on all three planes, χ² ndof (hit count in the last 26 cm), kaon-hypothesis range momentum, PFP parent/daughter links and end-to-start distances, a muon-candidate search among the remaining PFPs, and the BDT score, which is a deterministic function of the pair's χ² and lengths and should be recomputed, not generated.
5. χ² ndof per plane and the Noise Model (hit-level dQ/dx smearing) are not in the current dataframes; the ndof should be added to `pfp` if the CAF carries it, the Noise Model cannot be reproduced from tables and must be treated as an alternative MC response.
6. cohpi validation targets: the 3 × 3 / 3 × 3 / 4 × 4 smearing matrices and the per-bin SR efficiency (≈ 0.18–0.33); kaon targets: the pair efficiency (43 % valid pair, 12 % after BDT) and the 1.6 cm core width of the 53.9 cm decay-at-rest muon length.
7. Detector systematics are recombination/gain unisims (kaon) or a flat 10 % (cohpi); a surrogate conditioned on the calorimetry parameters (α, β90, R, C_cal) would cover both.
