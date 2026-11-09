# CC1p0pi containment closure (test files of jobs 0-199)

Script: `scripts/check_cc1p_containment.py` (run on sbndbuild03 with the shared `xsum_testsplit.py`).
Selection: PFP-level 2 prongs (dist < 6 cm, length > 4 cm, trackScore > 0.5), exactly 1 proton candidate
(chi2_mu_new_I2 > 25), no shower within 50 cm; containment = both track ends in the cohpi FV box.
Assumed truth signal: true vertex in the cohpi FV, numu CC, 1 mu (KE > 27 MeV), 1 p (KE > 20 MeV), 0 pi+- (> 30 MeV), 0 pi0.
179,331 test neutrinos, 9,931 true signal.

| selection | Pandora eff | XSuM eff | Pandora purity | XSuM purity |
|---|---|---|---|---|
| 2-prong | 0.390 | 0.397 | 0.482 | 0.487 |
| 2-prong, mu contained | 0.154 | 0.160 | 0.560 | 0.564 |
| 2-prong, p contained | 0.354 | 0.361 | 0.634 | 0.639 |
| 2-prong, both contained | 0.141 | 0.147 | 0.560 | 0.562 |

Straight-line end points on real tracks change the contained efficiencies by at most +0.003.

## Collaborator's CC1p0pi notebook (M. Fernandez, 2026-11-05): divergence at "2prong-contained"

Their cut: vertex and start/end of the two longest good PFP tracks (`trk_start_*`, `trk_end_*`) inside the Gen1 FV,
and all five points on the same side of the cathode (per-TPC). Truth signal requires the true muon and proton end
points inside one TPC's FV. On 10 jobs: real eff 0.446 -> 0.222 (purity 0.25 -> 0.42); surrogate 0.467 -> 0.161
(purity 0.25 -> 0.27); real/surrogate selected-event overlap drops from 51% to 30% at that stage.

Cause: the surrogate's transverse track directions are not tied to the event. Per-event Pearson correlation between
Pandora and XSuM on the same test neutrinos: long_dirx +0.06, long_diry +0.05 (sign of dir_x agrees 53%, random is
50%), against vtx +0.91..0.94 and long_dirz +0.34. The input features carry the leading particles' cos(theta) and
p_T but no azimuth (no px, py), so the model cannot know whether a track heads to the cathode, an x wall or a y wall.
It reproduces the containment *rate* (closure in the table above) but not *which* events are contained, which is
what a truth-contained signal definition tests. Straight-line end points are a minor effect (<=0.3%).

## After M3 (particle-set inputs), 2026-11-05

Surrogate `surrogate_official_m3_pfp` (M2 + M4 retrained with the 64-d embedding of all final-state primaries).
Same 200 jobs, same script (`accept_m3.txt`, raw per group `accept_m3_raw.txt`):

| stage | real eff / pur | v3 eff / pur / overlap | M3 eff / pur / overlap |
|---|---|---|---|
| 2prong | 0.637 / 0.132 | 0.652 / 0.133 / 0.51 | 0.613 / 0.125 / 0.53 |
| 2prong-contained | 0.626 / 0.431 | 0.387 / 0.243 / 0.31 | 0.543 / 0.339 / 0.42 |
| 2prong-mup-PID (final) | 0.346 / 0.890 | 0.202 / 0.502 / 0.23 | 0.305 / 0.780 / 0.41 |

Most of the gap is closed; still ~13% low efficiency and ~20% excess background at the containment cut.
Per-event real-vs-generated correlation of long_dirx 0.06 -> 0.57, muon_end_y 0.58 -> 0.95.

## End-point models and the signal definition, 2026-11-07

End-point precision models, same 200 jobs (`accept_m3b/m3e/m3g/m3w.txt`, diagnostics `diag_*_summary.txt`):
M3b (bend fix) and M3e (end point as a direct target + PFP-to-particle cross-attention) left containment unchanged
(longest-track end vs real ~20 cm 3D median). M3g (Geant4 start/end/length/flags on the particle tokens) cut it to
7 cm; M3w (+ track ends on the TPC wall as flag + offset, + straight-line TPC exit point) reproduces the real
pile-up of exiting track ends on the wall and halves the exiting-muon leak into the containment cut.
The errors are a resolution, not a bias (signed medians < 0.5 cm); real ends plus the XSuM residuals with random
signs reproduce the XSuM containment rates.

Signal definition. With truth containment of the mu / p end points the signal sits on one side of a sharp boundary
set by stochastic end points: any finite end-point resolution only moves signal out (and exiting background in),
while the reco containment cut still selects the same mixture. Primary acceptance definition from now on (decided
2026-11-07): numu CC 1mu1p0pi, p_mu and p_p < 1 GeV, true vertex in the FV outside the cathode band, no end-point
containment (`accept2_*.txt`, 7,686 signal neutrinos); the notebook's definition (2,408) is kept as the second table.

| 2prong-contained | selected | primary eff / pur | notebook eff / pur |
|---|---|---|---|
| real | 3,500 | 0.198 / 0.435 | 0.626 / 0.431 |
| M3 | 3,852 | 0.199 / 0.397 | 0.543 / 0.339 |
| M3g | 4,086 | 0.208 / 0.391 | 0.582 / 0.343 |
| M3w | 3,698 | 0.190 / 0.395 | 0.569 / 0.370 |

| final (2prong-mup-PID) | selected | primary eff / pur | notebook eff / pur |
|---|---|---|---|
| real | 936 | 0.109 / 0.895 | 0.346 / 0.890 |
| M3 | 942 | 0.106 / 0.861 | 0.305 / 0.780 |
| M3g | 971 | 0.111 / 0.877 | 0.329 / 0.816 |
| M3w | 910 | 0.104 / 0.876 | 0.316 / 0.836 |

## Shipped M4 (M3w0) and the origin of the purity gap, 2026-11-08

M3w0 = layout v4 (generated end displacement + wall flag/offset), no particle attention, generator truth only
(`accept_m3w0.txt`, `diag_m3w0_summary.txt`). Hybrid = real slice table + M3w0 PFPs (`accept_hyb.txt`).

| 2prong-contained | selected | primary eff / pur | notebook eff / pur | final primary eff / pur |
|---|---|---|---|---|
| real | 3,500 | 0.198 / 0.435 | 0.626 / 0.431 | 0.109 / 0.895 |
| M3w0 | 3,725 | 0.191 / 0.394 | 0.531 / 0.343 | 0.106 / 0.872 |
| hybrid | 3,500 | 0.194 / 0.426 | 0.559 / 0.384 | 0.112 / 0.880 |

M3w0 reproduces the wall pile-up per background category (`wall_by_cat.txt`; CC Npi on wall 0.36 vs 0.43, M3 0.09)
and the per-event exit (0.90), but purity stays at 0.394. The M4 closure with the CC1p0pi cuts per truth class gives
slightly fewer contained background than real; the hybrid recovers purity 0.426: ~3/4 of the gap comes through the
M2 slice block. `m2_purity_m3w0.txt`: containment probability high in every bin of every slice variable; the slice
block misses the wall (longest-track length = path to wall 6.0% vs real 14.5%; cohpi muon end on the wall 7.9% vs
42.9%, outside the TPC 18.7% vs 0.1%). Next: wall treatment for M2.

## M2 with wall targets (M2w), 2026-11-08

M2 retrained with the wall targets (longest-track length vs path to wall, flag + offset; cohpi muon / cpi end per-axis
wall flag + offset), regenerated slices, PFPs from the shipped M3w0 (`accept_m2w.txt`, `diag_m2w_summary.txt`,
`m2_purity_m2w.txt`; dataset `surrogate_official_m2w_pfp`, jobs 0-199).

| 2prong-contained | selected | primary eff / pur | notebook eff / pur | final primary eff / pur |
|---|---|---|---|---|
| real | 3,500 | 0.198 / 0.435 | 0.626 / 0.431 | 0.109 / 0.895 |
| M3w0 | 3,725 | 0.191 / 0.394 | 0.531 / 0.343 | 0.106 / 0.872 |
| hybrid | 3,500 | 0.194 / 0.426 | 0.559 / 0.384 | 0.112 / 0.880 |
| M2w | 3,639 | 0.185 / 0.390 | 0.515 / 0.341 | 0.106 / 0.870 |

Wall marginals now match (length = path to wall 0.163 vs real 0.146; muon end on wall 0.434 vs 0.429) but muon end
outside the TPC 4.2% (real 0.06%, unflagged ends not clipped). Purity unchanged; contained-background excess 370 -> 324.
Per event (M2 test slices): P(gen long track at wall | real at wall) = 0.56, P(gen at wall | real not) = 0.10. The
marginal is right, the event assignment is not: the hybrid carries the per-event exit from the real detector, the
generated chain must infer it from generator truth. Next: a generator-truth exit predictor (CSDA range vs path to wall).

## Why 2prong CC Npi is over-contained (topology x containment), 2026-11-08

`diag_cc1p_containment.py` now stores the slice counts; `diag_cc1p_topo.py` cross-tabulates them with the PFP-level
2prong (`topo_m2w.txt`, `topo_hyb.txt`; hybrid rebuilt with `pipeline_hyb.sh`). Overall longest-track containment
closes per category (CC Npi 0.514 vs 0.513); only the 2prong subset of CC Npi fails (0.518 vs 0.440; hybrid 0.454).
2prong is set by the slice n_pfp (P(2prong | n_pfp=3) = 0.93 real, 0.90 M2w) and M4 follows it. CC Npi, longest
track end in FV by slice n_pfp (real | M2w | hybrid): n_pfp=3 0.439 | 0.491 | 0.442; 4 0.499 | 0.508 | 0.495;
5 0.554 | 0.514 | 0.533. Real: few PFPs <-> exiting longest track; M2 generates the exit flat in n_pfp. The hybrid
(same M4, real slices) closes in every cell -> the missing correlation is between M2's topology class and M2's
longest-track length, not in M4 and not training statistics (~75k CC Npi 2prong-like training slices).

## M2t: topology sampled after the flow (+ range-curve flags), 2026-11-08

`accept_m2t.txt`, `diag_m2t_summary.txt`, `m2_purity_m2t.txt`, `topo_m2t.txt` (dataset `surrogate_official_m2t_pfp`, jobs 0-199).

| 2prong-contained | selected | primary eff / pur | notebook eff / pur | final primary eff / pur |
|---|---|---|---|---|
| real | 3,500 | 0.198 / 0.435 | 0.626 / 0.431 | 0.109 / 0.895 |
| M3w0 | 3,725 | 0.191 / 0.394 | 0.531 / 0.343 | 0.106 / 0.872 |
| M2w | 3,639 | 0.185 / 0.390 | 0.515 / 0.341 | 0.106 / 0.870 |
| M2t | 3,654 | 0.192 / 0.404 | 0.535 / 0.352 | 0.105 / 0.879 |

Purity up by 0.01-0.015, but not through the targeted correlation: CC Npi at slice n_pfp=3 longest end in FV 0.494
(real 0.439, M2w 0.491). The gain is from fewer background 2prong events (CC other 0pi 2088 vs real 2278, CC Npi 722
vs 759), i.e. a topology shift, partly an over-correction. Topology head with the block: logloss 1.36 (truth only 2.14).
Inclusive P(longest track at wall | n_pfp) is reproduced by both M2w and M2t; the CC Npi-specific link is not.
Side effects: cohpi 2-track tuple rate 0.0779 vs 0.0811, n_trk>=2 without short track 0.8% (real 0), pion range on
the curve 72% (real 93%).

## Swap test on the M2t topology head (`scripts/diag_m2_swap.py`, `swap_m2t.txt`), 2026-11-08

Truth CC + charged pion slices of the M2 test split (vertex in FV, nu_score > 0.45; 24.6k): P(n_pfp=3 | longest end in
FV / outside) real 0.108 / 0.135 (link +0.028); head on the real block +0.028, on the generated block +0.004. Replacing
one group of the real block by the generated one (event split fixed): short group +0.016 (presence alone +0.018, values
alone +0.027), two group +0.025, long dir +0.024, long length +0.026, others ~0.028. No single group restores the link
in the generated block. The link is carried jointly, the largest single piece by whether a short track exists, which
M2t samples from truth alone, independent of the flow's longest track. Next: one joint model for the counts, the
group presence and the always-present variables (stage 1), optional groups conditioned on it (stage 2).

Deleted 2026-11-08 (superseded studies): /pnfs datasets surrogate_hyb_realslc_m3w0_pfp, surrogate_official_m2w_pfp, surrogate_official_m2t_pfp and their generated parquets; the result files here are kept.

## M2s: two-stage M2 (`--two-stage --walls --range-curve`), 2026-11-08

`accept_m2s.txt`, `diag_m2s_summary.txt`, `m2_purity_m2s.txt`, `topo_m2s.txt` (dataset `surrogate_official_m2s_pfp`, jobs 0-199).

| 2prong-contained | selected | primary eff / pur | notebook eff / pur | final primary eff / pur |
|---|---|---|---|---|
| real | 3,500 | 0.198 / 0.435 | 0.626 / 0.431 | 0.109 / 0.895 |
| M3w0 | 3,725 | 0.191 / 0.394 | 0.531 / 0.343 | 0.106 / 0.872 |
| M2s | 3,198 | 0.165 / 0.396 | 0.454 / 0.342 | 0.090 / 0.876 |

The targeted link closes: CC Npi at slice n_pfp=3 longest end in FV 0.442 (real 0.439, M2w 0.491, M2t 0.494); the
other n_pfp cells and categories also agree. Consistency: n_trk>=2 without short track 0; rule fixes after stage 1
0.03-0.9% (n_proton 0.9%, short-for-n_trk=2 0.7%, two-outside-2trk 0.8%).
But the chain loses 2prong events (primary eff at 2prong 0.573 vs real 0.671): P(PFP 2prong | slice n_pfp=3) 0.85
vs 0.97 for signal, 0.73 vs 0.93 for CC Npi. Cause: the dequantised counts in the stage-1 flow do not keep the joint
count lattice. 17% of generated slices carry a (n_pfp, n_prong, n_trk, n_proton, n_shower) tuple that real slices do
not populate (P < 1e-4), n_pfp = n_trk + n_shower + 1 holds for 0.504 (real 0.590), and (3,2,2,*,0) is 0.341 of
n_pfp=3 (real 0.439). M4 then builds fewer 2-track sets. The muon end beyond the wall is 4.3% (real 0.06%); M2w had
the same. Fix to try: keep the counts categorical (topology class) inside stage 1.

## M2o: two-stage M2 with the counts as a one-hot topology class (`--two-stage --topo-onehot --walls --range-curve`), 2026-11-08

`accept_m2o.txt`, `diag_m2o_summary.txt`, `m2_purity_m2o.txt`, `topo_m2o.txt` (dataset `surrogate_official_m2o_pfp`, jobs 0-199).
First run killed by the 24 GB limit at stage 2 (`reports/pipeline_m2o_oom.log`); rerun after 78a8c43.

| 2prong-contained | selected | primary eff / pur | notebook eff / pur | final primary eff / pur |
|---|---|---|---|---|
| real | 3,500 | 0.198 / 0.435 | 0.626 / 0.431 | 0.109 / 0.895 |
| M3w0 | 3,725 | 0.191 / 0.394 | 0.531 / 0.343 | 0.106 / 0.872 |
| M2s | 3,198 | 0.165 / 0.396 | 0.454 / 0.342 | 0.090 / 0.876 |
| M2o | 3,511 | 0.170 / 0.371 | 0.462 / 0.317 | 0.090 / 0.847 |

The lattice is fixed: every generated count tuple is a real one, (3,2,2,*,0) is 0.435 of n_pfp=3 (real 0.439, M2s
0.341), P(PFP 2prong | slice n_pfp=3) 0.966 for signal (real 0.971, M2s 0.85), total-variation distance of the
tuple distribution 0.092 (M2s 0.170). But the argmax over 121 classes overweights the high-multiplicity tuples
(n_pfp >= 6 0.115 vs 0.084, n_shower >= 1 0.303 vs 0.257), the slice block is worse (long_trk_len W1 0.023 vs 0.005),
and the targeted link is lost again: CC Npi at n_pfp=3 longest end in FV 0.515 (real 0.439, M2s 0.442). Rule fixes:
short-for-n_trk=2 3.3%, two-outside-2trk 1.1%; cohpi SR 89 vs real 156 (M2s 104).
