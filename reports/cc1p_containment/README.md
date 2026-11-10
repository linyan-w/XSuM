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

Schematic of the shipped slice generator and the three variants: `docs/figures/slice_generator_proposal.svg`.

## M2c: two-stage M2, topology class from a classifier on truth + generated stage 1 (`--two-stage --topo-cls --walls --range-curve`), 2026-11-09

`accept_m2c.txt`, `diag_m2c_summary.txt`, `m2_purity_m2c.txt`, `topo_m2c.txt` (dataset `surrogate_official_m2c_pfp`, jobs 0-199).

| 2prong-contained | selected | primary eff / pur | notebook eff / pur | final primary eff / pur |
|---|---|---|---|---|
| real | 3,500 | 0.198 / 0.435 | 0.626 / 0.431 | 0.109 / 0.895 |
| M3w0 | 3,725 | 0.191 / 0.394 | 0.531 / 0.343 | 0.106 / 0.872 |
| M2s | 3,198 | 0.165 / 0.396 | 0.454 / 0.342 | 0.090 / 0.876 |
| M2o | 3,511 | 0.170 / 0.371 | 0.462 / 0.317 | 0.090 / 0.847 |
| M2c | 3,702 | 0.191 / 0.396 | 0.525 / 0.343 | 0.107 / 0.877 |

Counts fixed: tuple TV distance to real 0.017 (shipped 0.015, M2s 0.170, M2o 0.092), (3,2,2,*,0) 0.436 of n_pfp=3
(real 0.439), P(PFP 2prong | n_pfp=3) 0.970 for signal (real 0.971), n_pfp >= 6 0.083 (0.084); presence rates within
0.006; no slice without a compatible class, rule fixes <= 3e-4; cohpi 2trk 20203 vs 20116, SR 138 vs 156.
Link not closed: CC Npi at n_pfp=3 longest end in FV 0.488 (real 0.439; shipped 0.49). Acceptance equals the shipped
chain (0.191 / 0.396 vs 0.191 / 0.394).

Re-reading M2s: at the M2 level (test slices, truth CC + pion, vertex in FV, nu_score > 0.45, n_pfp=3) the longest
track ends in the FV for 0.486 (M2w 0.483, M2o 0.521, M2c 0.480; real 0.446). M2s's chain value 0.442 was therefore
not a closed link but the distorted n_pfp=3 population (P(2prong | n_pfp=3) 0.73 for CC Npi). No variant has
reproduced the real correlation between the longest-track exit and the PFP count.

## Where the exit <-> n_pfp link comes from (`scripts/diag_m2_link.py`, `scripts/diag_m2_link_head.py`; `link_m2c.txt`, `link_m3.txt`, `link_head_m3.txt`), 2026-11-09

M2 test slices, truth CC + charged pion, vertex in FV, nu_score > 0.45 (24.5k); exit = longest track ends outside the FV.
- The link is a truth-level effect, not a reconstruction-internal one: a GBDT P(n_pfp=3 | 80 truth features) gains
  nothing from adding the reco exit (logloss 0.2985 -> 0.2984, exit effect given truth +0.005; raw link +0.026).
  Split by the Geant4 containment of the leading muon (reference only) the link vanishes within each half
  (-0.006 / -0.003). Within bins of any single truth variable it survives (+0.02), so it is a joint dependence.
- The generated exit decision is right: agreement with the Geant4 muon 0.800 (real 0.809), P(exit | p_mu) matches
  in every bin, per-event agreement with the real exit 0.83.
- What the generators miss is P(n_pfp=3 | truth) in one kinematic region. With the exit predicted from the truth
  alone (GBDT, AUC 0.88) the real link is +0.035-0.040, the generated one +0.018 (shipped), -0.003 (M2c). In
  (p_mu, cos theta_mu) terciles: real / generated / shipped head probability at mid p_mu & forward 0.172 / 0.129 /
  0.144, high p_mu & forward 0.138 / 0.121 / 0.115, high p_mu & not forward 0.067 / 0.096 / 0.080. By truth-predicted
  exit probability: at 0.62-0.90 real 0.164, generated 0.127, head 0.127.
- So the topology classifier (softmax MLP 3 x 256, 15 epochs, truth + z) underfits a joint dependence of the PFP
  count on the muon kinematics; its binary n_pfp=3 logloss (0.288) is better than an 80-feature GBDT (0.2985), so the
  z embedding carries information, but the region above is 20-30% low. Every variant inherited this (M2s, M2o drew
  the counts from a flow on the same inputs; M2c's classifier read the generated block, which adds nothing beyond
  the truth). Candidate fix without retraining the flows: a stronger topology head on the same inputs (the shipped
  flow is conditioned on the class index, which is fixed by the training split), checked on the cells above.

## The link is the vertex-in-FV cut; the generated reco vertex is smeared 10x too much (`link_m3.txt` sections 6-8, `link_angles_m3.txt`, `m2_official_m3h.log`), 2026-11-09

Two negative results first. (a) A topology head with 4x the capacity and 40 epochs (`scripts/retrain_topo_head.py`,
evaluated on the shipped test files) does not recover the region: P(n_pfp=3 | truth-predicted exit 0.61-0.90) real
0.184, shipped head 0.143, new head 0.149; multi-class logloss 2.1436 vs 2.1415. (b) Adding the angles between the
muon and the other charged primaries from the particle set to an 80-feature GBDT (`scripts/diag_m2_link_angles.py`)
does not either (0.164 real, 0.145 -> 0.144). Collinear mu-pi pairs do have more n_pfp=3 (0.199 at cos > 0.97) but
are 5% of the events.

Then the selection: without the vertex-in-FV cut the real link is gone (+0.005; generated +0.014). It appears with
the cut on the RECO vertex (+0.026 real, +0.010 generated) and does not change with nu_score. Among n_pfp=3 CC + pion
slices whose TRUE vertex is in the FV, the reco vertex passes the FV for 0.975 of exiting and 0.958 of contained
slices in real, 0.932 and 0.968 in the generated ones (sign reversed). The reco-true vertex displacement is the
cause: real median 0.4-0.6 cm, P(> 5 cm) 0.08 (exit) / 0.13 (contained) for all CC + pion slices; generated median
4-6 cm, P(> 5 cm) 0.30 / 0.36. The tails (> 20, > 50 cm) match. Exiting-muon slices have true vertices nearer the
downstream and side walls, so a 4 cm smear pushes them out of |x| < 190, z < 450 (0.980 / 0.978 vs real 0.997 / 0.993)
while real keeps them in; contained slices with a badly misplaced vertex (real P(> 20 cm) 0.205) fall out in both.
So the exit <-> n_pfp link of the acceptance study is a property of the reco vertex resolution, not of the PFP counts,
and the generator's vertex core is 10x too wide. The vertex columns are generated as absolute positions with a
robust linear scaling, so a 0.5 cm core next to a 10-cm-scale distribution of misplacements is invisible to the flow.
Fix: generate the vertex as the residual to the true vertex with an asinh scale of ~1 cm (and keep the wall spikes).

## M2v: the vertex generated as the residual to the true vertex (`--vtx-resid`; `accept_m2v.txt`, `diag_m2v_summary.txt`, `m2_purity_m2v.txt`, `topo_m2v.txt`, `link_m2v.txt`), 2026-11-09

Shipped configuration (60 epochs, 6 x 1024, 3M slices, run `m2_official_m2v`) with the three vertex columns entering
the flow as asinh((reco - true) / 1 cm); at generation the true vertex is added back and the result clipped to the
real training range (`pipeline_m2v.sh`, PFPs from the shipped M3w0, dataset `surrogate_official_m2v_pfp`, jobs 0-199).

The vertex core is fixed. `link_m2v.txt` section 8, n_pfp=3 CC + pion slices with the true vertex in the FV: reco-true
displacement median 0.4 (exit) / 0.6 cm (contained), P(> 5 cm) 0.175 / 0.307 (real 0.159 / 0.338; shipped 4-6 cm,
0.43 / 0.56); the tails are unchanged. The reco FV pass is 0.950 (exit) / 0.967 (contained) against real 0.975 /
0.958 (shipped 0.932 / 0.968): the reversal is halved, not gone. `scripts/diag_m2_vtx_dir.py` (`vtx_dir_m2v.txt`):
the displacement components along the longest track and towards the nearest wall match real, yet exiting slices fail
the FV more than real at every true wall distance, including > 50 cm.

| 2prong-contained | selected | primary eff / pur | notebook eff / pur | final primary eff / pur |
|---|---|---|---|---|
| real | 3,500 | 0.198 / 0.435 | 0.626 / 0.431 | 0.109 / 0.895 |
| M3w0 (shipped) | 3,725 | 0.191 / 0.394 | 0.531 / 0.343 | 0.106 / 0.872 |
| M2v | 3,741 | 0.193 / 0.397 | 0.538 / 0.346 | 0.105 / 0.859 |

The acceptance is the shipped chain. CC Npi at n_pfp=3, longest end in FV: 0.502 (real 0.439, shipped ~0.49). The M2
link: real +0.025, M2v +0.012 (shipped +0.010); with the truth-predicted exit real +0.034, M2v +0.001.

Where the rest of the link is: the zone of the TRUE vertex (`vtx_dir_m2v.txt`, same table for `m3`, `m2w`, `m2c`).
P(reco vertex in FV | exit / contained) for n_pfp=3 CC + pion slices with a longest track and nu_score > 0.45:

| true vertex | share of n_pfp=3 | real exit / cont | shipped | M2w | M2c | M2v |
|---|---|---|---|---|---|---|
| in FV | 0.30 | 0.982 / 0.965 | 0.941 / 0.967 | 0.947 / 0.968 | 0.972 / 0.983 | 0.957 / 0.967 |
| in AV, outside FV | 0.09 | 0.061 / 0.176 | 0.096 / 0.234 | 0.094 / 0.186 | 0.077 / 0.203 | 0.061 / 0.164 |
| outside AV (dirt) | 0.61 | 0.316 / 0.123 | 0.275 / 0.196 | 0.281 / 0.178 | 0.274 / 0.175 | 0.250 / 0.190 |
| all | | 0.465 / 0.401 | 0.435 / 0.447 | 0.433 / 0.448 | 0.438 / 0.435 | 0.426 / 0.434 |

Contribution of each zone to P(FV | exit) - P(FV | cont): real in-FV -0.046, dirt +0.108; the generators give the dirt
zone +0.034 (shipped), +0.059 (M2w), +0.051 (M2c), +0.021 (M2v). Three fifths of the CC + pion n_pfp=3 slices at this
stage are from neutrinos outside the active volume, and that zone carries the link: in real a dirt slice whose
longest track exits has its reco vertex inside the FV 0.32 of the time, 0.12 when the track is contained.

`scripts/diag_m2_vtx_dirt.py` (`vtx_dirt_m2v.txt`, `_m3`, `_m2w`; true muon direction from the primary set, diagnostic
only) looks at the dirt slices whose true muon line enters the AV (76%): the reco vertex sits on the muon line at the
entry wall for 0.31 of them (P(FV) 0.006), off the line by > 5 cm for 0.67 (P(FV) 0.23), and these fractions and the
position along the chord are reproduced by every generator (M2v 0.24 / 0.75). What differs is the longest track given
the vertex: real P(exit | reco vertex in FV) 0.750, M2v 0.552 (shipped 0.604, M2w 0.686); real exiting tracks have
length = path to the AV wall within 5% for 0.51 of them, M2v 0.23 (shipped 0.26, M2w 0.51); among in-FV dirt slices
0.38 real, M2v 0.09, M2w 0.36. So the dirt link is the wall relation of the longest track (a piece of a through-going
muon from a mid-track vertex ends on the wall), which the wall targets of M2w restore at the marginal level; M2w's
acceptance (0.185 / 0.390) did not improve because its vertex core was still 4-6 cm and its per-event wall assignment
was 0.56 (above). The two fixes act on different zones (vertex residual: true vertex in FV; wall targets: dirt) and
have not been combined. Next, if approved: M2v + `--walls` (the shipped configuration with both), M2vw.

## M2vw: vertex residual + wall targets (`--vtx-resid --walls`; `accept_m2vw.txt`, `diag_m2vw_summary.txt`, `m2_purity_m2vw.txt`, `topo_m2vw.txt`, `link_m2vw.txt`, `vtx_dir_m2vw.txt`, `vtx_dirt_m2vw.txt`, `contain_joint_m2vw.txt`), 2026-11-09

The shipped configuration with both fixes (run `m2_official_m2vw`, `pipeline_m2vw.sh`, PFPs from M3w0, dataset
`surrogate_official_m2vw_pfp`, jobs 0-199).

| 2prong-contained | selected | primary eff / pur | notebook eff / pur | final primary eff / pur |
|---|---|---|---|---|
| real | 3,500 | 0.198 / 0.435 | 0.626 / 0.431 | 0.109 / 0.895 |
| M3w0 (shipped) | 3,725 | 0.191 / 0.394 | 0.531 / 0.343 | 0.106 / 0.872 |
| M2v | 3,741 | 0.193 / 0.397 | 0.538 / 0.346 | 0.105 / 0.859 |
| M2vw | 3,617 | 0.188 / 0.400 | 0.524 / 0.349 | 0.106 / 0.861 |

A small gain, within the statistical precision of the test sample (purity +-0.008). Both zones of the link moved
towards real: true vertex in FV, reco FV pass by exit 0.960 / 0.969 (real 0.975 / 0.958; M2v 0.950 / 0.967); dirt
zone P(reco vertex in FV | exit / contained) 0.248 / 0.166 (real 0.296 / 0.112; M2v 0.250 / 0.190; M2w 0.281 / 0.178),
P(exit | reco vertex in FV) 0.627 (real 0.750, M2v 0.552, M2w 0.686), exiting dirt tracks with length = path to the
wall within 5%: 0.466 (real 0.510). The M2-level link is +0.011 (real +0.025; shipped +0.010, M2v +0.012); the chain
CC Npi n_pfp=3 longest-end-in-FV fraction 0.476 (real 0.439; shipped 0.49, M2v 0.502, M2w 0.491). Wall marginals as
M2w (length at the path 0.163 vs 0.142 real).

Decomposition of the primary purity gap (0.435 -> 0.400), from the acceptance counts: primary signal selected 1,522
real vs 1,445 (-77), background 1,978 vs 2,172 (+194). The hybrid (real slices + M3w0 PFPs) has background +31 and
signal -31 (0.426), so the background excess is the slice generator's and the signal loss the PFP generator's.
Note on the diagnostic tables (`diag_*_summary.txt` section A): their `signal` is the truth-contained signal, so the
row `numu CC 1p0pi (any cont.)` contains the primary-signal neutrinos whose true muon or proton leaves the FV. Its
excess (+188 contained at 2prong) is almost all primary signal reconstructed as contained (190 vs 14 real, from the
acceptance file), i.e. the PFP generator's end-point leak-in, not background; the truth-contained signal lost is
-253. The background excess by category is CC Npi +72 (2prong population +8%, contained 0.339 vs 0.273), NC +42,
CC other 0pi -9, nue -5, plus neutrinos outside the diagnostic's matched table.

Side observation for the PFP generator (`contain_joint_m2vw.txt`): for the CC 1p0pi neutrinos whose true muon or
proton leaves the FV, the three containment conditions pass at the real rates one by one (longest end in FV 0.224 vs
0.236; second end 0.888 vs 0.889; same cathode side 0.665 vs 0.674) but their AND is four times too frequent (0.011
vs 0.042; hybrid 0.037): given both ends in the FV, real has the vertex and ends on one side of the cathode only
0.064 of the time, generated 0.220. A long muon whose reco end is inside the FV has in real crossed the cathode;
the generated end keeps the cathode side. The same correlation loss gives the signal containment 0.854 vs 0.986
(longest end out of FV 0.092, wrong cathode side 0.036). This is the notebook-purity gap (0.431 vs 0.35) and the
largest lever left in M4.

## M2vw adopted as the shipped slice generator (run `m2_official_m2vw`), 2026-11-09

Decision: M2vw replaces `m2_official_m3`. The vertex resolution is right (the previous run was 10x too wide), the
wall marginals of the slice block match real (length at the path 0.161 vs 0.145; muon end on the wall 0.433 vs 0.429;
muon end outside the TPC 0.000 vs 0.001), and every acceptance stage is at least as close to real as before; cost:
per-column relative W1 0.0073 -> 0.0091 (< 1% of the range). Before shipping, `apply_walls` was completed
(`xsum/train/m2.py`, `tests/test_walls.py`): a flagged end's offset is clipped to the wall band (3-7% of flagged
ends fell outside), and an unflagged end beyond the band (2% per axis, up to tens of cm; real has none) is mirrored
back inside about the wall plane. The slices and PFPs were regenerated with the fix (the CC1p0pi cuts use the PFP
track ends, not the cohpi end columns); the result files of the first pass are kept with the suffix `_noclip`.

Held-out split (see the next section; `accept_m2vw.txt`, `accept_m3w0.txt`, `diag_*_summary.txt`, `topo_*.txt`):

| 2prong-contained, held-out | selected | primary eff / pur | notebook eff / pur | final primary eff / pur |
|---|---|---|---|---|
| real | 3,555 | 0.197 / 0.425 | 0.623 / 0.418 | 0.109 / 0.885 |
| M3w0 (previous slice generator) | 3,720 | 0.193 / 0.398 | 0.537 / 0.344 | 0.108 / 0.875 |
| M2vw (shipped) | 3,654 | 0.192 / 0.403 | 0.538 / 0.351 | 0.113 / 0.869 |

The gap is -0.022 (previous -0.027); background excess +137 (previous +196), signal -38 (-31). M2vw is nearer real
at every stage: selected / real 1.006 vs 1.012 at the vertex cut, 1.004 vs 1.012 at nu_score, 1.028 vs 1.046 at
2prong-contained; overlap of the selected events at the last stage 0.404 vs 0.382. The held-out closure equals the
in-sample one within statistics (in-sample gap -0.035 / -0.041): the generators do not overfit their training files.

Datasets on /pnfs after the clean-up: `surrogate_official_m2vw_pfp` (jobs 0-199, the shipped chain; pass 2 on
approval), `surrogate_official_m3_pfp` (2,000 jobs, previous slice and PFP generators; kept until pass 2 exists).
Deleted: `surrogate_official_v3`, `surrogate_official_v3_pfp` (superseded, 150 GB) and `surrogate_official_m3w0_pfp`
(the previous shipped chain, jobs 0-199). The shared loader `dataframe/xsum_testsplit.py` now reads the split from
`dataframe/file_splits.parquet` (one copy, identical in every dataset) and defaults to the shipped dataset.

## The datasets' test split was not the generators' held-out split, 2026-11-09

Found while reinstalling the split file: `split_by_level` is a seeded shuffle of the *sorted file list*, so the
assignment depends on the whole list. The M3 encoder, the M2 flow and M4 are trained on the matched table
(745,363 files, test 74,537) and share one split, but the M2 cardinality head uses the 4M-row card table (740,304
files) and `write_surrogate_df.py` derived each dataset's `file_splits.parquet` from the generated index (745,350
files for the m3-era datasets, a different list again for v3). Different lists give unrelated assignments: the test
sets of the models and of the datasets overlap at 10-20%, and 71% of the datasets' "test" files (52,949 of 74,535)
are files the generators trained on. Every acceptance and diagnostic number above (shipped, hybrid, M2w, M2t, M2s,
M2o, M2c, M2v, M2vw) was therefore computed on a sample that is mostly in-sample for M2 and M4 (and the v3-era
`file_splits.parquet` of `surrogate_official_v3_pfp`, which the loader read until today, was yet another random
10%; the first regenerated-M2vw run, which read the m3-era copy, gave 0.190 / 0.404 against real 0.195 / 0.434 for
that reason). The comparisons between variants stand (same sample for all), the absolute closure does not.

Fix (no retraining): `xsum.data.splits.file_split_table` derives the split table from the generators' training
table (`table_matched_m3`; the 240 files with neutrinos but no matched slice are labelled test), written once by
`scripts/make_file_splits.py` to `data/prod_v10_14_02_03_official/file_splits.parquet` and installed as
`/pnfs/.../dataframe/file_splits.parquet` (read by the loader) and in every dataset directory;
`write_surrogate_df.py` now copies it (`--split-table`, required). The held-out sample of jobs 0-199 has 7,516 files
and 179,759 neutrinos, the same size as before. The M2 cardinality head remains trained on 90% of these files (it
decides only whether a neutrino gets a slice); the next training round should split by a hash of the file id so that
every table shares the assignment. Results on the held-out split: `accept_m2vw.txt`, `accept_m3w0.txt` (the
in-sample files are kept as `*_noclip` / `*_v3split`).
