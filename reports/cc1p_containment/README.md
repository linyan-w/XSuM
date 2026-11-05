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
