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
