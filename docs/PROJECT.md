# XSuM: SBND sim→reco surrogate, project design

Status: repository scaffold and data audit (2026-10-16). Sections 9 and 10 list the decisions still open.
Sister project: `opendata-sim2reco` (MINERvA open data); the design below follows it where the two detectors
allow, and departs where SBND's LArTPC reconstruction and the cafpyana analysis chain differ.

## 1. Goal

Learn, from SBND MC alone, a **per-event stochastic map from GENIE final-state truth to the reconstructed
quantities an SBND analysis actually uses**, and make it extrapolate sensibly beyond the phase space the
released MC covers (alternative generators, modified FSI, new kinematic regions), so that a forward-folded
reco-level prediction can be made for a model that was never simulated.

Immediate application: the CC coherent pion (`cohpi`) analysis built on cafpyana. The analysis already
produces a per-neutrino truth table (`mcnu`) and a per-selected-slice reco table (`evt`) with a truth-match
index between them. The surrogate replaces the Geant4 + detector simulation + Pandora + cafpyana chain that
connects the two tables: **`mcnu` → `evt`**.

What the surrogate is not: it does not fix data/MC disagreement. It is trained on MC and reproduces the MC
detector response, including its flaws. Its value is transporting that response to new truth inputs, and
doing so with per-prediction uncertainty.

## 2. Problem statement

For one neutrino interaction, let

- `X` = the set of primary particles leaving the nucleus (`rec.mc.nu.prim`: pdg, 4-momentum, start point),
  variable size,
- `c` = event-level context: true vertex position (which TPC, distance to cathode/anode/walls), and for
  SBND the fact that the event sits in a BNB spill with overlaid cosmics,
- `R` = the reconstructed record: whether a Pandora slice truth-matched to this neutrino exists and passes the
  analysis selection, and if so a fixed block of slice- and track-level variables (vertex, two track
  directions and lengths, range momenta, chi2 PID, containment flags, light yield, charge in spheres around
  the vertex).

We want `p(R | X, c)`, to sample from it, and to characterise where samples can be trusted.

Properties that drive the design (measured on the example dataset, `docs/dataset_notes.md`):

1. **Stochastic and multimodal.** Range momentum / true momentum for the selected signal muon has a median of
   0.89 but a 16–84 % band of [0.44, 1.02]; two thirds of selected signal muons exit, so range momentum is a
   lower bound for most of them. For the pion the median response is 0.57 (pions interact before stopping).
   A conditional generative model is needed, not a regression.
2. **Reco quality is an event property.** The slice is built from all hits in the drift window, with cosmics
   overlaid; which track is "long" and which is "short" is assigned by length, not by species, so the
   muon/pion swap is a discrete mode that depends on both particles at once.
3. **Everything visible must be explainable by what entered the detector.** Generator labels
   (`genie_mode`, `iscc`, `Q2`, `w`, the `nuint_categ` category) are not inputs: the detector only sees `X`.

## 3. Inputs

**Particle set.** `rec.mc.nu.prim` is the GENIE primary list after FSI, exactly what Geant4 propagates.
Keep mu±, e±, p, n, pi±, pi0, gamma, K, hyperons; drop outgoing neutrinos and nuclear remnants. Per-particle
features: species embedding plus `(px, py, pz)` in detector coordinates (GeV). Neutrons and pi0 stay as tokens.

**Context.** True vertex `(x, y, z)`; from it the model can learn TPC membership and containment. BNB spill
intensity and cosmic overlay conditions are marginalised over in v1.

**Example-dataset limitation.** The cohpi `mcnu` table carries only six kinematic truth variables
(`true_E_nu, true_p_mu, true_p_pi, true_cos_theta_mu, true_cos_theta_pi, true_t`), no vertex, no particle
list, and only **signal** neutrinos (3,962 of them). For the signal topology (1 mu, 1 pi, no protons above
20 MeV, no neutrons, no pi0) the six variables plus azimuth are almost the full particle set, so a first
surrogate can be trained on them; but the true vertex is missing and the sample is small. Section 8
describes the dataframe the full surrogate needs.

## 4. Outputs

**Tier 0: selection.** `selected ∈ {0, 1}`: a slice truth-matched to this neutrino exists in `evt` after the
analysis selection (FV, not clear cosmic, total PE > 1500, barycenter flash-match score > 0.05, exactly two
track candidates, no proton or shower candidates, direction and opening-angle cuts, chi2 cuts). In the
example dataset 1,463 / 3,962 signal neutrinos are selected (37 %). Every selected slice matches exactly one
neutrino and every neutrino has at most one selected slice, so Tier 0 is a Bernoulli head.

**Tier 1: fixed-size slice block** (about 45 numbers, listed in `configs/data.yaml`): reconstructed vertex,
long/short track direction, length, range momentum, start and end points, chi2 PID (muon, proton; old and
re-calibrated), containment flags, total PE, flash-match and neutrino scores, charge in 1–4 cm spheres around
the vertex. `reco_t`, `opening_angle` and `sphere_3_over_4m3` are deterministic functions of other columns
and are recomputed, not generated.

**Tier 2 (later): the variable-size PFP set** of the slice (every Pandora particle flow object with track
score, length, direction, PID, calorimetry), from which any analysis selection can be re-derived. This is
the SBND analogue of the MINERvA prong set and needs a richer `evt` table (section 8).

**Response parameterisation.** Internally model `log(range_p / p_true)`, `cos θ_reco − cos θ_true`,
`vtx_reco − vtx_true` and similar, with the long/short assignment handled as a discrete variable; these are
smoother in the inputs than raw values and extrapolate better.

## 5. Model

Same architecture family as sim2reco: a permutation-invariant set encoder over the particle tokens with a
global context token, a Bernoulli head for Tier 0, a conditional flow-matching generative model over the
Tier 1 vector, and later a conditional set flow for Tier 2. Bayesian last layers for epistemic uncertainty.
With the example dataset (one neutrino = one mu token + one pi token) the encoder reduces to an MLP on six
numbers, which is the M1 baseline.

## 6. Extrapolation and validation

Hold out physics-defined regions, never random rows:

- Train on `true_t < 0.1 GeV²` and test on the tail (coherent events are concentrated at low t).
- Train on `true_E_nu < 1.5 GeV`, test above.
- Train on muon-forward events (`cos θ_mu > 0.95`), test on the rest.
- With the full dataset (section 8): train on signal + CCQE + RES, test on DIS and 2p2h; train on one TPC,
  test on the other; train on contained-muon events, test on exiting.

Metrics: per-variable 1D distributions (KS, Wasserstein), conditional response vs true momentum, selection
efficiency vs `(p_mu, p_pi, t)`, the `reco_t` smearing matrix against the one from the true chain (this is
the object the cross-section unfolding uses), and a classifier two-sample test.

## 7. Milestones

- **M0** data pipeline: slim the analysis df to Parquet, build the per-neutrino training table
  (truth + `selected` + Tier 1), file-level splits, dataset summary. *Done for the example df.*
- **M1** baselines: logistic efficiency model and a mixture density network for Tier 1 on the six truth
  variables; closure plots; the `reco_t` smearing matrix from surrogate vs truth chain.
- **M2** flow-matching surrogate on the example df; holdouts in t and E.
- **M3** full dataset (section 8): particle-set inputs, all interaction types, true vertex; set encoder;
  efficiency including backgrounds.
- **M4** Tier 2 PFP set; Bayesian last layers; out-of-support flag.

## 8. Dataframe the full surrogate needs

The example df was made by `analysis_village/cohpi/configs/cohpi_all_weight_signal.py` in cafpyana, which
keeps only signal neutrinos and applies the final selection before writing `evt`. The XSuM configuration
(`configs/cafpyana/analysis_village/xsum/`, run on sbndbuild03, see `docs/dataset_notes.md`) produces, from
the same flat CAFs:

1. `mcnu`: **all** neutrinos (every `rec.mc.nu`), with `position.{x,y,z}`, `E`, `momentum`, `pdg`, `iscc`,
   `genie_mode`, `w`, `Q2`, the particle counts, and the cohpi kinematics; weights optional.
2. `mcprim`: the `rec.mc.nu.prim` table (pdg, genE, genp, start, end, length, contained).
3. `slc`: every slice (not clear cosmic), with the selection stages as boolean columns instead of filtering,
   plus `tmatch_idx`, `tmatch_eff`, `tmatch_purity`.
4. `pfp` (for Tier 2): every PFP of those slices with track score, length, direction, range momenta, chi2,
   shower energy and the truth-matched particle.

Input sample list: `data/sample_lists/sbnd/prod_2026/v10_14_02_03/` in cafpyana (2,996 merged flat CAFs on
dCache, `root://fndcadoor.fnal.gov`), the v10_14_02_03 BNB + cosmics CV production. The example df covers
565,463 subrun-files, 7.5 M triggers, 4.1e20 POT.

## 9. Open questions

- Should the surrogate be trained per analysis selection (fast, directly useful for cohpi) or on the
  selection-independent PFP set (general, harder)? Plan: both, Tier 1 first.
- Cosmic overlay: treat as marginalised noise (v1) or condition on the number of cosmic slices in the event?
- Does the analysis need per-universe systematic weights carried through? They are truth-level weights and
  pass through the surrogate unchanged; the surrogate only replaces the detector response.

## 10. Decisions taken

- 2026-10-16: package name `xsum`; truth → reco direction is `mcnu` → `evt`; splits by `file_idx`;
  generator-internal labels are never inputs; the example df is used only for M0–M2.
- 2026-10-16: scope fixed. XSuM generates, per true neutrino, the complete cafpyana reconstruction record
  (the `slc` row incl. hit-level quantities such as `total_pe` and the sphere charges, and the `pfp` set) in
  the same dataframe format as the production. Analysis selections are not modelled; they are recomputed from
  the generated variables by the analysers (cohpi: S. Oh, kaon: T. Wester), who compare the surrogated sample
  with the real chain. The technotes (`docs/analysis_notes.md`) serve only to know which variables matter and
  to sanity-check the production. Slice cardinality per neutrino (0, 1, several matched slices) is a target;
  cosmic-only slices are independent of the neutrino model and are reused from the MC.
