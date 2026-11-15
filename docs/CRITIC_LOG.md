# Critic log

## Round 1 - design review

Reviewer stance: adversarial statistician / model-risk auditor. Scope: `docs/DESIGN.md` v1 checked against `src/`, `config/assumptions.yaml`, `tests/`, and the shipped CSVs (seed 42). Nothing under `src/`, `app/`, `config/`, `data/` or `docs/DESIGN.md` was edited.

Evidence comes from `scripts/critic/probe_*.py`, run from the repo root with `python -m scripts.critic.<name>`. They use `tests/critic/simworld.py`, which rebuilds the seed-42 world (the same yard effects) and draws **fresh** lots, with the true marginal p0 = E[p|x] computed by Gauss-Hermite quadrature over the N(0, 1.3²) lot effect. Acceptance tests are in `tests/critic/` and run in about 25 s (`python -m pytest -q tests/critic`). Today 19 fail and 5 pass.

### Verdict
The design is mostly sound on paper. The code implements almost none of its validation section (§5.1 splits and baselines, §5.2, §5.3, §5.4, the R2 freeze). Four statistical errors would mislead the judges: a circular truth (D14), inflated P(≥15%) (D1), under-covering intervals (D2), and an ill-defined regret (D16).

### Faults

**BLOCKER**

- **D14: synthetic truth equals the planner's assumption.** `src/data/generate.py:112` reads `cfg["interventions"]["metaphylaxis_morbidity_rrr"]`, the same value the optimizer assumes. Every decision validation on default settings is therefore circular. The model's fitted metaphylaxis coefficient (−0.612, implied RRR 0.458) "confirms" 0.45 only because 0.45 was generated. ENH/receiving effects have no counterpart in the DGP at all, so in the regret study "truth" for those effects = assumption. *Fix:* put the true effects in `dgp:` (`true_metaphylaxis_rrr`, `true_enhanced_monitoring_cfr_rrr`, ...). Score plans under true effects that differ from the planner's, and state that ENH benefit can never be validated on this data.
- **D15: §5.4 regret study (the stated "primary result") does not exist.** There is no `policy_eval` module and no oracle. §5.3 stress tests, §5.2 oracle recovery and fresh-seed metrics are also missing. *Fix:* implement the APIs below.
- **D16: regret is ill-defined.** `allowed_matrix` (`optimize.py:146`) builds the care floor from *estimated* tiers. Planner and oracle therefore optimise over different feasible sets. Running the naive §5.4 recipe gives regret against a Bayes oracle of **−1.45k to −5.1k USD in 6/6 weeks**. The planner "beats" the oracle because the model's low bias (D8) lets it escape the binding floor. *Fix:* compute regret on one feasible set (floor from TRUE risk), so that regret ≥ 0. Report floor violations against true p0 separately, and report the cost of the floor itself.
- **D1: P(≥15%) is miscalibrated (normal approximation misused).** `risk.py:146-150` centres a logit-normal at logit(p). Here p is the median of draws from a population-averaged fit, which approximates a *mean*. For p < 0.5 the mean is above the median, so P(high) is inflated. On 1,000 fresh untreated lots, mean p_high is **0.447 vs a true frequency of 0.359**. By bin: 0.19 predicted vs 0.12 observed, and 0.80 vs 0.66. Binomial noise and epistemic spread are also ignored. *Fix:* integrate over the epistemic draws and over the lot effect, with the centre solved so that the logit-normal mean = p (as `followup._prior_particles` already does), and add binomial noise for "observed ≥15%". Then calibrate on fresh seeds.

**MAJOR**

- **D2: the epistemic 90% range under-covers.** Coverage of the true E[p|x] is **0.77–0.80** (nominal 0.90). The causes: bootstrap refits reuse the full-data medians and scaler (`risk.py:199-201`), the model is misspecified (no hinges), and 30 models × 60 draws gives noisy 5–95% percentiles. *Fix:* refit the preprocessing inside each bootstrap, use at least 200 draws, report coverage on fresh seeds, and widen the range or disclose the shortfall.
- **D3: the primary splits are missing.** `evaluate()` (`risk.py:283`) is a random stratified split only. Temporal and LOYO splits do not exist, yet the brief quotes AUC 0.82 from the random split. Probe results: temporal AUC 0.826 / Brier 0.141; **LOYO AUC A 0.714, B 0.733, C 0.826, D 0.781**; random 0.844. LOYO is the honest number. Yard B is 99.7% light lots, so B's held-out performance rests on extrapolation.
- **D4: temporal validation cannot fail.** The DGP has no time term (morbidity 0.107 / 0.117 / 0.106 for 2024 / 2025 / 2026). "Temporal" is a random split under another name. *Fix:* add a drift scenario (intercept or source-mix shift) and show the temporal split detecting it.
- **D8: untreated risk is biased low exactly where decisions are made.** Predicted p0 minus true marginal p0 by weight band: **<500 −0.7 to −0.85 pp; 500–600 −0.9 to −1.0; 600–700 −1.35 to −1.44; 700–800 −0.9; ≥800 −0.3**. The cause is misspecification (linear weight vs the 700-lb hinge, crowding and shrink hinges) combined with confounding adjustment through an additive logit term. Positivity is thin: 26 / 40 / 54 untreated lots in the <500 / 500–600 / 600–700 bands (83–88% treated). Yard B has 49 untreated lots out of 358. Bias near 15% shifts tiers and feeds D16/D17. *Fix:* add splines on weight, shrink and crowding. Report recovery by band with bootstrap CIs. Consider a sensitivity analysis on the treatment-model form (multiplicative vs logit).
- **D7: oracle recovery against the wrong estimand.** `_true_p0` includes the lot effect. RMSE against it is **0.156** (an irreducible floor), compared with **0.030** against E[p|x]. §1 defines p0 as the conditional expectation, so recovery and the "oracle planner" must use E[p|x] (Bayes oracle). The clairvoyant oracle should be reported separately as an unattainable bound.
- **D9: CCC/MSEP cannot fail and is invalid as used.** The true E[p|x] scores only **CCC 0.515 with 100% "random" MSEP** against observed morbidity. The brief's "CCC 0.58, 99.7% random" is indistinguishable from a perfect model, so it carries no information. It is also unweighted across lots of 40–320 head, and computed on observed, post-treatment morbidity. *Fix:* report it beside the simulated ceiling, weight by head, or drop it.
- **D13: aleatoric SD is understated.** resid_sd is **1.14 vs the DGP's 1.3**. It is estimated in-sample from a population-averaged fit, with a delta-method binomial correction applied to empirical logits (+0.5). *Fix:* fit a beta-binomial or random-intercept model out of fold, or calibrate resid_sd on held-out lots.
- **D12: day-0 and check-in disagree.** Arrival P(≥15%) and the day-0 check-in prior with no pulls differ by up to **0.147** (mean −0.095) for the same lots. That is two definitions of the same quantity in one UI. *Fix:* use one shared function.
- **D18: the stress tests point in the favourable direction.** S4 (true RRR 0.25 < 0.45) *helps* a plan that withdraws metaphylaxis: realised saving rises to **21.2k vs 14.96k estimated**. The adverse case (true RRR 0.65, zero ENH effects) gives realised saving **0.64k/week, a loss in 52% of weeks, +1.85 deaths/week** against the current approach. *Fix:* stress both directions and add zero-ENH-effect worlds. Headline the worst case.
- **D20: the MC verdict is judged inside its own model.** `uncertainty.compare` evaluates the chosen plan on the same model's epistemic draws (no aleatoric noise, no model-form error). In 10 fresh weeks, mean P(cheaper) was 0.93 and the verdict was "confident" in 8/10. Those weeks happened to be cheaper on the truth, but only in a world where truth = assumption (D14). *Fix:* report verdict calibration from the regret study across stress worlds.
- **D21: the imputation story is incoherent.** Training uses median fill plus missing indicators (`risk.py:111`). Prediction swaps in donor draws but keeps the indicators (`risk.py:176-177`), so the indicator coefficient (learned at the median) is applied to non-median values. The wider intervals for missing lots (**9.6 vs 4.1 pp**) are an artefact of this mismatch, not proper multiple imputation (MI) with Rubin pooling. *Fix:* do MI at fit time (chained or donor) with pooled refits, or drop the donor step and propagate indicator uncertainty honestly.

**MINOR**

- **D10:** the R2 freeze is not implemented: no config hash, no manifest, no fresh-seed list. `dgp.lot_unobserved_sd` and `intercept` were tuned while watching AUC (`assumptions.yaml:163,178`). The existing test `test_synthetic_auc_not_overclaimed` (window 0.65–0.88) cannot fail in practice and is the very metric that was tuned.
- **D11:** the design claims median − mean < 1 pp, but it reaches **2.2–2.7 pp**. Outcomes are linear in p, so use the mean.
- **D5/D6:** the calibration slope, calibration-in-the-large and the B0/B1/B2 baselines are missing from the code. The probe (temporal split, Brier score): B0 0.182, B1 weight-only 0.158, B2 yard×source 0.153, full 0.141. The full model wins, but the result is not shown.
- **D17:** the care floor relies on a biased point estimate, so truly-high lots receive STD about 0.1–0.2 times per week (within 1% on the test seeds; regression guard).
- **D19:** forking paths and magic numbers: verdict thresholds 0.70 / 0.80 (`uncertainty.py:410`), "≥2 missing critical fields", 5/95 percentiles, "1.5× expected + 3", `DEMO_SEED=108` chosen after looking at outputs, and `SLACK_PENALTY`. These break rule 2. Move them to config and disclose them.
- **D22:** the high_risk label is *observed post-treatment* morbidity ≥15% (treated lots 0.304 vs untreated 0.215). The reported AUC therefore validates a different estimand from the one the decision uses (untreated p0). Say so, or validate on untreated lots or on fresh untreated worlds.
- **D23:** the penalty C = 1 is applied to weights normalised to a mean of 1 over 2n pseudo-rows. Its effective strength depends on n, not on head count, so "fixed a priori" has a data-dependent meaning. The metaphylaxis adjustment coefficient is also penalised, which leaves residual confounding.
- **D24:** design §8 cites `docs/LITERATURE_REVIEW.md`, which does not exist. Positivity diagnostics (§4) are not shown anywhere in the code or the UI.

### Leakage audit result
I found no direct leakage. Current arrivals carry no outcome columns. The prediction path forces meta = 0. Medians, scaler and donors come from training rows only. No ID or date features enter the model. The look-ahead risk in the check-in is controlled: the function sees only pulls to date. However, `simulate_pulls` draws "truth" from the model's own prior, so the demo cannot reveal model error. The remaining "leakage" is researcher-level: generator tuning (D10), the truth equalling the assumption (D14), and seed selection (D19).

### Acceptance tests (`tests/critic/`) and expected APIs
| Test | Fault | Today |
|---|---|---|
| test_leakage: prediction forces meta 0; no ID/date features; preprocessing train-only | D0 guard | pass |
| test_leakage::test_temporal_split_exists_and_is_strict: `src.model.validation.temporal_split(hist, cutoff) -> (train, test)` | D3 | FAIL |
| test_leakage::test_leave_one_yard_out: `leave_one_yard_out(hist, cfg, n_boot=0) -> DataFrame[yard, n_test, auc, brier, cal_in_the_large, cal_slope]` | D3 | FAIL |
| test_leakage::test_temporal_validation_can_fail: `src.model.stress.generate_scenario("drift", seed) -> (hist, eval)` with a ≥3 pp year shift | D4 | FAIL |
| test_validation_protocol::evaluate keys `cal_in_the_large`, `cal_slope`, `split=="temporal"` | D5 | FAIL |
| ::baselines: `validation.baselines(train, test, cfg) -> {B0,B1,B2,full: {brier, auc}}`, full beats B1 | D6 | FAIL |
| ::oracle_recovery: `validation.oracle_recovery(model, cfg, n, seed) -> DataFrame[band, n, bias, rmse]` against E[p\|x] | D7 | FAIL |
| ::untreated_risk_unbiased_near_threshold: \|bias\| < 0.75 pp per band | D8 | FAIL |
| ::ccc_reported_with_noise_ceiling: `evaluate()["adequacy"]["ccc_ceiling"]` | D9 | FAIL |
| ::ccc_ceiling_is_far_below_one | D9 evidence | pass |
| ::generator_frozen_with_manifest: `generate.config_hash()` + `data/generator_manifest.json{seed, config_hash, tuning_seeds, fresh_eval_seeds}` | D10 | FAIL |
| test_uncertainty::p_high_is_calibrated (\|Δ\| < 0.03) | D1 | FAIL (0.447 vs 0.359) |
| ::epistemic_interval_covers_estimand (≥ 0.85) | D2 | FAIL (0.772) |
| ::point_estimate_is_the_mean (< 1 pp) | D11 | FAIL (2.2 pp) |
| ::day0_and_checkin_agree (< 0.02) | D12 | FAIL (0.147) |
| ::aleatoric_sd_not_understated (≥ 0.95 × 1.3) | D13 | FAIL (1.14) |
| test_decision_regret::true_effect_is_not_the_planner_assumption | D14 | FAIL |
| ::regret_study_api: `src.decision.policy_eval.regret_study(model, cfg, n_weeks, lots_per_week, seed, true_effects=None) -> {per_week: DataFrame[week, policy, realised_cost, cases, deaths, regimens, floor_violations, regret], summary}`, policies ⊇ {recommended, current, oracle}, regret ≥ 0 | D15 | FAIL |
| ::regret_well_defined_with_estimate_dependent_floor | D16 | FAIL (6/6 negative) |
| ::truly_high_lots_not_left_on_standard_care (≤ 1%) | D17 guard | pass |
| ::stress_includes_adverse_effect_direction: `src.model.stress.SCENARIOS` with `true_metaphylaxis_rrr` > planner and `true_enhanced_monitoring_cfr_rrr` == 0; `run_stress(...) -> DataFrame[scenario, auc, cal_slope, mean_regret, realised_saving, p_loss_week, d_deaths]` | D18 | FAIL |
| ::verdict_thresholds_in_config | D19 | FAIL |

### Required before claiming §7 "will claim" items
D14, D15, D16 and D1 must be fixed. Until then, "quantified regret against an oracle" and "measured advantage in simulation" are unsupported, and the second of these is circular.

## Round 2 - verification of fixes and new findings (2026-11-14)

Scope: `src/` after the builder's round-1 fixes (including the late `sorted(set(...))` fix in `_donor_fill`), `reports/validation_results.json` and `docs/VALIDATION_REPORT.md` (config hash `e1c24b9bc2d314db`), and app/docs text. New probes: `scripts/critic/probe_r2_replicates.py` (10 independent histories, seeds 70000+), `probe_r2_misspec.py` (source of lot-level bias), `probe_r2_decisions.py` (realised-outcome loss weeks, seed stability, invariance, permutation canary, monotonicity). New and rewritten tests: `tests/critic/test_replicates.py`, `tests/critic/test_round2.py`, plus two in `test_decision_regret.py`. Tests marked `slow` refit several models, about 1 minute each; the whole critic suite takes about 10 minutes. I also fixed my own helper: `simworld.fresh_lots(rrr=...)` still wrote the planner's parameter, which became a no-op after D14; it now writes `dgp.true_metaphylaxis_rrr`.

### Rulings on the builder's three rebuttals
1. **D16: ACCEPTED.** The round-1 test gave the oracle `risk_lo = true p`, so the rule-6 clause "High lots with risk_lo ≥ 15% keep metaphylaxis" bound for every truly-high lot. The negative "regret" was therefore the cost of the floor, and the floor is not negotiable. A floor-free oracle is a valid lower bound. The test is replaced by `test_regret_definitions_policy_eval`, which checks:
   - regret ≥ 0 for every policy and week;
   - oracle regret = 0;
   - `oracle_floor` has zero violations against TRUE risk;
   - its mean regret equals `cost_of_floor`.

   Condition: the headline must say that the oracle ignores the floor *and* has unlimited hospital capacity (`_oracle` sets `hosp_free=1e9`). Part of every policy's "regret" is therefore the cost of rules and limits, not planning error.
2. **D2: ACCEPTED in substance; the shortfall is real but small.** One dataset's coverage is a single correlated draw, so the single-CSV test was unsound.
   - Over 10 fresh histories, mean coverage is **0.849** (range 0.68-0.94). The builder's 8 histories gave 0.88.
   - The sharp check is whether the bootstrap SD matches the TRUE sampling SD of each lot's prediction across histories. The ratio is **1.06** (median over lots) and **1.11** (mean), so the bootstrap SD is close to honest.
   - The remaining 2-5 point shortfall comes from duplicated draws (R2-6) and a small lot-level bias (R2-2).
   - New tests in `test_replicates.py` (6 histories): SD ratio ≤ 1.25 and mean coverage ≥ 0.83. Both pass.
   - **Required:** describe the range as "about 85-88% coverage" rather than "calibrated" until R2-6 is fixed.
3. **D8: ACCEPTED.** Band-mean bias over 10 fresh histories (pp, mean ± SE):

   | Weight band (lb) | Bias (pp) |
   |---|---|
   | <500 | −0.25 ± 1.07 |
   | 500-600 | −0.20 ± 0.67 |
   | 600-700 | −0.04 ± 0.40 |
   | 700-800 | +0.14 ± 0.18 |
   | ≥800 | +0.16 ± 0.14 |

   The CSV's −1.72 pp in the 600-700 band is −1.4 SD within that distribution, so it is sampling error. But the per-history SD in the <500 band is **3.4 pp** (one history −6.5 pp). That is the price of only 26 untreated light lots (positivity), and the report should say so. The test is rewritten as multi-history mean bias (|mean| ≤ 2.5 SE + 0.25 pp per band). It passes, and it still fails for a systematic −1.4 pp.

### Status of D1-D24
| ID | Status | Round-2 evidence |
|---|---|---|
| D1 P(≥15%) | RESOLVED | `test_p_high_is_calibrated` passes; report mean 0.37 vs true share 0.36 |
| D2 interval coverage | PARTIAL | Multi-history mean 0.85-0.88 vs nominal 0.90; bootstrap/sampling SD ratio 1.06; residual causes R2-2, R2-6 |
| D3 splits | RESOLVED | `temporal_split` and `leave_one_yard_out` implemented and reported (LOYO AUC 0.71-0.83) |
| D4 temporal can fail | RESOLVED | Drift world: temporal calibration-in-the-large +3.8 pts; nothing acts on it (R2-8) |
| D5 calibration slope / ITL | RESOLVED | `evaluate()` returns both: slope 1.06, ITL +0.3 pts |
| D6 baselines | RESOLVED | Brier: B0 0.182, B1 0.158, B2 0.153, full 0.134 |
| D7 estimand | PARTIAL | Recovery is now against `_marg_p0`, but that conditions on COMPLETE covariates, not recorded ones (R2-9) |
| D8 band bias | RESOLVED (sampling error) | Ruling 3; lot-level residual in R2-2 |
| D9 CCC | RESOLVED in code/app | CCC 0.48 vs simulated ceiling 0.49; brief and DEMO_CASE still quote 0.58 (R2-10) |
| D10 freeze | RESOLVED | Manifest and hash test pass; tuning seed disjoint from evaluation seeds |
| D11 mean | RESOLVED | `assess` uses the mean; test passes |
| D12 day-0 vs check-in | RESOLVED | Shared `prob_high_from_draws`; test passes |
| D13 lot SD | RESOLVED | OOF MLE 1.28 (DGP 1.30); 1.23-1.37 across 10 histories |
| D14 truth ≠ assumption | RESOLVED (mechanism) | `dgp.true_*` drive the simulator; the base study still has truth = assumption (R2-1, R2-11) |
| D15 regret study | RESOLVED | `policy_eval.regret_study` with oracle, oracle_floor and verdict calibration |
| D16 regret definition | RESOLVED | Ruling 1; new definitions test passes |
| D17 truly-high on STD | **REGRESSED** | Guard now fails: 2 of 134 truly-high lots on STD over 12 weeks; over 40 weeks 7/436 = 1.6% (≈0.18/week); 5 of the 7 were flagged borderline by the tool (R2-13). Drift world: 0.40/week (R2-8) |
| D18 two-sided stress | RESOLVED | S5/S6/S7 added; S7 headlined in report and app |
| D19 magic numbers | PARTIAL | Verdict, interval and escalation cut-offs now in config; `SLACK_PENALTY = 1e4` still in code; DEMO_SEED is **150** in code (builder note says 64; docs say 108) |
| D20 verdict calibration | OPEN → R2-1 | Calibrated only where truth = the prior's mode; under S6/S7, 'confident' in 72% of weeks and right in 6% of them |
| D21 imputation | RESOLVED (caveat) | Donor MI at fit and predict, same indicators, refit per member. Caveat: pools are yard × source only and ignore the shrink-transit link (DGP shrink = 2.5 + 0.12 × transit), so the imputation is incongenial for lots missing shrink but not transit (R2-9) |
| D22 label ≠ estimand | RESOLVED (disclosed) | Report §2.1 states it; §3 checks untreated risk against the truth |
| D23 penalty | OPEN (MINOR) | C = 1 on weights normalised by 2n pseudo-rows; part of the shrink-to-mean pattern in R2-2 |
| D24 lit review / positivity | PARTIAL | `LITERATURE_REVIEW.md` exists; positivity is in the report but not in the app |

### Leakage audit (round 2)
No leakage found in the new code.
- Donor pools are built from the rows passed to `fit()`. In `evaluate()` and LOYO these are training rows only (the held-out yard falls back to the all-yards pool), and donors carry no outcome columns.
- The OOF lot SD uses fold-train donors and fold-train fits.
- Canaries pass:
  - `test_evaluate_ignores_test_outcomes`: scrambling 2026 outcomes leaves every 2026 prediction bit-identical.
  - `test_permutation_canary_auc_collapses`: permuted outcomes give a temporal AUC of 0.54.
- Determinism across `PYTHONHASHSEED` holds after the builder's fix (`test_fit_is_deterministic_across_hash_seeds`, which runs two subprocesses).

**Is the replicate-world evidence circular?** Partly, by design. The histories share the seed-42 yard effects, the DGP, and a model class fitted to that DGP. They separate sampling error from systematic error *within the assumed world*. They say nothing about misspecification, new yards or real data; S1-S3 and LOYO cover those. The report's "approximately unbiased estimator" claim needs the qualifier "in our own simulated world".

### New findings

**MAJOR**

**R2-1: the 'confident' verdict is confidently wrong inside the planner's own declared ranges.**
- *Mechanism.* `uncertainty.sample_params` draws each effect from triangular(low, value, high). "Confident" therefore means "confident under a prior centred on our assumption".
- *Evidence.* S6 sets the enhanced-monitoring and receiving effects to 0, which is the low end of their declared ranges, not outside them. There the tool said "confident" in **72-75%** of weeks and was truly better (cheaper and no more deaths) in **0-6%** of them (validation JSON rows S6/S7; my test reproduces 0.75 / 0.00, +2.46 deaths per week).
- The only verdict calibration quoted in the report text is the base world's 100%, where truth equals the prior's mode, so it is uninformative.
- The death benefit rests almost entirely on the assumed enhanced-monitoring CFR effect, which rule 3 says these data cannot validate.
- *Fix.* Either (a) "confident" requires the criteria to hold at the unfavourable end of every unvalidated intervention range (corner or one-at-a-time check), or (b) relabel the verdict "favourable if the assumed effects hold" and show the S6/S7 verdict calibration next to the base figure.
- *Test.* `test_verdict_not_confidently_wrong_inside_declared_ranges` (slow).

**R2-11: the decision headlines read as findings but rest on truth = assumption.**
- *Where.* VALIDATION_REPORT headline 4, and the app's "Weekly regret vs oracle" and "Realised saving vs current" metrics. In the base world the true enhanced-monitoring and receiving effects equal the planner's.
- "Realised saving $13,310 (tool estimated $13,463)" therefore agrees by construction.
- "Cost more in 0% of weeks" is scored on expected values given E[p|x], with no lot-level outcome noise. With realised lot effects, the probe gives loss weeks **3%** and more deaths in **10%** of weeks (`probe_r2_decisions.py` [1]).
- *Fix.* Label the base numbers "if our assumed intervention effects are right". Put S6/S7 regret next to them: recommended-plan regret $11,730 (S6) and $12,470 (S7) vs $4,933 (base). Rename "realised" to "expected under the simulated truth", or add an outcome-noise version.

**R2-3: risk falls when commingling rises or pen space shrinks (face validity, live-demo risk).**
- *n_sources.* On the shipped model, +5 sources LOWERS risk for 22 of 24 demo lots (down to −1.1 pp). The coefficient is −0.009 per source vs DGP +0.06, and 35% of bootstrap members are negative.
- *Pen space.* −5 m²/head lowers risk (positive coefficient in 95% of members vs the DGP crowding penalty).
- *Bunk space.* It is absent from the DGP but gets a non-zero coefficient (90% of members negative).
- *Cause.* Weak identification: small DGP effects, n_sources collinear with auction source, 35% missing in auction lots. Yet these coefficients drive the "drivers" text and the what-if, so a judge who types "more sources" sees risk fall.
- *Fix.* Sign-constrain literature-signed factors (sources, transit, shrink and crowding never protective), or drop factors whose bootstrap interval spans 0. Never print a driver whose sign contradicts the literature.
- *Test.* `test_risk_monotone_in_literature_signed_factors`.

**R2-13: borderline lots are not protected by the care floor (rule 6 with rule 7).**
- *Mechanism.* The floor protects only lots whose point estimate is High. Lots the tool itself flags "borderline" (90% range crosses 15%) can get standard care only.
- *Evidence.* They are most of the truly-high lots left on STD: 5 of 7 over 40 fresh weeks. On 12 fresh weeks, 4 lots with risk_hi 0.16-0.18 got STD. This is why the D17 guard now fails (1.6% vs 1%).
- *Fix.* Borderline lots get at least enhanced monitoring, or an escalation if rider hours bind. The cost is small: these are few lots per week.
- *Test.* `test_borderline_lots_not_left_on_standard_care`.

**R2-7: the demo plan depends on the Monte Carlo seed.**
- *Evidence.* With app defaults (n_boot = 30, 60 draws), changing only the bootstrap seed changes **2, 2 and 0** of 24 lots' options (seeds 1-3 vs 0). One lot's risk moves by up to **2.8 pp**, and confidence labels flip (up to 4 lots in the probe).
- *Fix.* Use at least 200 bootstrap members (asked in round 1), or report the Monte Carlo error and treat options within it as ties.
- *Test.* `test_recommendation_stable_across_bootstrap_seeds` (slow).

**R2-8: drift is detected but not acted on, and the floor fails 10× more often.**
- *Evidence.* In the drift world, temporal calibration-in-the-large is +3.8 pts. Yet truly-high lots on STD rise from 0.04 to **0.40 per week**, and the estimated saving ($12.1k) misses the realised saving ($18.8k) by 55%. The report presents this favourably. "Monthly recalibration" exists only in DESIGN §8.
- *Fix.* Implement intercept recalibration on a recent window (or add year as a covariate), or raise a drift alarm when recent calibration-in-the-large exceeds a configured bound. Report floor violations under drift in the headline.

**MINOR**

- **R2-2: lot-level shrinkage toward the average lot.**
  - *Evidence.* Band means cancel, but over 10 histories 20% of lots have a mean prediction more than 2 SE from the truth (≈8% expected).
  - *Pattern.* +0.5 pp for truly low-risk lots, −1.0 pp at 30-50% true risk; auction, high-weather and crowded lots are under-predicted. This is what a logistic fit with an L2 penalty does to random-intercept data (D23).
  - *Size.* On complete-record lots the de-noised RMS bias is **0.77 pp** vs a bootstrap SD of **1.90 pp**, so it is small.
  - *Test.* Guard `test_lot_level_misspecification_bias_small` (≤ 0.5 × bootstrap SD) passes.
  - *Optional fix.* Fit the random-intercept binomial GLMM already used for the lot SD for all coefficients, then integrate to the marginal mean.
- **R2-4: a lot's risk depends on its row position.** The donor-imputation RNG is consumed row by row, so permuting rows and renaming lot_ids moves risk_mean by up to **0.65 pp** (options unchanged this week). *Fix:* per-lot RNG streams keyed on covariates, or more imputations. *Test:* `test_recommendation_invariant_to_row_order_and_lot_id`.
- **R2-5: hash-seed non-determinism (reported by the builder) is fixed.** Guard test added; it passes.
- **R2-6: the "90%" range is about 87% even in theory.**
  - `assess` takes 60 draws from 30 cycled members, so complete-record lots get 30 values, each repeated twice.
  - The Weibull 5-95% interval of that set is roughly the [2nd, 29th] of 30 members, with expected coverage **0.871** (simulated).
  - The report's replicate run (n_boot = 30, 40 draws) double-weights 10 members.
  - *Fix:* set n_draws = n_boot (or use more members than draws). *Test:* `test_interval_uses_distinct_bootstrap_members`.
- **R2-9: the truth label is wrong for lots with missing data.**
  - `_marg_p0` is computed from complete, unmasked covariates, but VALIDATION_REPORT §3.1 calls it "E[p | recorded data]".
  - For lots with missing fields, "bias", coverage and the regret oracle therefore include information no model could have.
  - *Fix:* relabel it E[p | complete covariates], or report recovery on complete-record lots separately (my R2-2 guard does this). Also impute shrink conditional on transit.
- **R2-10: stale or false text** (builder to fix):
  - `docs/TECHNICAL_BRIEF.md:60` still gives the old random-split numbers: "synthetic hold-out, 350 lots... AUC 0.82, Brier 0.15, sensitivity 58%, specificity 83%, PPV 56%... concordance 0.58... 99.7% random". Current temporal results: n_test 378, Brier 0.134, sensitivity 0.40, specificity 0.95, PPV 0.72, CCC 0.48 vs ceiling 0.49. The brief also has **no validation, regret or stress section**, although that is now the strongest evidence.
  - `docs/DEMO_CASE.md`:
    - `:68`: should also cite LOYO AUC 0.71-0.83.
    - `:69`: "seed 108 sits at the median of 200 seeds", 125 pulls. The code has DEMO_SEED = 150 and a median of 123.
    - `:71`: "Concordance 0.58... 99.7% random error", with no ceiling.
  - `docs/DESIGN.md`:
    - `:101`: "we use the median of the epistemic draws" is now the mean.
    - `:36` and `:38`: should point to the implemented tests.
    - §5 "status" is stale.
  - `data/DATA_DICTIONARY.md:51`: "synthetic PPV of about 56%" is now 0.72.
  - `data/DATA_DICTIONARY.md:60`: seed 108 and 125 pulls are now 150 and 123.
  - The builder's message says DEMO_SEED is 64; the code says 150.
  - `docs/VALIDATION_REPORT.md`:
    - "approximately unbiased recovery" needs "within our simulated world";
    - headline 4 needs the R2-11 caveat;
    - headline 3 should say 85-88%;
    - headline 5 "Worst case" should say "most adverse scenario tested" (R2-12);
    - the S6/S7 verdict calibration (0.06) belongs in the text.
  - App "Model & evidence" tab: the regret and realised-saving metrics lack the truth = assumption caveat (R2-11), and positivity is not shown (D24).
- **R2-12: the "worst case" is only the worst corner of the effect ranges.** S7 sets the planner's own range ends (meta 0.65, enhanced monitoring 0, receiving 0) in an otherwise correctly specified world. It is never combined with S2 heterogeneity or drift, and no economics range is stressed. *Fix:* add one combined scenario (S7 + S2 + drift), or rename the label.

### Critic tests at the end of round 2
| Test | Fault | State |
|---|---|---|
| test_replicates::untreated_risk_unbiased_across_histories | D8 rewritten | pass |
| test_replicates::bootstrap_spread_matches_sampling_spread | D2 rewritten | pass |
| test_replicates::mean_coverage_across_histories | D2 rewritten | pass |
| test_replicates::lot_level_misspecification_bias_small | R2-2 guard | pass (0.77 vs 1.90 pp) |
| test_decision_regret::regret_definitions_policy_eval | D16 rewritten | pass |
| test_decision_regret::truly_high_lots_not_left_on_standard_care | D17 guard | **FAIL** (2 of 134) |
| test_decision_regret::verdict_not_confidently_wrong_inside_declared_ranges | R2-1 | **FAIL** (75% confident, 0% truly better) |
| test_round2::evaluate_ignores_test_outcomes | leakage canary | pass |
| test_round2::permutation_canary_auc_collapses | leakage canary | pass (0.54) |
| test_round2::fit_is_deterministic_across_hash_seeds | R2-5 | pass |
| test_round2::recommendation_invariant_to_row_order_and_lot_id | R2-4 | **FAIL** (0.65 pp) |
| test_round2::risk_monotone_in_literature_signed_factors | R2-3 | **FAIL** (n_sources −1.1 pp) |
| test_round2::interval_uses_distinct_bootstrap_members | R2-6 | **FAIL** (30 of 60 distinct) |
| test_round2::borderline_lots_not_left_on_standard_care | R2-13 | **FAIL** (4 lots) |
| test_round2::recommendation_stable_across_bootstrap_seeds | R2-7 | **FAIL** (2 flips, 2.8 pp) |
| test_uncertainty, test_leakage, test_validation_protocol (remaining) | D1, D3-D7, D9-D13 | pass |

### Verdict (round 2)
The statistical core is now sound:
- no leakage;
- the band bias is sampling error;
- the uncertainty is close to honest (2-5 points under nominal, for known reasons);
- the regret is correctly defined.

What remains is in the **decision layer and its presentation**:
- a verdict that says "confident" when its death claim rests on an unvalidated effect (R2-1);
- base-world headlines that read as findings (R2-11);
- borderline lots left on standard care (R2-13);
- risk that falls as commingling rises (R2-3);
- a seed-dependent demo plan (R2-7);
- drift that is detected but not acted on (R2-8).

Each fix is small. R2-1, R2-11, R2-13 and R2-3 are what a scientist judge would find within five minutes.

## Round 3 - re-review after the round-2 fixes (2026-11-14)

**Scope.** `src/` as of 14:31, plus `reports/validation_results.json` and `docs/VALIDATION_REPORT.md`, both regenerated at 16:03, after the fixes (same config hash `e1c24b9bc2d314db`). I did not accept any builder claim without a probe or a test.

**New probes:**
- `scripts/critic/probe_r3_drift.py`: false-alarm rate and power of `check_drift` over 80 histories.
- `scripts/critic/probe_r3_floor_verdict.py`: cost of the borderline floor, and how often the three-state verdict says "robust".

**New tests** (in `tests/critic/test_round2.py`):
- `test_drift_check_operating_characteristics` (slow);
- `test_headline_worst_case_reports_worst_health_outcome`.

**Suite result.** The full critic suite (35 tests, before the two new ones) **passes**: 35/35, which took 80 minutes because three model-fitting jobs ran at once. Of the two new tests, the drift test passes and the worst-case test fails.

### Status of the round-2 findings
| ID | Status | Round-3 evidence |
|---|---|---|
| R2-1 verdict | RESOLVED | "Robust" requires winning at all 8 corners of the unvalidated effect ranges. The robust share is 0% in S6, S7 and S8, and 3% in base (2 of 60 weeks, both truly better). Probe: 1 of 12 fresh weeks robust, 7 favourable-if-assumed, 4 low confidence, so the label is not vacuous. `test_verdict_not_confidently_wrong_inside_declared_ranges` passes. The base-world "100%" in report §4 rests on 2 weeks; state the n (MINOR). |
| R2-2 lot-level shrinkage | Guard passes | — |
| R2-3 monotonicity | RESOLVED | Active-set sign constraints; `test_risk_monotone_in_literature_signed_factors` passes. Bunk-space sign is tagged ASM, which is fine. |
| R2-4 row order | RESOLVED | Per-lot imputation streams; test passes. Caveat in R3-4. |
| R2-5 determinism | RESOLVED | Subprocess test passes. |
| R2-6 distinct draws | RESOLVED | 100 distinct members; test passes. Replicate coverage is now a **90%** mean (range 83-96%) and 88% for the CSV (report §3.3). |
| R2-7 seed stability | RESOLVED | Seed-independent point estimate (10 full-data members on fixed streams) plus logit recentring; test passes. |
| R2-8 drift | RESOLVED, caveats in R3-2 | Shipped data: z = +0.96, no alarm. Drift world: alarm ON, offset +0.22, floor violations 0.40 → **0.00 per week**. |
| R2-9 estimand label | RESOLVED | Report §3.1 now says "E[p \| complete covariates]". |
| R2-10 stale text | PENDING | TECHNICAL_BRIEF, DEMO_CASE and DATA_DICTIONARY are unchanged since 12:43 (builder: to do after validation). See below. |
| R2-11 truth = assumption caveat | RESOLVED | Headline 4 carries the caveat and places S6 ($11,580), S7 ($12,922) and S8 ($22,839) regret beside the base result. It says "expected saving under the simulated truth" and states the oracle's floor and hospital assumptions. |
| R2-12 combined scenario | RESOLVED | S8_combined added. Its headline selection is wrong, see R3-1. |
| R2-13 borderline floor | RESOLVED | Floor violations are 0.00 per week in all 10 worlds (report §5). Probe over 12 weeks: the floor costs about **$10 per week** and 0.4 regimens per week, with deaths −0.06 per week. It is cheap. |
| D17 truly-high on standard care | RESOLVED | Guard passes; recommended plan 0.00 per week. |
| D19 magic numbers | RESOLVED | `slack_penalty_usd`, the drift cut-offs and `robust_effect_keys` are in config. |
| D24 positivity | RESOLVED | Positivity table is in the evidence-tab expander (`app.py:894`). |

### Verification of the new mechanisms
- **Drift check** (`probe_r3_drift.py`):

  | Histories | n | z mean (SD) | Alarm rate |
  |---|---|---|---|
  | No drift, base world | 30 | −0.18 (1.08) | 1/30 |
  | No drift, S2 | 15 | — | 0/15 |
  | No drift, S1 | 15 | z mean +0.60 | 0/15 |
  | Drift 0.35 logit/yr | 10 | +3.77 | 10/10 (offset +0.22) |
  | Drift 0.15 logit/yr | 10 | — | **1/10** |

  The self-normalised z is close to N(0, 1) under the null, including in misspecified worlds. That makes it a sound test.
- **Leakage.** `check_drift` uses only the training rows' window and fits a model on the pre-window rows, so it brings no test information into `evaluate()`. The canaries still pass.
- **Recentring.** P(≥15%) calibration (D1) and day-0/check-in agreement (D12) still pass after the logit recentring of draws.

### New findings
**MAJOR**
- **R3-1: the "most adverse scenario" headline is chosen by cost alone and hides the worst health outcome.**
  - `scripts/run_validation.py:125` and `app/app.py:855` both use `min(stress, key=realised_saving)`. On the current results that picks S8: saving −$4,880 per week, but deaths **−2.34** per week, i.e. fewer than the current approach.
  - S7 is the world with **+4.49 deaths per week** and a loss in 80% of weeks, and it appears in no headline.
  - A reader of headline 5 or the app's "Most adverse scenario tested" metric concludes that the worst case costs money but saves lives, which is the opposite of S7.
  - This breaks the brief's rule that health and cost are reported together, with health first (rule 6).
  - *Fix:* report the worst scenario on each dimension (lowest saving and most extra deaths), or headline S7 for deaths and S8 for cost, in both the report and the app.
  - *Test:* `test_headline_worst_case_reports_worst_health_outcome`, which **fails**.
- **R2-10 (pending, MAJOR if submitted as is).**
  - `docs/TECHNICAL_BRIEF.md:60` still reports the random-split numbers (350 lots, Brier 0.15, PPV 56%, CCC 0.58), and the brief has no validation or regret section.
  - `docs/DEMO_CASE.md:69,71` and `data/DATA_DICTIONARY.md:51,60` still say seed 108, 125 pulls, CCC 0.58 and PPV 56%.
  - Because the model changed (100 members, sign constraints, recentring, borderline floor), **every lot-level number in DEMO_CASE** (C0015, C0008, hospital pen 24.3 vs 22.5, 893 regimens, the verdict text) must be regenerated, not just the seed.
  - The demo week's verdict is now "favourable if assumed effects hold", so any "confident" wording in the demo script is false.
  - The brief is a required submission item. I will judge this once it has changed.

**MINOR**
- **R3-2: the drift alarm only catches large shifts, and it recalibrates the level only.**
  - Power is 10% at 0.15 logit/yr (about +1.5 pp in the window).
  - The offset is estimated in-sample after the full model has partly absorbed the shift. In the drift world the estimated saving ($13.8k) still misses the true saving ($20.4k) by 48%.
  - *Fix:* state the detectable shift (about 0.3 logit, or about 3 pp of morbidity, over 180 days) in the app and the report. Optionally add arrival year as a covariate, or recalibrate on an out-of-sample offset.
  - `scripts/run_validation.py:253` still says "The fix in practice is monthly recalibration (DESIGN s8)". It should point to `check_drift`.
- **R3-3: the coverage wording contradicts its own numbers.** Headline 3 says "about 85-88%" and then reports a 90% mean (83-96%) with 88% for the CSV. Say "about 88-90% on average (83-96% across histories)".
- **R3-4: the imputation-stream key can collide.** `_lot_keys` hashes categorical fields plus head count. Two lots in one upload with the same yard, source, sex, vaccination history, quarter, weather and head count get identical donor draws for every missing field. Their imputations are then perfectly correlated, although the expected value is not biased. There are no collisions in the 24-lot demo week, but a simulated 1,000-lot upload has 16 duplicate keys, and rounded head counts would make collisions more frequent. *Fix:* add the row's observed numeric fields to the key, or salt it with an occurrence index among duplicate keys. The what-if stability argument still holds if the key uses only fields the what-if does not edit.
- **R3-5: the verdict calibration n is too small to report as a percentage.** Report §4 says the plan was "truly cheaper with no more deaths 100% of the time" on 2 weeks. Report the count, e.g. "2 of 2".

### Tests at the end of round 3
- All 35 earlier critic tests pass. This includes the 8 that failed in round 2: D17, R2-1, R2-3, R2-4, R2-6, R2-7, R2-13, and seed stability under app defaults.
- New `test_drift_check_operating_characteristics`: pass.
- New `test_headline_worst_case_reports_worst_health_outcome`: **FAIL** (R3-1).

### Verdict (round 3)
The round-2 MAJOR findings are all fixed, and each fix is confirmed by a probe or a test. I found no leakage, no regression, and no new statistical fault. **One MAJOR issue remains in the code and report: R3-1**, a one-line selection rule. **One MAJOR issue remains pending in the submission documents: R2-10**, the stale brief, DEMO_CASE and DATA_DICTIONARY, which the builder has scheduled. Once both are fixed and the demo numbers are regenerated, I expect to be able to say there are no remaining MAJOR issues.


## Round 4 - verification of the R3 fixes and the regenerated documents (2026-11-14)

Scope:
- `scripts/run_validation.py` and `app/app.py` (R3-1);
- `src/model/risk.py::_lot_keys` (R3-4);
- the re-rendered `docs/VALIDATION_REPORT.md` (16:10), together with `reports/validation_results.json` (16:03);
- the rewritten `docs/TECHNICAL_BRIEF.md`, `docs/DEMO_CASE.md`, `data/DATA_DICTIONARY.md` and `SUBMISSION_OVERVIEW.md`.

New probe: `scripts/critic/probe_r4_demo_numbers.py` recomputes every demo figure from the default app path (`fit()` defaults, `assess`, `optimize`, 200 Monte Carlo draws, `corner_check`).

New tests, in `tests/critic/test_round2.py`:
- `test_docs_corner_claim_matches_corner_table`
- `test_duplicate_key_lots_invariant_to_row_order`

### Status of round-3 items
| ID | Status | Evidence |
|---|---|---|
| R3-1 worst health case | RESOLVED | Headline 5 now also reports S7 (+4.49 deaths/week, cost higher in 80% of weeks); the app does the same. `test_headline_worst_case_reports_worst_health_outcome` passes. |
| R2-10 stale documents | RESOLVED for numbers, see R4-1 for wording | The probe reproduces the brief's and DEMO_CASE's demo table exactly: current 354 / 24.8 / 3,345 / $167,311; plan 365 / 22.0 / 2,506 / $154,569; metaphylaxis-all 337 / 23.6 / 3,828 / $173,805; none 613 / 42.9 / 767 / $169,861. It also reproduces 3,406 head, 17 changes, 6 withdrawals (852 head), 13 lots on enhanced monitoring, the Yard B shortfall of 22.8 vs 22.5, the adverse corner (+$6.1k, +3.9 deaths) and a case-fatality break-even near 0.04-0.05. Every evaluation-table figure matches the JSON: regret $5.0k vs $21.9k; S6/S7/S8 regret $11.6k / $12.9k / $22.8k; coverage 90% (83-96%); drift z +1.0 / +4.4. The seed-108, PPV-56% and CCC-0.58 text is gone. |
| R3-2 drift power | RESOLVED | Headline 7 states the probe's detection rates and that the correction is level-only. |
| R3-3 coverage wording | RESOLVED | The report quotes the computed 90% (83-96%). |
| R3-4 key collisions | PARTIAL, see R4-2 | An occurrence index separates duplicate keys, but it reintroduces row-order dependence for those lots. |
| R3-5 verdict n | PARTIAL, see R4-3 | |

### New findings
**MAJOR**

**R4-1: the brief's "key finding" misstates the corner result. This is the main take-home message, and a judge can check it in the app.**

What the documents say:
- TECHNICAL_BRIEF §4: "The plan loses only at one corner: metaphylaxis works better than assumed **and** extra riding does nothing."
- DEMO_CASE: "At one corner the plan loses."
- VALIDATION_REPORT headline 6 and `run_validation.py:181`: "the plan always loses at one corner (metaphylaxis works better than assumed AND extra riding does nothing)."

What the demo week's corner table shows (the same table the app displays in the What-if expander):

| Metaphylaxis effect | Extra-riding effect | Receiving effect | Extra deaths per week |
|---|---|---|---|
| 0.25 | 0 | 0 | +1.5 |
| 0.65 | 0 | 0 | +3.9 |
| 0.65 | 0 | 0.25 | +0.3 |

So the plan has **more deaths at 3 of 8 corners**. Only the cost loss is unique to one corner (+$6.1k).

The death risk therefore does not require metaphylaxis to work better than assumed. It appears whenever extra riding has no effect on case fatality, even when metaphylaxis is at the *weak* end of its range. The managerial message should be: "deaths hinge on the extra-riding (case-fatality) effect; cost hinges on the metaphylaxis effect".

*Fix (text only):* correct the three documents and `render()`. In SUBMISSION_OVERVIEW §4 point 5, say which effect drives which outcome.

*Test:* `test_docs_corner_claim_matches_corner_table`, which **fails**.

**MINOR**

- **R4-2: duplicate-key lots depend on row order again.** `_lot_keys` gives the first duplicate the bare key and the second `key#1`.
  - Two lots that share the hashed fields but differ elsewhere (arrival weight, missingness pattern) therefore swap random streams when their rows swap.
  - Probe on the n_boot = 30 model: duplicating a demo lot with +150 lb and reversing the order moves risk_mean by **2.3 pp for C0017**, 1.0 pp for C0007 and 0.6 pp for C0010; risk_hi moves by up to 2.7 pp.
  - Duplicate keys are rare in a weekly upload, so this is MINOR.
  - *Fix:* break ties with the lot's own non-what-if fields (arrival weight, missingness pattern) rather than its position; reserve an index for fully identical rows.
  - *Test:* `test_duplicate_key_lots_invariant_to_row_order`, which **fails**.
- **R4-3: the brief says "the verdict never claimed robustness", but the JSON shows it did.**
  - `regret_base.headline.share_weeks_confident = 0.0333`, i.e. 2 of 60 base weeks robust, both truly better. The report's own §4 says "3% of weeks".
  - The builder's note ("the base regret run also gave 0% robust") is wrong. The 0% figures come from the 25-week stress runs (all 10 worlds at 0.0).
  - Headline 6's "base 0%" uses the stress base, which contradicts §4 three paragraphs later.
  - *Fix:* "robust in 2 of 60 base weeks (both truly better) and 0 of 25 weeks in every stress world". The finding is actually stronger stated this way.
- **R4-4: SUBMISSION_OVERVIEW §5 says "We iterated until no major issue remain[ed]".** That was not true when it was written (R4-1 is open). Make sure it is true at submission, or say "until the critic's remaining issues were minor or text-only".
- **R4-5: the Monte Carlo percentage is reported inconsistently.** DEMO_CASE and the brief say the plan is cheaper in 96% of draws. The default path gives **95%**, deterministically (`compare` seed 7, fixed draws). The disclaimer "can move by a point or two between runs" is not accurate for deterministic code: quote the computed value.
- **R4-6: the "full text" disclosure is not supported for Neal 2024.** The brief's disclosure says Baptiste 2017 and Neal 2024 were "verified against the full text". `docs/LITERATURE_REVIEW.md:8` says Neal 2024 was verified from the **abstract**. Say "abstract" for Neal (rule 1). Separately, `LITERATURE_REVIEW.md:46` still tabulates the metaphylaxis-only subgroup RR 0.42 (0.35-0.49), which line 7 says has not been re-located in the text. Mark it unverified in the table too.

### Tests at the end of round 4
All earlier critic tests pass on the current code. That includes `test_headline_worst_case_reports_worst_health_outcome`; the background run of `test_round2.py` and `test_decision_regret.py` is in the final report. New tests: `test_docs_corner_claim_matches_corner_table` (R4-1) **fails**, and `test_duplicate_key_lots_invariant_to_row_order` (R4-2) **fails**.

### Verdict (round 4)
- The code is in good shape, and every regenerated number in the documents reproduces from the code and the JSON.
- One MAJOR issue remains: **R4-1**, the brief's key-finding sentence about the corners, repeated in DEMO_CASE, the validation report and the overview. It is a text-only fix.
- The rest is minor.
- Once R4-1 is corrected and its test passes, there are no remaining MAJOR issues.


## Round 5 - verification of the R4 fixes (2026-11-14)

**Scope:** `src/model/risk.py::_lot_keys`, the re-rendered `docs/VALIDATION_REPORT.md`, the corner and robustness text in `docs/TECHNICAL_BRIEF.md`, `docs/DEMO_CASE.md` and `SUBMISSION_OVERVIEW.md`, and the builder's edit to my test. I added two new tests to `tests/critic/test_round2.py`: `test_docs_death_corner_claim_has_no_counterexample` and `test_lot_risk_independent_of_other_lots_in_upload`.

### Status of round-4 items
| ID | Status | Evidence |
|---|---|---|
| R4-1 corner claim | RESOLVED (residual wording in R5-1) | The single-corner phrases are gone and `test_docs_corner_claim_matches_corner_table` passes. Deaths are now attributed to the extra-riding effect and cost to metaphylaxis. DEMO_CASE lists the 3 losing corners with the correct values. |
| R4-2 row order with duplicate keys | RESOLVED (side effect in R5-2) | `test_duplicate_key_lots_invariant_to_row_order` passes. **The builder's test edit is accepted:** `worst = (0.0, "")` fixes a genuine bug in my test (comparing a str with None on an exact tie) and does not weaken the test. |
| R4-3 robust share | RESOLVED | Headline 6, the brief and the overview say "2 of 60 base weeks (both truly better), 0 of 25 in each stress world", consistent with the JSON. Report §4 still says "3% of weeks", which is consistent. |
| R4-4 overview claim | RESOLVED, given this round's verdict | "No remaining major issue" is true as of this round (see verdict). |
| R4-5 Monte Carlo share | RESOLVED (rebuttal accepted) | `p_cheaper` = 0.955 on the default path. The app formats 95.5 with `:.0f` as **96%**, so 96% is what users see. My earlier "95%" came from `%.2f` formatting in the probe. The disclaimer has been removed. |
| R4-6 Neal 2024 verification | RESOLVED | Neal is now cited as verified against the abstract and Baptiste against the full text. The Baptiste 0.42 / 0.53 subgroup is marked unverified and unused. |

### New findings (all MINOR)
- **R5-1: the new death claim is too strong, and the demo table contains a counterexample.**
  - *Claim.* The brief, report headline 6 and DEMO_CASE now say deaths rise "whenever extra riding does not lower case fatality, whatever metaphylaxis does".
  - *Counterexample.* The demo corner with metaphylaxis 0.25, extra riding 0 and receiving 0.25 gives **−4.7 deaths**.
  - *What the table actually shows.* Deaths rise when the enhanced-monitoring **bundle** (extra riding *and* low-stress receiving) does little. Of the four corners with zero riding effect, deaths rise at three: +1.5, +3.9 and +0.3. They fall at the fourth, where receiving is at 0.25. DEMO_CASE lists the three losing corners but omits the fourth, so its "whenever" reads as if all four lose.
  - *Why MINOR.* The managerial conclusion is right: the death advantage depends on the riding and receiving effects, cost depends on metaphylaxis, and a pen trial of the bundle is the most valuable next step. Only the generalisation is wrong. Fix it before submission anyway, because judges can open this table.
  - *Fix.* "Deaths rise when extra riding and low-stress receiving do little (3 of the 4 zero-riding corners; receiving at 25% offsets it when metaphylaxis is weak)."
  - *Test.* `test_docs_death_corner_claim_has_no_counterexample` **fails**. It reports the phrases in TECHNICAL_BRIEF, DEMO_CASE and VALIDATION_REPORT.
- **R5-2: an unrelated new lot can change a lot's risk (batch dependence).**
  - *Mechanism.* A lot's imputation key gains weight and missing pattern only when *another* lot in the upload shares its base key. Adding an unrelated lot can therefore change an existing lot's imputations.
  - *Evidence.* Adding a colliding copy (+150 lb) of a demo lot moves the original lot's risk_mean by up to **1.5 pp (C0004)**. This happens only on a base-key collision, which is rare in a weekly upload.
  - *Fix.* Always key on the lot's own fields (categoricals, head count, weight, missing pattern) and regenerate the demo numbers, or document the behaviour.
  - *Test.* `test_lot_risk_independent_of_other_lots_in_upload` **fails**.
- **R5-3: the live script cites a red row the app does not draw.**
  - *Problem.* DEMO_CASE step 3 says "Open the corner table under the verdict: the one red row is the scenario to discuss". The app (`app.py:502-505`) renders the corner table as a plain dataframe with no row highlighting, and there are now 3 losing rows, not one.
  - *Fix.* Either highlight losing rows in the app or change the script, for example "the rows with a positive deaths or cost difference".

### Tests
`test_round2.py` subset (corner, duplicate-key, worst-case, new R5 tests): 3 passed, 2 failed (the R5-1 and R5-2 tests). I did not re-run the rest of the suite this round; the builder reports 72 passed before these doc edits, and the R4 fixes did not touch model code other than `_lot_keys`.

### Verdict (round 5)
**No remaining MAJOR issues.** The remaining items are MINOR: R5-1 (one sentence that overgeneralises, in three documents), R5-2 (rare batch dependence of the imputation key) and R5-3 (a demo-script line). R5-1 and R5-3 should be fixed before the live demo, and each takes a few minutes.

### Builder response to Round 5 (all MINOR; critic verdict was "no remaining MAJOR issues")
- **R5-1:** the death claim now reads "deaths rise when extra riding does not lower case fatality *and* low-stress receiving does little" (brief, report headline 6, overview, demo case). DEMO_CASE lists all four no-riding corners, including the counter-example (metaphylaxis 25%, receiving 25%: 4.6 *fewer* deaths).
- **R5-2:** `_lot_keys` now always builds the key from the lot's own fields (categorical fields, head count, arrival weight, missing-data pattern), so a lot's risk no longer depends on which other lots are in the upload. Demo numbers and the validation run were regenerated with this code (validation re-run 2026-11-14); every quoted figure was updated.
- **R5-3:** the corner table highlights every losing row (positive cost or deaths difference); the demo script refers to the highlighted rows.
- Validation re-run: worst-cost and worst-health scenario now coincide (S7: −$4,863/week, +4.67 deaths/week); report and app say so in one sentence instead of repeating it.
