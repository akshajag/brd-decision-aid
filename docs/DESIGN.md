# Design: BRD Arrival Decision Aid (data-science design review)

Status: **v2, implemented** (critic rounds logged in `docs/CRITIC_LOG.md`; results in `docs/VALIDATION_REPORT.md`). Changes from v1 are marked **[v2]**.

## 0. The one-paragraph version
We estimate, for each arriving lot, the 45-day BRD morbidity it would have **without** metaphylaxis. We then choose one of four management options per lot to minimise expected cost plus a stewardship penalty, within labour and hospital capacity and above a care floor. Every result is compared with the yard's current approach. No real data exists for this challenge, so the system is a **transparent decision framework demonstrated on synthetic data**. The design's job is to make sure that (a) nothing in the pipeline uses information unavailable at decision time; (b) validation numbers are not inflated by how the synthetic data was made; and (c) the decision layer is judged by the **outcomes of its decisions**, not only by how well the model predicts.

## 1. Estimands (what exactly we estimate)
| Quantity | Definition | Unit of analysis | Available when |
|---|---|---|---|
| p0(lot) | E[45-day first-treatment BRD morbidity \| arrival covariates, no metaphylaxis] | lot | day 0 |
| High risk | p0 ≥ 0.15 (Rojas 2022 cut-off, applied to expected risk) | lot | day 0 |
| Option effect | relative change in morbidity / case fatality under option k | lot × option | **assumed** (EST/ASM), not estimated |
| Policy value | E[cost + λ·regimens] under a plan, with health, antimicrobial-use (AMU) and feasibility reported separately | week of arrivals | day 0 |
| Updated risk | posterior p0 given pulls to date d | lot | days 1–14 |

Two kinds of uncertainty are kept apart:
- **Epistemic:** model refits, imputation of missing records, and uncertainty in effect and price parameters. This is the 90% range of *expected* risk.
- **Aleatoric:** lot-to-lot heterogeneity (resid_sd) and binomial noise. This is what P(≥15%) and outcome ranges include.

Mixing these up is a classic error; the UI labels them separately.

## 2. Data-generating process (synthetic) and its known risks
- **Structure.** Logit of p0 = linear predictor (weight linear up to 900 lb, then flat; hinge terms for shrink above 3% and crowding below 15 m²) + yard effect + an unobserved lot effect N(0, 1.3²). Cases are Binomial(head, p0·(1 − RRR·meta)). **[v2]** The simulator's TRUE effects live in `dgp:` (`true_metaphylaxis_rrr` etc.), separate from the planner's assumptions, so decisions can be scored under a truth that differs from what the planner assumes (stress tests, §5.3). Historical metaphylaxis is assigned by weight (85% of light lots, 10% of heavy lots), i.e. **confounding by indication**. Missingness depends on source, with sale-barn lots the worst.
- **Risk R1: circular validation.** Same family, same variables, so the model will look better than on real data. Mitigations:
  - (i) the fitted model is deliberately *not* the oracle form: it is linear in the raw variables, with no hinges, and metaphylaxis enters additively on the logit rather than multiplicatively on p. **[v2]** We tested the oracle-like alternatives (multiplicative treatment term by maximum likelihood, quadratic terms): they did not reduce bias, so the simpler model stays;
  - (ii) misspecification stress tests (§5.3);
  - (iii) every metric is always shown beside Rojas 2022 (real data).
- **Risk R2: oracle tuning.** The generator's parameters (intercept, lot SD) were tuned while watching the hold-out AUC. That is mild researcher leakage. Mitigations: freeze the generator (seed 42; **[v2]** config hash in `data/generator_manifest.json`, checked by a test); report metrics from **fresh seeds** listed in the manifest and never used during tuning; disclose the tuning.

## 3. Leakage audit (decision-time availability)
| Item | Risk | Control |
|---|---|---|
| Outcome columns in current arrivals | direct leakage | the generator writes no outcome columns; a test checks this |
| `metaphylaxis_given` as a feature | it is a treatment chosen at day 0, so it is allowed as an *adjustment* term; predictions set it to 0 | prediction path forces 0; test |
| Imputation medians, donor pools, standardisation | fitted on all data, they leak test information | fitted inside each training fold only (`fit()` receives train rows only); **[v2]** refit inside every bootstrap member too; test |
| `arrival_date`, `lot_id` | ID or time leakage | not features; quarter only; test |
| Random train/test split | lots from the same yard and period on both sides inflate performance | **temporal split** (train 2024–2025, test 2026) and **leave-one-yard-out** as primary; random split reported only for comparison |
| Hyperparameters chosen on the test set | optimistic bias | C = 1.0 fixed a priori; no tuning on test folds |
| Threshold for "predict high" (P ≥ 0.5) | chosen on test | fixed a priori, matching the decision rule |
| Days 1–14 update using day-45 information | look-ahead | the update uses only pulls up to day d and the arrival prior |

## 4. Model
- **Logistic regression with binomial weights** on lot case counts. Lot-level, interpretable, and as good as random forest in Rojas 2022 (AUC 0.785 vs 0.789). L2 penalty C = 1 on standardised features, missing-value indicators, one-hot categories (the penalty is weak relative to ~1,400 lots; removing it changes the metaphylaxis coefficient from −0.612 to −0.617).
- **Uncertainty [v3]:** an ensemble of 5 full-data members (different donor imputations of the training data; their average is the point estimate) plus 100 bootstrap members (the spread). Each member refits its own medians and scaler and fits on a donor-imputed copy of its training rows (donors = same yard and source), keeping the missing-value indicators. Prediction imputes missing fields from a random stream keyed on the lot's own fields, so a lot's risk does not depend on its row position or ID (critic R2-4). Each draw uses a **distinct** bootstrap member (R2-6). The draws are recentred on the logit scale so their mean equals the seed-stable point estimate (D11 and R2-7): the bootstrap gives the spread, the full-data fits give the centre. The 90% range is the 5th–95th percentile with Weibull plotting positions.
- **Sign constraints [v3]:** six factors have literature-fixed directions (lighter, longer haul, more sources, more shrink, less pen space and less bunk space can never lower risk). A coefficient that comes out with the wrong sign is set to zero and the member is refit (active set, R2-3). Bunk-space direction is a team assumption.
- **Drift alarm [v3]:** a model fitted before the last 180 days predicts that window; if observed minus expected cases exceeds 2.5 standard errors (lot-level residuals), the tool raises an alarm and recalibrates the intercept on the window (R2-8). Both cut-offs are in config.
- **Lot-to-lot spread [v2]:** a random-intercept logit-normal-binomial model, fitted by maximum likelihood **out of fold** (5 folds) and profiled on the fitted mean, with metaphylaxis acting multiplicatively on untreated risk. It gives SD 1.28 on the shipped data (simulator 1.30).
- **P(≥15%) [v2]:** the probability that the lot's own untreated morbidity is ≥15%. It integrates over the epistemic draws and the lot spread, with the logit-normal centred so its mean equals each draw. The days 1–14 prior uses exactly this distribution (stratified particles), so day 0 of the check-in reproduces the arrival table.
- **Confounding by indication:** we adjust for historical metaphylaxis and predict with it set to 0. **Positivity diagnostic:** treated share by weight band; untreated-equivalent predictions for light lots rest on the 15% of untreated light lots. We check recovery against the generator's true p0 by weight band (§5.2).

## 5. Validation protocol (pre-specified)
### 5.1 Predictive validity (synthetic)
- **Splits:** temporal (train arrivals < 2026-01-01, test ≥ 2026-01-01); leave-one-yard-out (4 folds); random stratified (for comparison only).
- **Metrics:**
  - discrimination: AUC for high risk;
  - probabilistic accuracy: Brier;
  - calibration-in-the-large (observed − predicted mean) and calibration slope (logit of observed on logit of predicted, ideal 1);
  - continuous adequacy: CCC and the MSEP split (Lin 1989; Tedeschi 2006).
- **Baselines the model must beat:**
  - B0: overall mean;
  - B1: weight-rule logistic (weight only);
  - B2: yard × source mean.

  If the full model does not beat B1 on Brier score in temporal validation, we say so.

### 5.2 Oracle recovery (only possible because the data is synthetic)
- RMSE and bias of predicted p0 against the generator's **true** p0 on held-out lots, overall and by weight band and source.
- This tests the confounding adjustment directly. An honest result may show bias for light lots, and we report it.

### 5.3 Misspecification stress tests
Regenerate history and evaluation data under alternative processes, refit the unchanged pipeline, and report AUC, calibration slope and decision regret (§5.4):
- **S1:** source × weather interaction and quadratic transit;
- **S2:** larger unobserved heterogeneity (lot SD 1.8);
- **S3:** missing-not-at-random: transit missing more often when it is long;
- **S4:** a smaller true metaphylaxis effect (RRR 0.25) than the planner assumes (0.45);
- **[v2] S5:** a larger true effect (0.65), which works *against* a plan that withdraws metaphylaxis;
- **[v2] S6:** no true benefit from extra riding or low-stress receiving;
- **[v2] S7 (worst case):** S5 and S6 together;
- **[v2] drift:** risk rising 0.35 logit per year, so the temporal split has something to detect.

S4–S7 are planner-assumption errors. They are the most important stresses, because effect sizes are assumed. The worst case is the headline (see `docs/VALIDATION_REPORT.md`).

### 5.4 Decision validation: policy value and regret (the primary result) [v2: implemented in `src/decision/policy_eval.py`]
Prediction quality is not the objective; decisions are.
1. Simulate N = 200 future weeks of arrivals from the generator, with true p0 and true effects known.
2. For each week, the planner chooses a plan from the model's *estimated* risk and *assumed* effects.
3. Score each plan with the **true** p0 and **true** effects.
4. Report, against the current approach and an **oracle planner** (true E[p|x], true effects, same rider-hour limits, **no care floor**, so regret ≥ 0 for every policy). **[v2]** A second oracle with the care floor built from true risk measures the cost of the floor itself. Floor violations are counted against true risk. (A naive oracle that re-uses the estimate-based floor with zero uncertainty forces metaphylaxis on every truly-high lot, so its "regret" is mostly the cost of the floor; see critic D16.)
   - realised cost, cases, deaths, regimens and care-floor violations;
   - **regret** = realised objective − oracle objective.

This is out-of-sample: separate seeds, never used for fitting or tuning. It also captures the **optimizer's curse**, because plans chosen on noisy estimates are evaluated on the truth.

### 5.5 Decision under uncertainty
- Paired Monte Carlo over epistemic parameters (200 draws).
- The verdict thresholds are pre-specified: P(cheaper) ≥ 0.70 and P(no more deaths) ≥ 0.80 (**[v2]** in `config/assumptions.yaml`, `decision:`). **[v3]** It has three states (critic R2-1). *Robust* also requires the plan to cost no more and cause no more deaths at every end (all 8 corners) of the three unvalidated intervention-effect ranges. *Favourable if assumed effects hold* means the Monte Carlo passes but a corner loses. *Low confidence* means the Monte Carlo fails. Only *robust* counts as confident; its calibration is checked in the regret study under each stress world.
- **[v3] Borderline floor:** a lot whose 90% range reaches 15% never gets standard care only (R2-13).
- Tornado charts for cost and deaths; re-optimisation sweep for metaphylaxis effect.

## 6. Decision layer
- **Formulation:** mixed-integer program (HiGHS):
  - minimise Σ cost + λ·regimens;
  - one option per lot;
  - hard rider-hour limit per yard;
  - soft hospital and AMU caps with reported shortfalls;
  - care floor (High never STD; High with risk_lo ≥ 15% keeps metaphylaxis unless the vet overrides; insufficient-data lots keep baseline protection).
- **Known limitations:**
  - Linear expected-value objective: risk-neutral, no CVaR.
  - Point estimate of p0 in the objective. Outcomes are linear in p, so the plug-in is exact when the point is the mean; **[v2]** we use the mean of the epistemic draws (v1 used the median, which differed by up to 2.7 percentage points); **[v3]** the draws are recentred on a multiply-imputed full-data estimate, so the mean is also stable across bootstrap seeds.
  - Hospital load uses a uniform-onset approximation.

## 7. What we will and will not claim
- **Will claim:**
  - the pipeline is leakage-free;
  - it recovers a known process under stated conditions;
  - it degrades in measured ways under misspecification;
  - its decisions have quantified regret against an oracle and a measured advantage over the current rule *in simulation*;
  - every assumption is visible and adjustable.
- **Will not claim:**
  - real-world accuracy;
  - real intervention effects;
  - savings for these yards.

## 8. Path to real data
1. Load the yard's historical lots in the same schema.
2. Temporal and leave-one-yard-out validation.
3. Recalibrate the intercept and slope per yard.
4. Replace assumed effects with veterinarian-confirmed or trial-based estimates (see `docs/LITERATURE_REVIEW.md`).
5. Monitor calibration drift monthly.
6. Re-run the regret study with the yard's own process fitted as the simulator.
