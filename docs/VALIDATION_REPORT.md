# Validation report (synthetic data)

Generated 2026-11-14 by `python -m scripts.run_validation` · generator config hash `e1c24b9bc2d314db`

> **Read this first.** No real data was provided. Every number below comes from data we simulated from our own assumptions. These checks show the pipeline is leakage-free, recovers a known process, fails in measured ways when that process is changed, and makes decisions with quantified regret. They are **not** evidence of real-world accuracy. On real feedlot data, Rojas et al. (2022) report cohort AUC 0.68-0.79 and PPV at most 25%.

## 1. Headline findings
1. **Prediction (temporal hold-out, 2026 arrivals).** AUC 0.82, Brier 0.134, calibration slope 1.08, calibration-in-the-large +0.3 pts. The full model beats every simple baseline on Brier score (weight-only rule 0.158, yard × source 0.153, full model 0.134).
2. **Honest transfer to a new yard (leave-one-yard-out).** AUC A 0.71, B 0.74, C 0.83, D 0.78. This is the number to quote for a yard the model has never seen.
3. **Uncertainty is close to nominal on average, not in every dataset.** Over 8 fresh histories, the 90% range covered the true expected risk for 90% of lots on average (range 82-96%). For the shipped dataset it was 88%: all lots share one fitted model, so a fit that is off by chance is off for everyone. Mean P(≥15%) was 0.37 against a true share of 0.36.
4. **Decisions (regret study, fresh weeks).** Against a clairvoyant oracle that knows each lot's true expected risk and the true effects, the recommended plan's regret averaged $4,951/week versus $21,855/week for the current approach. Under the simulated truth (which equals the planner's own assumptions, see caveat) the expected saving versus the current approach was $13,382/week (the tool estimated $13,516); it cost more in 0% of weeks; deaths changed by -2.94/week. The care floor itself costs $3,638/week relative to the oracle (the oracle ignores the floor and has unlimited hospital space). **Caveat: here the simulated truth equals the planner's own assumptions**, so this shows the risk model and optimizer work, not that the assumed effects are right; savings are expected values (lot-level chance is not added). When the effects are wrong the regret rises: S6 $11,490, S7 $13,163, S8 $22,786/week.
5. **Most adverse scenario tested.** Under 'Adverse effects corner: metaphylaxis 0.65 AND no benefit from extra riding/receiving', the expected saving falls to $-4,863/week, with a loss in 80% of weeks and +4.67 deaths/week versus the current approach. This is also the worst scenario for cattle health. **This is the headline risk:** the recommendation depends on assumed intervention effects that these data cannot validate.
6. **Does the verdict know when to be unsure?** The tool only says *robust* when the plan also wins at every end of the unvalidated effect ranges; otherwise it says *favourable if assumed effects hold* or *low confidence*. In the base regret study it said *robust* in 2 of 60 weeks (2 of them truly better); in the stress worlds (25 weeks each) it said *robust* in S6 0, S7 0, S8 0 weeks, but *favourable if assumed effects hold* is conditional: in S6, S7 and S8 the plan cost more in 44%, 80% and 72% of weeks while the tool said exactly that. In a typical week the plan loses at several corners of the declared effect ranges: **deaths rise when extra riding does not lower case fatality and low-stress receiving does little**, and **cost rises when metaphylaxis works better than assumed and the other effects are small**. Narrowing those two ranges (veterinary judgement, or a pen-level trial of extra riding) is what would make a robust call possible: in our simulations the decision was far more sensitive to these two effect sizes than to the risk-model misspecifications we tested (S1–S3).
7. **Drift alarm.** On the shipped history the recent 180-day check gives z = +1.00 (alarm off). On the drift scenario z = +4.45 (alarm ON; intercept shifted by +0.22 logit when on). The critic's probe found the alarm detects a drift of 0.35 logit/year in 10 of 10 histories but 0.15 logit/year in only 1 of 10, with about 1 false alarm in 30 stable histories; it corrects the level of risk only, not changes in individual risk factors.

## 2. Prediction
### 2.1 Temporal split (train before 2026-01-01, test after)
n_train 1022, n_test 378, observed high-risk prevalence 0.24. Sensitivity 0.37, specificity 0.95, PPV 0.70, NPV 0.83. PPV rises with prevalence: synthetic high-risk prevalence is 24% here versus 10.85% in Rojas 2022, whose real-data PPV was at most 25%, so 0.70 is not comparable. Lot-to-lot logit SD estimated out of fold: 1.28 (simulator: 1.30).

The label is *observed* morbidity ≥15% after any metaphylaxis, while the decision uses *untreated* risk. Section 3 therefore checks untreated risk directly against the known truth.

Concordance (CCC) 0.48 against a simulated ceiling of 0.49: the ceiling is what a perfect expected-risk model scores against noisy observed morbidity, so CCC alone cannot separate a good model from a perfect one.

### 2.2 Baselines (temporal split, observed high-risk label)
| model | brier | auc |
|---|---|---|
| B0 | 0.182 | 0.500 |
| B1 | 0.158 | 0.732 |
| B2 | 0.153 | 0.763 |
| full | 0.134 | 0.824 |

B0 overall prevalence · B1 arrival weight only (the information the current rule uses) · B2 yard × source prevalence.

### 2.3 Leave-one-yard-out
| yard | n_test | auc | brier | cal_in_the_large | cal_slope |
|---|---|---|---|---|---|
| A | 456 | 0.711 | 0.091 | -0.002 | 0.94 |
| B | 358 | 0.738 | 0.206 | -0.006 | 0.82 |
| C | 327 | 0.835 | 0.127 | +0.006 | 1.22 |
| D | 259 | 0.779 | 0.186 | +0.019 | 0.94 |

## 3. Recovery of the known truth (only possible with synthetic data)
### 3.1 Oracle recovery: predicted untreated risk vs the true E[p | complete covariates], 1,000 fresh lots
| band | n | bias | rmse |
|---|---|---|---|
| [0, 500) | 119 | -0.0031 | 0.0409 |
| [500, 600) | 273 | -0.0093 | 0.0419 |
| [600, 700) | 286 | -0.0096 | 0.0272 |
| [700, 800) | 224 | -0.0095 | 0.0195 |
| [800, 2000) | 98 | -0.0033 | 0.0107 |
| all | 1000 | -0.0081 | 0.0314 |

### 3.2 Positivity: how many untreated lots support the untreated-risk estimate
| band | lots | treated_share | untreated_lots |
|---|---|---|---|
| [700, 800) | 338 | 0.10 | 305 |
| [500, 600) | 338 | 0.88 | 40 |
| [0, 500) | 150 | 0.83 | 26 |
| [600, 700) | 379 | 0.86 | 54 |
| [800, 2000) | 195 | 0.07 | 181 |

Light lots were mostly treated, so their untreated risk rests on few untreated lots (confounding by indication). Their predictions lean on the adjustment for metaphylaxis.

### 3.3 Is the error systematic or chance? Independent histories from the same world
| history | lot_sd | coverage_90 | mean_p_high | true_share_high | bias_[0, 500) | bias_[500, 600) | bias_[600, 700) | bias_[700, 800) | bias_[800, 2000) |
|---|---|---|---|---|---|---|---|---|
| shipped CSV (seed 42) | 1.28 | 0.88 | 0.342 | 0.359 | -0.0088 | -0.0098 | -0.0147 | -0.0090 | -0.0037 |
| fresh seed 9000 | 1.34 | 0.96 | 0.370 | 0.359 | -0.0009 | +0.0054 | +0.0060 | +0.0056 | +0.0043 |
| fresh seed 9001 | 1.34 | 0.92 | 0.371 | 0.359 | +0.0132 | +0.0044 | +0.0043 | +0.0049 | +0.0050 |
| fresh seed 9002 | 1.31 | 0.91 | 0.366 | 0.359 | -0.0125 | -0.0000 | +0.0033 | +0.0064 | +0.0105 |
| fresh seed 9003 | 1.33 | 0.85 | 0.386 | 0.359 | +0.0129 | +0.0087 | +0.0082 | +0.0106 | +0.0081 |
| fresh seed 9004 | 1.29 | 0.82 | 0.369 | 0.359 | -0.0055 | +0.0057 | +0.0082 | +0.0028 | +0.0003 |
| fresh seed 9005 | 1.30 | 0.92 | 0.359 | 0.359 | -0.0151 | -0.0051 | -0.0036 | +0.0021 | +0.0059 |
| fresh seed 9006 | 1.32 | 0.89 | 0.341 | 0.359 | -0.0172 | -0.0169 | -0.0096 | -0.0040 | -0.0017 |
| fresh seed 9007 | 1.30 | 0.91 | 0.379 | 0.359 | +0.0023 | +0.0077 | +0.0064 | +0.0061 | +0.0066 |

Bias columns are predicted minus true expected untreated risk by arrival-weight band (lb). The shipped dataset's fit sits at the low end of what independent histories produce; averaged over histories the bias is small. That is the behaviour expected of an approximately unbiased estimator with real sampling error, which the 90% range is meant to express.

## 4. Decisions: regret against an oracle
Each fresh week (24 lots) is planned from estimated risk and assumed effects, then scored with the true expected risk and true effects. *oracle* = best plan under the truth, within the rider-hour limits, ignoring the care floor (so regret ≥ 0 for every policy). *oracle_floor* = the same with the care floor built from true risk. Floor violations = truly-high lots on standard care only.

| policy | realised_cost | cases | deaths | regimens | floor_violations | regret |
|---|---|---|---|---|---|---|
| current | $152,930 | 350.7 | 24.55 | 2613 | 1.37 | $21,855 |
| oracle | $137,434 | 400.5 | 24.15 | 1342 | 0.23 | $0 |
| oracle_floor | $137,974 | 342.3 | 20.97 | 1961 | 0.00 | $3,638 |
| recommended | $139,548 | 354.1 | 21.61 | 1909 | 0.00 | $4,951 |

Verdict calibration (critic D20): the tool said *robust* in 3% of weeks (2 of 60); in those weeks the plan was truly cheaper with no more deaths 100% of the time. In this base world, truth equals the planner's assumptions, so this is the favourable case. See section 5.

## 5. Stress tests (pipeline refit on each world; planner keeps its default assumptions; 25 weeks per world, so the base row differs from the 60-week study above)
| scenario | auc | cal_slope | mean_regret | estimated_saving | realised_saving | p_loss_week | d_deaths | floor_violations_per_week |
|---|---|---|---|---|---|---|---|---|
| base | 0.805 | 1.04 | $4,405 | $13,665 | $14,269 | 0.00 | -2.20 | 0.00 |
| S1_interactions | 0.828 | 1.03 | $4,915 | $14,490 | $15,095 | 0.00 | -3.51 | 0.00 |
| S2_heterogeneity | 0.725 | 0.99 | $3,587 | $12,109 | $12,089 | 0.00 | -2.71 | 0.00 |
| S3_mnar | 0.801 | 1.03 | $4,760 | $13,340 | $13,946 | 0.00 | -2.19 | 0.00 |
| drift | 0.835 | 1.06 | $2,400 | $13,739 | $20,408 | 0.00 | -10.65 | 0.00 |
| S4_weaker_meta | 0.806 | 1.04 | $15,560 | $14,284 | $23,272 | 0.00 | -4.39 | 0.00 |
| S5_stronger_meta | 0.803 | 1.04 | $3,270 | $13,825 | $5,814 | 0.04 | +0.62 | 0.00 |
| S6_no_enh_benefit | 0.805 | 1.04 | $11,490 | $13,665 | $1,300 | 0.44 | +2.77 | 0.00 |
| S7_adverse | 0.803 | 1.04 | $13,163 | $13,825 | $-4,863 | 0.80 | +4.67 | 0.00 |
| S8_combined | 0.764 | 1.04 | $22,786 | $15,216 | $-4,819 | 0.72 | -2.36 | 0.00 |

- **base**: Documented process (truth = planner's assumptions)
- **S1_interactions**: Unmodelled sale-barn x high-weather interaction and curved transit effect
- **S2_heterogeneity**: More unrecorded lot-to-lot variation (logit SD 1.8 vs 1.3)
- **S3_mnar**: Transit time missing more often when the haul was long (not at random)
- **drift**: Risk rises over time (+0.35 logit per year): a temporal split must detect it
- **S4_weaker_meta**: True metaphylaxis effect 0.25 (planner assumes 0.45) - favours the plan
- **S5_stronger_meta**: True metaphylaxis effect 0.65 (planner assumes 0.45) - works against the plan
- **S6_no_enh_benefit**: Extra riding and low-stress receiving have NO true effect
- **S7_adverse**: Adverse effects corner: metaphylaxis 0.65 AND no benefit from extra riding/receiving
- **S8_combined**: Most adverse tested: S7 effects + more lot variation (SD 1.8) + drift (+0.35 logit/yr)

Drift check: with risk rising 0.35 logit/year, the temporal split shows calibration-in-the-large +2.2 pts (observed above predicted), so the temporal split *can* fail and does flag drift. In the tool, the drift check runs at every refit and recalibrates the intercept when it fires (headline 7).

## 6. What this does and does not show
- **Shows:** no leakage (tests in `tests/critic/test_leakage.py`); approximately unbiased recovery of a known process; calibrated P(≥15%); quantified regret; failure modes in both directions.
- **Does not show:** real-world accuracy, real intervention effects, or savings for these yards. The benefit of extra riding and low-stress receiving can never be validated on these data, because the history contains no such interventions.
