# Likely judging rubric (inferred) and where the submission meets it

**No rubric was published.** The brief says "No dataset, no starter code, and no rubric are provided". The rubric below is **our inference**. It draws on:
- the brief's hard requirements (`docs/EVIDENCE_INVENTORY.md` §1);
- the brief's stated failure modes ("static or precomputed output is insufficient"; leadership rejects cost-shifting or leaving vulnerable cattle without care);
- the likely judges' fields. The shared-folder list includes researchers in BRD epidemiology and economics (a co-author of two of the supplied papers), animal-science modelling and model evaluation, and data science.

Treat the weights as a planning aid, not as fact.

| # | Criterion (likely weight) | What judges will look for | Where we show it |
|---|---|---|---|
| 1 | **Problem framing and decision relevance (15%)** | Clear user, decision unit, decision times and 45-day window. Risk → Action → Consequences → Feasibility connected. | Brief §1; app "Start here" four-question strip; `docs/DESIGN.md` §1 estimands |
| 2 | **Scientific credibility of evidence and assumptions (20%)** | Every number sourced and tagged as observed, estimated or assumed. No invented data. Effects treated as assumptions. Literature used appropriately. | `config/assumptions.yaml` (value, range, tag, source); `docs/SOURCES.md`; `docs/LITERATURE_REVIEW.md`; Model & evidence tab parameter table |
| 3 | **Analytical rigour and validation (20%)** | Leakage-free pipeline; honest validation (temporal, by yard); calibration; uncertainty; misspecification; decision-level validation; synthetic circularity acknowledged. | `docs/DESIGN.md` §3–5; `docs/VALIDATION_REPORT.md`; `docs/CRITIC_LOG.md`; tests in `tests/` and `tests/critic/` |
| 4 | **Working, interactive dashboard (15%)** | The 5 manager capabilities; live recompute; flags low confidence; vet caveat; runs free with one command. | `app/app.py` (8 tabs); README launch steps; Playwright test |
| 5 | **Consequences and feasibility realism (10%)** | Health, cost and antimicrobial use reported together; capacity limits bind; shortfalls shown; care floor; no cost-shifting. | Plan tab (comparison, waterfall, constraints); days 1–14 hospital outlook |
| 6 | **Uncertainty and "can't decide" honesty (10%)** | Ranges, probabilities, a stated verdict rule, and sensitivity analysis showing what flips the answer. | Three-state verdict (*robust* / *favourable if assumed effects hold* / *low confidence*) with a corner check of every unvalidated effect range; tornado charts; the demo week is honestly *favourable if assumed effects hold*, and the auction-only baseline is *low confidence* |
| 7 | **Communication and presentability (10%)** | A judge understands the approach, solution and key findings in 2 minutes. Plain-language explanations; a clean brief; a reproducible demo case. | `SUBMISSION_OVERVIEW.md`/PDF at the ZIP root; Start-here tab; brief ≤3 pages; `docs/DEMO_CASE.md` |

## Scoring bands we expect judges to use
| Criterion | Weak (1-2) | Adequate (3) | Strong (4-5): our target |
|---|---|---|---|
| 1 Framing | Generic BRD predictor; no decision | Lot-level decision named | Decision unit, timing, options, baseline and 4 consequence dimensions linked in one flow |
| 2 Evidence | Unsourced numbers; effects stated as facts | Sources listed | Every number tagged EST/ASM with a range; numbers checked against full text; gaps named |
| 3 Rigour | Random split on own synthetic data, AUC only | Temporal split, calibration | Temporal + new-yard splits, baselines, truth recovery, leakage canaries, decision regret, stress worlds, adversarial critic log |
| 4 Dashboard | Static or pre-computed | Interactive charts | Edit a lot or a slider and the plan, verdict and summary recompute live; low-confidence flags; vet caveat |
| 5 Feasibility | Cost only | Cost + health | Health, cost, antimicrobials and capacity reported together; binding constraints and shortfalls shown; care floor enforced |
| 6 Uncertainty | Point estimates | Intervals | Probabilities, a pre-registered verdict, a corner check, and a tool that says "not sure" when it isn't |
| 7 Communication | Jargon, no story | Clear brief | 1-page overview, plain-language Start tab, glossary, a rehearsed 3-minute demo with one memorable example |

## Finalist round (5-minute talk + 3-minute live demo)
Likely questions and the answer to have ready:
- "Your data are synthetic, so what did you show?" That the pipeline is leakage-free, recovers a known truth, makes decisions with low regret, and **says when it cannot decide**. The Rojas benchmark (AUC 0.68-0.79, PPV ≤ 25%) is shown next to every synthetic number.
- "What if extra riding does nothing?" Show the corner table: the plan can cost more and add deaths, so the verdict is *favourable if assumed effects hold*, not *robust*. That is why the veterinarian confirms the effects.
- "Does it ever withhold care?" No. High and borderline lots never get standard care only. Clearly-high lots keep metaphylaxis unless a vet overrides. Shortfalls are escalated, never rationed.
- "What would you do with real data?" Refit on the yard's own lot records. The drift alarm and the recalibration-in-the-large handle season and buyer changes. Then run a pragmatic pen-randomised trial of extra riding to replace the biggest assumption.

## Red flags judges punish (and our status)
| Red flag | Status |
|---|---|
| Fabricated data or citations | None. Synthetic data is labelled everywhere; citations are checked against supplied PDFs. Four team-added citations are disclosed: two methods (Lin, Tedeschi) and two parameter sources (Baptiste and Kyvsgaard, Neal). |
| Claiming validated effect sizes | Not claimed. Effects are sliders with ranges; the verdict is only *robust* if the plan wins at every end of those ranges. |
| Validation that cannot fail (circular, random splits) | Temporal and leave-one-yard-out splits are primary; baselines; oracle recovery; stress tests; Rojas benchmark shown alongside. |
| Static dashboard | Live MILP plus Monte Carlo; every input recomputes. |
| Ignores vulnerable cattle or capacity | Hard care floor; rider hours are hard limits; shortfalls escalated, never hidden. |
| Medication advice without a vet | No drug names or doses; vet sign-off flags everywhere. |
| Over 3 pages | Body is 3 pages; references on page 4. |

## Self-assessment (honest)
We expect to score strongest on criteria 2, 3 and 6. The main residual weakness that cannot be fixed before the deadline is that **no real data exists**: every performance number is synthetic. That is why the decision-regret study and stress tests matter, and we say so up front.
