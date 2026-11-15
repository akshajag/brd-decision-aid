# Implementation Plan — BRD Arrival Decision Aid

Deadline: **2026-11-15 23:00 CST**. Optimize for a demo that **always runs end to end**. Every feature must trace back to a line in the brief (see `docs/EVIDENCE_INVENTORY.md` §1).

## Design decisions (defend these in the brief)
| Decision | Choice | Why |
|---|---|---|
| Decision unit | **Arrival cohort (pen lot)** | Metaphylaxis and processing decisions are made per group. Rojas 2022 models cohorts. Group records are what the yard actually has at arrival. |
| User | Yard manager, with the consulting veterinarian as reviewer | The brief names managers. Any drug decision needs vet sign-off. |
| Decision times | **Day 0** (arrival and processing) and **Days 1–14** (monitoring intensity, pulls), with a look-ahead to day 45 | Matches the 45-day outcome window and the information actually available at each point. |
| Current operating approach (baseline) | ASM, **editable**: weight-rule metaphylaxis (under 700 lb) by default, or auction lots under the cut-off | NAHMS 2019 reports by weight class but only 15.6% of cattle group-treated, so the default is generous and the stricter rule is offered. |
| Data | **Synthetic, literature-anchored demo data** for 4 yards, with missing values injected | No dataset is provided. Must show operating differences, missing information and binding constraints. |
| Risk model | Transparent logistic score on arrival covariates, with **uncertainty intervals** | Explainable for each lot. Rojas shows logistic regression (AUC 0.785) is about as good as a random forest. |
| Optimizer | Simple, deterministic allocation of interventions to lots under capacity limits (greedy, or a small integer program with a fallback) | Reliable and explainable. Fast enough to rerun live when a slider moves. |
| Stack | **Python + Streamlit**, pandas, scikit-learn, numpy, plotly, pytest, Playwright for UI tests | Free, one-command launch, built-in upload and sliders, quick to build. |

## Priorities

### P0 — Minimum to submit (must all work by Nov 15, ~12:00 CST)
1. **Repo skeleton** and `requirements.txt` (pinned), `README.md` launch instructions (`streamlit run app/app.py`).
2. **Synthetic data generator** `src/data/generate.py` (seeded): 4 yards with different arrival profiles (weight, source/commingling, transit distance, season, pen density). Ground-truth outcomes come from a documented data-generating process. Missing values are injected and missingness flags kept. Output: `data/demo_cohorts.csv` + `data/DATA_DICTIONARY.md`, with every variable tagged OBS/EST/ASM and given a source.
3. **Risk module** `src/model/risk.py`: logistic score → P(≥15% morbidity), plus expected morbidity %. Bootstrap or parameter-uncertainty intervals. Per-lot driver contributions. **Low-confidence flag** when key fields are missing or the interval crosses the decision threshold.
4. **Consequence model** `src/decision/consequences.py`: for each lot × option → expected cases, deaths, treatment cost, net-return change and antimicrobial doses (AMU), with ranges. Every effect size lives in one `config/assumptions.yaml` with a source, label and range.
5. **Options**: (a) baseline weight-rule metaphylaxis; (b) risk-targeted metaphylaxis; (c) no metaphylaxis + enhanced pen riding; (d) risk-targeted metaphylaxis + enhanced pen riding for medium-risk lots. Effect sizes are **ASM sliders**, never presented as validated.
6. **Constraint-aware allocation** `src/decision/optimize.py`: daily pen-riding labour hours, hospital-pen head capacity, chute/processing slots. Report which constraints are binding and which lots are left under-served. **Hard rule: no option may leave a lot below its minimum-care floor.**
7. **Dashboard** `app/app.py` with five tabs, one for each manager capability in the brief: Data (upload, edit, inspect) → Risk (ranked lots, drivers, uncertainty) → Options vs baseline (trade-off table and chart: health, cost, AMU) → What-if (assumption and constraint sliders, live recompute) → Decision summary (downloadable Markdown/PDF with a "Requires veterinary review" block and the low-confidence warnings).
8. **Demonstration case**: one fixed scenario, e.g. Yard 2 in the fall, a 3-lot arrival, hospital capacity binding. Scripted in `docs/DEMO_CASE.md` and reproducible from a seed.

### P1 — Credibility (Nov 15 afternoon)
9. Model evaluation on held-out synthetic data: AUC, Brier score, a calibration plot. Show it **next to the published 0.68–0.79 benchmark**, with the explicit caveat that synthetic validation is circular.
10. Sensitivity analysis: a tornado chart showing which assumptions flip the recommendation.
11. Playwright smoke test (webapp-testing skill): app loads, demo data appears, a slider changes the result, the summary downloads.
12. Technical brief, 3 pages maximum (`docs/TECHNICAL_BRIEF.md` → PDF via the pdf skill), plus an AI and external-resource disclosure.

### P2 — Only if P0 and P1 are green
13. ~~Individual-animal pull-list view~~ → built as a **lot-level days 1–14 check-in** (pulls to date update risk; hospital outlook; monitoring actions). An animal-level view would need animal records we do not have.
14. Weather input per yard: built as a **manual forecast override** (no external data source, so nothing to license).
15. Scenario save and load: **built** (JSON). Visual polish: small fixes only.

## Agent workflow
| Phase | Lead agent | Gate |
|---|---|---|
| Lock assumptions table and sources | `brd-scientific-researcher` | Every parameter has a source and an OBS/EST/ASM tag |
| Generator, risk model, evaluation | `brd-data-scientist` | Pipeline runs from the seed. Metrics are honest. |
| Options, consequences, constraints | `brd-decision-optimizer` | Baseline comparison works, binding constraints are surfaced, care floor holds |
| Review before submission (run twice: after P0 and before submitting) | `hackathon-judge` | No unresolved "blocker" findings |

## Out of scope (deliberately)
Real-time sensors, deep learning, multi-page auth, cloud deploy, drug-specific dosing advice.

## Known risks
- The judges will ask how this differs from fitting to our own assumptions. Answer: the tool is a **transparent decision framework**. Its value is in structured trade-offs and handling constraints. Swap real data in via the upload path, and the same pipeline recalibrates.
- A Streamlit cold start during the live demo. Mitigation: a pre-warmed local run and a short recorded backup.
