# BRD Arrival Decision Aid

A working decision aid that helps feedlot managers decide what to do with arriving cattle lots during the **first 45 days on feed**, to manage bovine respiratory disease (BRD). Built for the AISFS × Merck Animal Health Interdisciplinary Hackathon 2026.

**In plain English:** cattle often get pneumonia (BRD) in the first weeks after they arrive at a feedlot. Each week the manager must decide, for every group of new cattle, whether to give them a preventive antimicrobial (metaphylaxis), watch them more closely, or both, without running out of staff or hospital pens. This tool estimates each group's risk, recommends an action, shows what that would mean for sick cattle, deaths, antimicrobial use and money compared with today, and says plainly when the evidence is too weak to be sure. Open the app's **★ Start here** tab for a one-minute overview and a glossary.

It connects four things for every arriving lot:
- **Risk**: which lots warrant attention, why, and how sure we are.
- **Action**: standard care, arrival metaphylaxis, or enhanced monitoring with low-stress receiving.
- **Consequences**: health, cost and antimicrobial use, each compared with the current operating approach.
- **Feasibility**: rider hours, hospital-pen capacity, an optional antimicrobial cap and a reported processing-chute check, under a hard minimum-care floor. High-risk lots keep metaphylaxis unless a veterinarian overrides it.

The current operating approach that everything is compared against is editable. The default (metaphylaxis under 700 lb) is deliberately generous; NAHMS 2019 reports 15.6% of cattle group-treated. Scope: day-0 arrival decisions, plus a days 1–14 check-in that updates each lot's risk from BRD pulls recorded so far.

**Judges:** `SUBMISSION_CHECKLIST.md` maps each submission requirement to its file. `SUBMISSION_OVERVIEW.pdf` is the 2-page summary.

> **Demonstration build.** All lots are **synthetic**, because no dataset was provided. Intervention effects are **adjustable assumptions**, not validated results. **Medication-related recommendations require veterinary review.**

## Run it (free, local, about 2 minutes)
Requires **Python 3.12, 3.13 or 3.14**. On Windows with several Pythons installed, use `py -3.12 -m venv .venv`. If the exact pins fail to install, use `pip install -r requirements-ranges.txt` instead.
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.data.generate --seed 42   # optional: data/ already contains the seeded demo files
streamlit run app/app.py                # opens http://localhost:8501
```
Run the tests (unit tests plus a headless end-to-end run of the dashboard):
```bash
pytest -q
```
Optional real-browser smoke test (uses your installed Edge or Chrome; skipped if Playwright is missing):
```bash
pip install playwright==1.65.2
pytest -q tests/e2e
```
Re-run the full validation protocol (temporal and leave-one-yard-out splits, baselines, recovery of the known truth, decision regret against an oracle, stress tests; about 15 minutes on a laptop) and the critic's acceptance tests:
```bash
python -m scripts.run_validation        # writes reports/validation_results.json and docs/VALIDATION_REPORT.md
python scripts/build_brief.py            # docs/TECHNICAL_BRIEF.md -> PDF (needs Edge or Chrome)
python -m scripts.build_overview         # SUBMISSION_OVERVIEW.md -> PDF
python -m scripts.build_zip              # dist/BRD_Decision_Aid_submission.zip
pytest -q tests/critic                  # adversarial acceptance tests (about 10 minutes)
```
The first launch takes about 15 s while the risk model fits. The fitted model is then cached in `.cache/` (rebuilt automatically when the data, assumptions or model code change), so later launches start in a few seconds. Changing any input recomputes in a few seconds.

## Using the dashboard
0. **★ Start here**: this week's answer in plain language (what changes, what it means for health, antimicrobials and cost, how sure we are, what needs a person), the four-step logic, what to trust and what not to, and a glossary.
1. **Lots & data**: use the demo week or upload a CSV (a template is in the sidebar). You can edit cells directly. Missing records stay visible and widen uncertainty.
2. **Risk priorities**: expected 45-day morbidity without metaphylaxis, with a 90% range, main drivers, and a confidence label (`confident` / `borderline` / `insufficient data`).
3. **Plan vs current approach**: the recommended plan compared with the current weight-rule approach and two simple policies. Shows the trade-off chart, a partial-budget waterfall of where the money moves, lot-by-lot reasons, binding constraints, capacity shortfalls, and the probability the plan beats the current approach under uncertainty.
4. **What-if & sensitivity**: change assumptions and capacities in the sidebar. A tornado chart shows which inputs could flip the advantage, and the plan is re-optimised across metaphylaxis effect sizes.
5. **Decision summary**: a downloadable Markdown summary and lot plan CSV, with a veterinary review block.
6. **Days 1–14 check-in**: enter pulls to date per lot (or use the synthetic demo counts). Each lot's 45-day risk is updated, lots tracking above plan are flagged, the hospital pen outlook for the next 7 days is projected, and monitoring or staffing actions are suggested (never drug advice).
7. **Model & evidence**: temporal hold-out metrics next to the published benchmark, calibration slope and plot, model adequacy (concordance with its "perfect model" ceiling), the decision-validation results (regret against an oracle, leave-one-yard-out, baselines, stress tests with the worst case stated), and every parameter with its source tag.

Sidebar extras: a per-yard **weather forecast** override (manual input; no external weather feed) and **save / load scenario** (JSON of all sidebar settings, not the lot data).

The scripted walkthrough for the live demo is in `docs/DEMO_CASE.md`.

## Repository
```
app/app.py                    Streamlit dashboard
src/data/generate.py          seeded synthetic data generator
src/model/risk.py             risk model (bootstrap x multiple imputation), P(>=15%), drivers, confidence flag, temporal evaluation
src/model/validation.py       temporal / leave-one-yard-out splits, baselines, oracle recovery, positivity
src/model/stress.py           misspecification and assumption stress scenarios
src/decision/policy_eval.py   decision regret study against an oracle on fresh synthetic weeks
src/decision/followup.py      days 1-14 check-in: Bayesian risk update, hospital outlook, actions
src/decision/consequences.py  per-lot outcomes for each option
src/decision/optimize.py      constraint-aware allocation (MILP via scipy/HiGHS + greedy fallback)
src/decision/uncertainty.py   paired Monte Carlo comparison of plans
src/decision/summary.py       decision summary builder
config/assumptions.yaml       every parameter: value, range, unit, EST/ASM tag, source
data/                         demo CSVs + DATA_DICTIONARY.md
scripts/run_validation.py     full validation protocol -> reports/ and docs/VALIDATION_REPORT.md
docs/                         design, validation report, critic log, literature review, sources, demo case, brief
tests/                        unit tests, headless app test, Playwright browser test (tests/e2e),
                              critic acceptance tests (tests/critic)
```

## Data and evidence
- `data/historical_cohorts.csv` (1,400 lots with 45-day outcomes) and `data/current_arrivals.csv` (24 lots arriving this week) are **synthetic**. They are generated from documented assumptions, so they can be shared freely. Definitions and provenance are in `data/DATA_DICTIONARY.md`.
- Each parameter is tagged **EST** (published estimate from another population) or **ASM** (team assumption). None is **OBS** (observed for these yards). Sources are listed in `docs/SOURCES.md`.
- **Model metrics computed on synthetic data are circular.** They show the pipeline works, not real-world accuracy. The realistic benchmark is Rojas et al. 2022: cohort AUC 0.68–0.79.
- To use real data, upload lots with the same columns. To refit the model, replace `data/historical_cohorts.csv` with real records in the same schema.

## Disclosure
This project was built with the help of AI tools (Claude Code, Anthropic) for code, data-generation design, documentation and an internal critique pass. Open-source software: Python, Streamlit, pandas, NumPy, scikit-learn, SciPy and Plotly. The team is responsible for verifying every claim, citation and result. External resources are listed in `docs/SOURCES.md`.
