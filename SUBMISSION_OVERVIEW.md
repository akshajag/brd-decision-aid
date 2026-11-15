# BRD Arrival Decision Aid: read this first

*AISFS × Merck Animal Health Hackathon 2026 · Problem 1 · a working, local, free decision aid for the first 45 days on feed*

> **All data in this submission are synthetic.** No dataset was provided. Every performance number shows that the pipeline works on data we generated from documented assumptions, not real-world accuracy. Intervention effects are assumptions with ranges, not results. Medication changes require veterinary review.

## 1. The problem in one paragraph
Each week a feedlot manager decides how to receive every arriving lot: standard pull-and-treat, arrival metaphylaxis, extra pen riding with low-stress receiving, or both. Today the decision is a weight rule (metaphylaxis under 700 lb). In our synthetic demo week that rule treats some low-risk lots, misses some risky heavy lots, and ignores labour and hospital limits. Leadership wants fewer sick and dead cattle and fewer antimicrobial doses, with no plan that simply shifts cost or leaves vulnerable cattle without care.

## 2. Approach: Risk → Action → Consequences → Feasibility
| Step | What we do | Why it is credible |
|---|---|---|
| **Risk** | Lot-level binomial logistic regression predicts each lot's 45-day BRD morbidity *if untreated*, with a 90% range (100 bootstrap refits × multiple imputation of missing records) and P(morbidity ≥ 15%). Literature-signed factors are sign-constrained, and a drift alarm recalibrates when recent lots stop matching | Logistic regression, as in Rojas et al. 2022 (real-data cohort AUC 0.68–0.79). Missing records widen the range and are never hidden. The lot-to-lot spread is estimated out of fold. |
| **Action** | A mixed-integer program picks one option per lot to minimise expected cost plus a stewardship weight per antimicrobial regimen | **Care floor:** high-risk and borderline lots never get standard care only. Rider hours are hard limits. Hospital or antimicrobial overload is reported as a shortfall to escalate, never silently rationed. |
| **Consequences** | Cases, deaths, regimens and cost for every plan, always shown together against the current approach | Every parameter lives in `config/assumptions.yaml` with value, range, source and an EST/ASM tag |
| **Feasibility + uncertainty** | Paired Monte Carlo over all assumption ranges plus a check of every corner of the unvalidated effect ranges gives a three-state, pre-registered verdict: *robust*, *favourable if assumed effects hold*, or *low confidence* | The tool says when it cannot decide, and which assumption would change the answer |
| **Days 1–14** | Pull counts update each lot's risk (Bayesian), flag lots tracking above plan, and project hospital load | Monitoring and staffing actions only; never drug advice |

## 3. The solution (what a judge can click)
`streamlit run app/app.py` opens an 8-tab dashboard: **★ Start here** (this week's answer in plain words), Lots & data (upload or edit), Risk priorities, Plan vs current approach, What-if & sensitivity, Decision summary (downloadable, with a veterinary sign-off block), Days 1–14 check-in, and Model & evidence. Every input recomputes live. A 3-minute script is in `docs/DEMO_CASE.md`.

## 4. Key findings
**Demo week** (24 lots, 3,406 head, default assumptions):

| Policy | Expected cases | Deaths | Antimicrobial regimens | Expected BRD cost |
|---|---:|---:|---:|---:|
| Current approach (metaphylaxis < 700 lb) | 353 | 24.7 | 3,343 | $166.9k |
| **Recommended plan** | 363 (+10.6) | 22.0* | **2,504 (−25%)** | **$154.2k (−$12.7k)** |

\*Deaths fall only if extra riding cuts case fatality by at least 4%, an unsourced team assumption (default 20%). The firmer results are 839 fewer regimens (100% of draws) and $12.7k lower expected cost (96% of draws), for about 11 more cases (10.6).

Under uncertainty, the plan is cheaper in 96% of draws, has no more deaths in 94%, and uses fewer regimens in 100%. **Verdict: favourable if assumed effects hold, not robust.** If extra riding does nothing for case fatality and low-stress receiving does little, the plan has more deaths (up to 3.9). If metaphylaxis also works better than assumed (65%), it costs $6.1k more too.

**Validation against a known truth** (`docs/VALIDATION_REPORT.md`; possible only because the data are synthetic):

1. **Prediction.**
    - Temporal hold-out AUC 0.82 with calibration slope 1.08.
    - A yard the model never saw gives AUC 0.71–0.83.
    - The model beats a weight-only rule on Brier score (0.134 vs 0.158).
    - The real-data benchmark is AUC 0.68–0.79.
2. **Uncertainty is close to nominal.** Across 8 independent synthetic histories, the 90% range covered the true expected risk for 90% of lots (82–96%).
3. **Decisions.** On fresh weeks scored with the true risk, weekly regret against a clairvoyant oracle was $5.0k for the plan versus $21.9k for the current rule. Truly high-risk lots left on standard care: **0** per week, versus 1.37 under the current rule. *Caveat:* there the truth equals our assumptions.
4. **Where it can go wrong.**
    - If the effects are wrong, regret rises to $11.5k–$22.8k per week.
    - In the worst health case (metaphylaxis 0.65, extra riding useless), deaths rise by 4.7 per week.
    - The tool called **none** of those stress weeks *robust* (and only 2 of 60 base weeks, both truly better). It said *favourable if assumed effects hold*, a conditional call: in those worlds the plan cost more in 44–80% of weeks.
5. **In our simulations the decision hinges on two effect sizes more than on the risk model.** Deaths hinge mainly on the extra-riding effect; cost rises only at joint corners where metaphylaxis works better than assumed and the other effects are small. The dashboard's *What would make it robust?* panel quantifies this: on its own, only extra riding flips the result (deaths rise below a 4% cut in case fatality), and narrowing all three effect ranges 65% toward their defaults turns "favourable" into "robust" (uniform narrowing, so an upper bound on what is needed). The tool's sensitivity points to a pen-level trial of extra riding as the next step.

## 5. How we kept ourselves honest
- **A design written first** (`docs/DESIGN.md`). An **adversarial critic** then reviewed design and code in rounds, wrote acceptance tests and probe scripts (`tests/critic/`, `scripts/critic/`), and every finding and response is logged in `docs/CRITIC_LOG.md`. We iterated until the critic reported no remaining major issue (round-by-round log in `docs/CRITIC_LOG.md`).
- **Leakage controls tested:** preprocessing is fitted on training rows only, there are no ID or date features, and the prediction path ignores any treatment column. A generator config hash freezes the synthetic world, and fresh evaluation seeds are separate from the tuning seed.
- **A literature screen** of about 400 titles, with numbers that feed the model checked against source text (`docs/LITERATURE_REVIEW.md`, `docs/SOURCES.md`).

## 6. What is in this ZIP
| Path | Contents |
|---|---|
| `SUBMISSION_OVERVIEW.md / .pdf` | This page |
| `docs/TECHNICAL_BRIEF.pdf` | Technical brief (3 pages + references) |
| `README.md` | Install, run and test steps |
| `app/`, `src/`, `config/` | Dashboard, model, optimizer, all assumptions |
| `data/` | Synthetic demo data, data dictionary, generator manifest |
| `docs/` | Design, validation report, critic log, demo case, literature review, sources, judging rubric |
| `reports/validation_results.json` | Machine-readable validation results |
| `tests/`, `scripts/` | Unit, app, browser and critic tests; validation runner |

**Run:** `pip install -r requirements.txt` then `streamlit run app/app.py`. **Test:** `pytest -q`. **Reproduce validation:** `python -m scripts.run_validation`.

*Built with AI assistance (Claude Code); the team verified the claims, citations and results. Disclosure in `README.md`.*
