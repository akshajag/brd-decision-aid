# Submission checklist

Where each required item is in this ZIP. All paths are relative to the `BRD_Decision_Aid/` folder.

| # | Requirement | Where it is | Notes |
|---|---|---|---|
| 1 | **Working dashboard** with launch instructions and no paid access | `app/app.py`, launched as described in `README.md` under "Run it" | Free and local: Python 3.12–3.14 plus open-source packages, with no account, API key or paid service. `streamlit run app/app.py` opens http://localhost:8501. It has 8 tabs: Start here, Lots, Risk, Plan, What-if, Summary, Days 1–14 check-in, and Model & evidence. |
| 2a | **Shareable data** | `data/historical_cohorts.csv` (1,400 lots with 45-day outcomes), `data/current_arrivals.csv` (24 arriving lots) | **Synthetic.** No dataset was provided with the challenge, so there is no restricted source data to redistribute. The demonstration data is fully shareable and labelled synthetic in the dashboard, the data dictionary and the brief. |
| 2b | **Data dictionary** | `data/DATA_DICTIONARY.md` | Every variable, with units, definitions and provenance. |
| 2c | **Preparation / generation code** | `src/data/generate.py` (`python -m src.data.generate --seed 42`), `data/generator_manifest.json` | Seeded and deterministic. The manifest records the seed and a hash of the generating configuration. |
| 2d | **Analysis code** | `src/model/risk.py` (risk model, intervals, drift check), `src/decision/` (consequences, optimizer, uncertainty, summary, days 1–14 follow-up), `scripts/run_validation.py` (validation protocol), `tests/` (unit, critic and end-to-end tests) | Every parameter, with its value, range, OBS/EST/ASM tag and source, is in `config/assumptions.yaml`. Validation outputs are in `reports/validation_results.json` and `docs/VALIDATION_REPORT.md`. |
| 2e | **Dependencies** | `requirements.txt` (exact pins), `requirements-ranges.txt` (compatible ranges) | `pytest -q` runs the test suite. |
| 2f | **Access instructions where data cannot be shared** | `README.md` under "Data and evidence"; brief §5, "Using real data" | Not needed for redistribution. To use real records, upload lots and historical outcomes in the same schema; the model refits and nothing else changes. |
| 3 | **Technical brief** (max. 3 pages excluding references) | `docs/TECHNICAL_BRIEF.pdf` (source: `docs/TECHNICAL_BRIEF.md`) | 3 body pages; references and disclosure are on page 4. Sections: 1 decision problem · 2 data and provenance · 3 methods (evidence and assumptions) · 4 evaluation · 5 trade-offs, assumptions and limitations. |
| 4 | **Demonstration case**: one end-to-end decision | `docs/DEMO_CASE.md` | Inputs: 24 lots, 3,406 head, 4 yards, capacities. Alternatives: current rule, recommended plan, metaphylaxis for all, no metaphylaxis. Recommendation: 17 lots change, with reasons and veterinary flags. Projected consequences: cases, deaths, antimicrobial regimens, cost and capacity, with uncertainty and the confidence verdict. It also includes a 3-minute live script. The dashboard's **Summary** tab downloads the same decision as Markdown. |
| — | **Disclosure** of external resources and AI tools | `README.md` under "Disclosure", `docs/SOURCES.md` (every source with its licence), the disclosure paragraph on page 4 of the brief | AI tool: Claude Code (Anthropic), used for code, synthetic-data design, documentation and an internal critique pass. External sources: the challenge-supplied papers plus four the team added (Lin 1989 and Tedeschi 2006 as methods; Baptiste and Kyvsgaard 2017 and Neal et al. 2024 as parameter sources). Software: Python, Streamlit, pandas, NumPy, scikit-learn, SciPy, Plotly. The team is responsible for verifying claims, citations and results. |

**Start here:** `SUBMISSION_OVERVIEW.pdf` is a 2-page summary of the approach, the solution and the key findings.

## Honesty notes for reviewers
- All metrics computed on the synthetic data show that the pipeline works. They are not real-world accuracy. The real-data benchmark is Rojas et al. 2022, with cohort AUC 0.68–0.79.
- Intervention effects are adjustable assumptions, not demonstrated results. The plan's death reduction holds only if extra riding cuts case fatality by at least 4%.
- Medication-related recommendations require veterinary review. No drug names or doses are given.
