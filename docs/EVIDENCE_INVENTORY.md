# Evidence Inventory — what we were given, what it supports, what is missing

Audited 2026-11-14 from the challenge folder `Problem 1_ BRD Decision Aid/`.
Every number below is quoted from a supplied source. Tag each number used in code or the brief as **OBS** (observed in a dataset), **EST** (published estimate from another population), or **ASM** (our assumption). Nothing in this file is our own observation; at best it is EST.

## 1. Challenge documents

| File | Content |
|---|---|
| `Hackathon _BRD Decision Aid.docx` | Official brief (650 words). |
| `Copy of Hackathon _BRD Decision Aid.docx`, `Copy of Copy of ...docx` | **Same text** as the original. The larger file size comes only from embedded fonts. No extra content. |

No dataset, no starter code, and no rubric are provided. The brief says outright: *"No dataset will be provided."*

### Hard requirements, taken from the brief
- Decision window: **first 45 days after arrival**. The team picks the decision unit (individual or group), the intended user, the decisions supported and the information available when each decision is made.
- The tool must connect four things: **Risk** (who, and why), **Action** (what, and when), **Consequences** (health, cost, antimicrobial use), **Feasibility** (does it fit within constraints).
- Scope: animal, management, nutrition and environmental factors. The team justifies relevance.
- Operation: **4 yards, Texas Panhandle + western Kansas**. Records are incomplete. Staff time, pen space and treatment capacity are limited. Leadership rejects plans that only shift cost or that leave vulnerable cattle without care.
- The dashboard must let a manager: (1) enter or upload data and inspect cattle or groups; (2) see priorities **with basis and uncertainty**; (3) compare options **against a defined current operating approach**; (4) change inputs, assumptions or constraints and see the consequences; (5) produce a **decision summary**. It must show trade-offs and **flag when information cannot support a confident decision**. **Static or precomputed output is insufficient.** Medication recommendations stay **subject to veterinary review**.
- Data: document provenance, variable definitions, outcome definitions, access permissions and limitations. Separate observations, estimates and assumptions. The demo must show **meaningful operating differences, incomplete information and binding constraints**.
- Submission (deadline **2026-11-15 23:00 CST**): a working dashboard with launch instructions, free to run; shareable data, a data dictionary, generation code, analysis code and dependencies; a **technical brief of at most 3 pages**, excluding references; **one end-to-end demonstration case**; disclosure of external resources and AI tools. Finalists give a 5-minute talk, a 3-minute live demo and Q&A on 2026-11-18.

## 2. Supplied resources (`Resources_ BRD Decision Aid/`)

`US Antimicrobial Use.pdf` and `Copy of US Antimicrobial Use.pdf` are byte-identical. The `.docx` version has the same text.

### Rojas et al. 2022, *Pathogens* 11:442 — "Predicting BRD Risk in Feedlot Cattle in the First 45 Days Post Arrival" (CC BY)
The most directly relevant source.
- Data: 10 US feedlots, **1,733 cohorts, ~188,000 head**, proprietary and **not available**.
- Unit: **cohort**. Outcome: high risk = **≥15% first-treatment BRD morbidity within 45 days on feed (DOF)**.
- Predictors available at arrival: average arrival weight, cohort size, arrival quarter, sex (steer/heifer/mixed), pen area, bunk space, pen area per head and bunk space per head (measured from Google Earth). There was **no prior management or health history**. Cohorts with missing data were dropped.
- Mean 45-day morbidity was 6.30%: 27.07% in high-risk cohorts and 3.77% in low-risk cohorts. **High-risk prevalence was 10.85%.**
- Test-set AUC ranged from **0.682** (decision tree) to **0.789** (random forest). Logistic regression scored 0.785. PPV was only 10.9–25.0% and NPV 92.6–97.9%.
- Deterministic economics: cost of a false positive ≈ **$23.60/head** (unneeded metaphylaxis; this is the NAHMS 2011 treatment cost reused). A false negative costs ≈ **$151.18 lost value per treated animal**, which includes medicine costs and comes from 2000–01 Ranch to Rail data. **Metaphylaxis efficacy was set at 0.5 (50% morbidity reduction)**, citing Avra et al. (ref. 35); this is a modelling input, not a measured result. The best model gained only about **$5/head** over human classification, and the gain depended on prevalence.
- Limitation stated by the authors: metaphylaxis status was unknown, so outcomes may be confounded by treatment already given.
- **Use**: anchors the model's form, the predictors, the 15% cutoff, realistic discrimination and the cost of misclassification. **Do not** claim AUC above about 0.8 without real data.

### Weber, Schroeder, Renter, Pendell, Tonsor 2024 — AAEA poster, "BRD morbidity and net return distributions"
- Data: proprietary pen data (1,157 pens, 2014–15; 2,592 closeouts, 2001–06). Pen of 120 head, simulated 10,000 times.
- Mean net return per head **with no animals treated vs. all treated at least once**:
  - 550 lb: **+$25.80 vs −$221.03**
  - 700 lb: **+$8.96 vs −$169.95**
  - 850 lb: **+$5.11 vs −$151.78**
- Mean (SD) mortality %: 550 lb 2.01 (1.60), 700 lb 1.07 (1.42), 850 lb 0.80 (1.38).
- Cull proportions 0.0226 / 0.0079 / 0.0050. Interest 4%. Price assumptions: the poster text gives "purchase price of $1.7397 and a fed cattle price of $1.2628 a pound" for 550-lb calves. Purchase prices of 1.7397 / 1.5570 / 1.4558 $/lb appear to map to 550 / 700 / 850 lb. The PDF table layout is scrambled, so **verify against the original PDF before citing any price**.
- **Use**: economic consequences by placement weight. These are poster-level estimates under price conditions that have since changed. Treat them as EST and let the user adjust them.

### Johnson & Pendell 2017, *Front. Vet. Sci.* 4:189 — Market impacts of reducing BRD (CC BY)
- NAHMS 2011: **21.2%** of placed cattle affected by respiratory disease. BRD causes 45–55% of feedlot deaths. Direct treatment cost **$23.60 per case** (vs $12.59 in 1999).
- Literature ranges for treated cattle: average daily gain (ADG) down by up to 0.98 kg/day, hot carcass weight (HCW) down by up to 19.96 kg, marbling down by up to 28 points.
- Sector-level partial-equilibrium findings. These are **not** a farm-level decision input. Use only for context.

### USDA APHIS NAHMS 2019 — Antimicrobial Use and Stewardship on U.S. Feedlots, 2017 (public domain)
- 378 feedlots, data from 2016, collected before the FDA's Veterinary Feed Directive (VFD) changes took effect.
- **2016 mortality: 2.3% of cattle placed at <700 lb vs 1.2% at ≥700 lb.** 48.9% of cattle were placed at <700 lb.
- **15.6% of cattle were treated as a group** with an injectable antimicrobial in 2016. Our default weight-rule baseline (about 85% of head) is therefore generous, so the baseline is editable.
- Defines metaphylaxis (at least 90% of a pen treated on arrival or at risk). Prevalence of metaphylaxis use broken down by weight class, plus in-feed antimicrobial use (chlortetracycline, ionophores).
- **Use**: antimicrobial-use (AMU) accounting categories, a plausible current-practice baseline (weight-based metaphylaxis), and mortality benchmarks.

### Adekunle & Kaniyamattam 2026, *Agriculture* 16:311 — BRD epidemiology review (CC BY)
- Narrative review: pathogens, transmission and economics (losses above $2B per year). Metaphylaxis "substantially reduces" morbidity and mortality, **but no pooled effect size is reported in the text**.
- Preconditioning: a cited secondary claim of 50–60% lower morbidity and 30–35% lower mortality (ref. 87). This is about **pre-arrival** management, so the feedlot cannot control it after arrival. It can, however, be used as an arrival risk input.
- An SEIR-style framing with vaccination (V), quarantine (Q) and metaphylaxis (M) parameters. This is conceptual, with no fitted values.

### Adekunle et al. 2026, *Animals* 16:1726 — Scoping review of BRD risk-factor variability (CC BY)
- 91 studies, 23 risk factors grouped as biological, operational or environmental. 62% of studies are feedlot-based. Transport stress and commingling are the most-studied factors in stocker and feedlot systems.
- **The authors warn that frequency counts reflect research attention, not effect size.** Never turn "most studied" into "most important".
- Appendix Table A1 defines the high-risk state for each factor. Use it to justify which variables enter the risk score.

## 3. What is missing (and must be stated, not invented)
1. **No individual- or cohort-level dataset.** Any demo data will be **synthetic**, generated from the EST distributions above, and must be labelled that way everywhere.
2. **No validated effect sizes for interventions** in the supplied material. The only number is Rojas 2022's *modelling input* of 50% metaphylaxis efficacy (citing Avra et al.). There is nothing for delayed processing, extra pen riding, hospital-pen capacity or nutrition changes. These must be treated as **user-adjustable assumptions with ranges**, and tested with sensitivity analysis. They must never be presented as results.
3. **No yard-specific data** for the four yards: capacity, labour, pen inventory, weather. These are defined by us as ASM scenarios and must be justified.
4. **No external validation is possible.** A model fitted to synthetic data generated by our own assumptions only proves the pipeline works. It shows nothing about real predictive accuracy. Report the published AUC range (0.68–0.79) as the realistic benchmark.
5. Prices date from 2017–2024. Make them inputs.
6. Environmental and weather effects and nutrition effects are only qualitative in the supplied sources. External sources may be added, but each one must have its citation recorded in `docs/SOURCES.md`.
