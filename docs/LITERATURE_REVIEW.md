# Literature screen: BRD arrival decision aid (2026-11-14)

Scope: evidence for the parameters in `config/assumptions.yaml` and for the methods used in the risk model and optimizer. This was a rapid, single-reviewer screen done in one session. It is **not** a systematic review: no protocol, no second screener, no risk-of-bias scoring. Every number below was seen in a fetched abstract, full text or bibliographic record. Items marked *(secondary)* were seen only as quoted by another paper. Items marked *(snippet)* were seen only in a search-engine summary and have not been verified.

## 0. Team verification and what we changed (added after the screen)
The screen was run by an AI research agent; the team then checked the numbers that feed the model against the source text.
- **Baptiste & Kyvsgaard 2017.** In the full text (PMC7108556) the pooled prophylaxis + metaphylaxis BRD morbidity RR is **0.49 (95% CI 0.45–0.53)**, or **0.52 (0.48–0.57)** after trim-and-fill, and the pooled mortality RR is **0.62 (0.54–0.72)**. The table below quotes a metaphylaxis-only subgroup (0.42; bias-adjusted 0.53) that we have **not** re-located in the text, so `config/assumptions.yaml` cites only the pooled figures we verified. Either way, our default RRR of 0.45 sits inside the published range.
- **Neal 2024** verified (abstract): BRD case fatality 6.3%, all-cause 10.0%, first-treatment success 67.8%, from 132,521 records in 14 yards. `case_fatality_rate` (0.07) is now tagged **EST** with this source.
- **Proposals we did not adopt, and why:**
  - *Separate metaphylaxis effect on case fatality.* In our model, deaths already fall in proportion to cases. Metaphylaxis therefore cuts deaths by 45% (RR 0.55), close to the pooled mortality RR of 0.62. Adding a second effect would double-count.
  - *Widening the metaphylaxis range to 0.15–0.70, retreat fraction 0.30, receiving effect 0.05.* These are reasonable. We kept the current values so the validated numbers in `docs/VALIDATION_REPORT.md` stay reproducible before the deadline. The stress tests already cover a true metaphylaxis effect of 0.25 and of 0.65, and a receiving/riding effect of zero.
- Items marked *(snippet)* or *(secondary)* below are unverified and are not used anywhere in the model or the brief.

## 1. Screening log (honest counts)

| Stage | Count | Notes |
|---|---|---|
| Records identified | ~530 displayed hits (plus several thousand more undisplayed database hits) | 22 web searches (~220 result links, many duplicates or non-article pages), 9 Europe PMC result pages (267 titles), 2 arXiv API queries (44 titles). Europe PMC hit totals were larger (e.g. 715, 500, 187), but only the displayed page was screened. |
| Titles screened | ~400 unique (approx.) | Duplicates were removed by eye, not formally. Two Europe PMC full-text queries were badly polluted (AMR, virology, human medicine) and were screened by title only. |
| Assessed (abstract or full text actually read) | 55 | 48 relevant records read at abstract level (including arXiv abstracts). Of these, 6 also had full text or tables read: Baptiste 2017, O'Connor 2020, Dennis 2020, Kopp 2024, Schupbach 2025 and the Ambs 2022 knowledge summary. Babcock 2013 was read via search summary only. 4 methods papers were checked on bibliographic metadata only. 3 off-topic records (2 ICU, 1 pig) were returned by an author query and excluded. |
| Included | 50 new + Rojas 2022 (supplied) | Excluded after assessment: Picault 2024 (software paper, no data), Eyango Tabi 2024 (French lung-ultrasound batches, not feedlot), plus the 3 off-topic records. Not counted: the MLA B.FLT.0225 report (PDF unreadable) and the AABP acclimation study of about 5,200 calves (PDF unreadable), both mentioned in snippets only. |

**Search strings** (web search unless noted). Metaphylaxis: "metaphylaxis bovine respiratory disease network meta-analysis morbidity mortality feedlot"; "Avra metaphylaxis meta-analysis"; "Abell O'Connor mixed treatment comparison metaphylaxis"; "Baptiste Kyvsgaard 2017 … 169 trials". Risk models: "feedlot cattle arrival BRD risk prediction model cohort AUC calibration"; "machine learning bovine respiratory disease prediction feedlot". Timing and outcomes: "temporal distribution of first BRD treatment days on feed"; "BRD case fatality risk … retreatment … first treatment success"; "earlier treatment BRD … treatment timing". Receiving: "delayed arrival processing vaccination … randomized"; "acclimation low-stress handling receiving calves BRD". Clustering: "BRD pen-level spatial clustering …"; "intraclass correlation BRD pen feedlot …". Decisions: "Lhermie economic optimization antimicrobial metaphylaxis"; "value of information decision analysis livestock disease". **Europe PMC** (title/abstract fields): TITLE("respiratory disease") × {predict*/classif*/machine learning; accelerometer/feeding behavio(u)r/ultrasonograph*/remote monitoring/early detection/DART/pen riding; preconditioning/commingl*/transport*/auction/sale barn/low-stress/delayed processing/receiving; economic*/cost*/net return/case fatality/retreatment/treatment failure/success; decision/optimi*/partial budget/threshold/stochastic/simulation; transmission/contagio*/SIR/SEIR/clustering/intraclass/contact structure}. **arXiv API**: all:"bovine respiratory" OR (cattle AND respiratory) OR (feedlot AND disease); abs:(cattle|livestock|calves) AND (disease|health) AND (prediction|detection|accelerometer).

**Institution flag.** The requested Harvard/Yale/Princeton/Stanford/MIT (HYPSM) filter found almost nothing. The only HYPSM-affiliated included paper is Van Boeckel et al. 2015 (Princeton), which is AMU context only. Feedlot BRD evidence comes overwhelmingly from Kansas State, Iowa State, Calgary, Saskatchewan, Sydney/Queensland, INRAE-Oniris and Cornell. arXiv holds almost no BRD work: 2 relevant items out of 44 titles.

**Conflict-of-interest flag.** Several trials are industry-funded or co-authored, including by the hackathon sponsor (Merck Animal Health: Schupbach 2025, Nickell 2021; Elanco: Neal 2024; Bayer/BI: Rogers 2016). We report this; we do not judge it.

## 2. Evidence table

Q1: "Q1" means we are reasonably confident the journal is first-quartile in veterinary or animal science (JAS, Prev Vet Med, Vet Res, Front Vet Sci, BMC Med, Med Decis Making, Biometrics, PNAS, Agric Syst). "?" means unverified. *Bovine Practitioner*, *Transl Anim Sci*, *Vet Sci* and AJVR are not Q1 or are unverified.

### (a) Arrival/cohort risk prediction: discrimination and calibration
| Study | Journal / Q1 / institution | Design, population | Key finding | Bears on |
|---|---|---|---|---|
| Rojas 2022 (supplied), doi:10.3390/pathogens11040442 | Pathogens / ? / K-State | 1,733 cohorts, 10 US yards | AUC 0.682–0.789; PPV ≤25% | Benchmark (already in use) |
| Babcock 2013, Can J Vet Res 77:33–44 *(abstract via search summary)* | ? / K-State | Arrival data plus daily counts, negative binomial | Accuracy 56%→69% from arrival to DOF 8 (64%→74% when risk code known); December lots only 28% correct | Supports a days 1–14 risk update; season matters |
| Amrine 2019, doi:10.1016/j.compag.2018.11.035 | Comput Electron Agric / ? / K-State | 141 lots, 618 purchase groups, 35,027 head; 14-DOF morbidity | Median 14-DOF morbidity 2.1%; sale-barn data added nothing over arrival data at 2–4% cut-offs | Arrival-only predictors are defensible |
| Heinen 2023, doi:10.2460/ajvr.23.05.0094 | AJVR / – / K-State | 96,382 BRD cases, 12 yards, 2016–21 | 14.2% of cases did not finish; median AUC 0.675 | Realistic ceiling for individual-animal models |
| Heinen 2024, doi:10.2460/ajvr.24.06.0169 | AJVR / – / K-State | Economics of mortality prediction at Tx1/Tx2 | AUC >0.7; positive net return only at specificity >90% | The decision threshold must follow the economics, not AUC |
| Babcock 2009, doi:10.2527/jas.2008-1201 | JAS / Q1 / K-State | 31,131 treated cattle | Effect of first-treatment week on net return depends on arrival weight class | Interaction of weight with timing |

### (b) Metaphylaxis efficacy
| Study | Journal / Q1 / institution | Design | Key finding | Bears on |
|---|---|---|---|---|
| Baptiste & Kyvsgaard 2017, doi:10.1093/femspd/ftx083 | Pathog Dis / ? / Danish Medicines Agency | SR+MA, 58 publications, 169 RCTs (47 metaphylaxis RCTs) | Team-verified (full text): pooled prophylaxis + metaphylaxis morbidity RR **0.49 (0.45–0.53)**, trim-and-fill **0.52 (0.48–0.57)**. (A screening agent's metaphylaxis-only subgroup figures, 0.42 / 0.53, could not be verified and are not used.) high heterogeneity; funnel asymmetry. Pooled mortality RR (prophylaxis + metaphylaxis) 0.62 (0.54–0.72), but 82% of metaphylaxis RCTs had zero control deaths | **metaphylaxis_morbidity_rrr** |
| O'Connor 2020, doi:10.1017/S1466252320000031 | Anim Health Res Rev / ? / Iowa State, Guelph, K-State, Sask. | NMA, 46 studies, 167 arms, BRD within 45 d | Injectables cut 45-d BRD risk "by 1.5-fold on average" vs placebo (≈RR 0.67, our conversion). Placebo log-odds prior mean −0.376 (≈41% baseline risk, our conversion) | Same; trial populations are high-risk |
| Abell 2017, doi:10.2527/jas.2016.1062 | JAS / Q1 / K-State | MTC, 29 studies, 37 trials, 8 drugs | Drugs fall into tiers for d1–60 morbidity; no tiers possible for mortality or retreatment (CrIs overlap). Tier ORs ≈0.16 and 0.62 *(secondary, via Dennis 2020)*; 0.3 and 0.6 *(secondary, via Kopp 2024)* | Effect varies by product class, so the range must be wide |
| Schupbach 2025, doi:10.1093/tas/txaf067 (full text) | Transl Anim Sci / – / Merck AH, K-State | RCT, 2,542 auction heifers (257 kg), 1 OK yard, 2021 | Metaphylaxis-to-all vs none: 60-d BRD morbidity **36.6% vs 42.5%** (≈14% RRR); closeout BRD mortality **3.3% vs 7.1%**; overall mortality 4.2% vs 8.7%. Targeted program used 25% less control antimicrobial (range 18–33%) but had 51.0% morbidity | Morbidity RRR can be small in very high-risk lots while deaths still halve |
| Nickell 2021, doi:10.1093/tas/txab081 | Transl Anim Sci / – / Merck AH, K-State | RCT at 4 sites, targeted vs blanket | Targeted groups used 11–43% less BRD antibiotic; morbidity lower than negative control at 3 of 4 sites | Risk-targeted metaphylaxis is plausible, though site-dependent |
| Kopp 2024, doi:10.3390/ani14101423 (full text) | Animals / ? / K-State | 12,785 non-metaphylaxis cohorts, 2.2 M head, 13 yards; 2018 prices | **72%** of cohorts would not benefit economically; median cohort morbidity **5.1% (IQR 2.5–10%)**; morbidity ORs 0.6 / 0.3 assumed | Our weight-rule baseline over-treats; our synthetic morbidity (~11%) is above the median |
| Dennis 2020, doi:10.1093/tas/txaa020 | Transl Anim Sci / – / K-State, UNL | Stochastic net-return simulation, 10 yards | Upper-tier metaphylaxis +$90–149/head vs none; mortality 0.94–1.57% vs 5.9–9.8% | High-risk economics; price-year dependent |

### (c) Early detection and enhanced monitoring
| Study | Journal / Q1 / institution | Design | Key finding | Bears on |
|---|---|---|---|---|
| Schupbach 2025 (above) | – | ADD technology vs pen riding, 18 pens per arm | Days to first Tx 14.0 vs 16.1 (P=0.03). 60-d BRD mortality 2.9% vs 4.4% (P=0.06). **BRD case fatality among treated 6.0% vs 6.0%** (P=0.98). BRD pen-deads 0.4% vs 1.7% (P=0.01). Fallouts 11.1% vs 15.2% | **enhanced_monitoring_cfr_rrr**: the benefit comes from fewer untreated deaths, not lower case fatality among treated cattle |
| White 2016, doi:10.1016/j.prevetmed.2016.01.027 | Prev Vet Med / Q1 / K-State | Bayesian latent class | Visual observation Se 64.5% (57.9–70.8), Sp 69.1%; remote monitoring Se 81.3%, Sp 92.9% | Pen riding misses about a third of cases |
| Timsit 2016, doi:10.1016/j.prevetmed.2016.11.006 | Prev Vet Med / Q1 / Calgary, McGill, Montréal | Bayesian latent-class MA, 7 studies | Clinical illness Se **0.27 (0.12–0.65)**, Sp 0.92 (0.72–0.98) vs lung lesions | Treatment-based morbidity under-counts true BRD; caveat for the case definition |
| Wolfger 2015a, doi:10.2527/jas.2013-8030 | JAS / Q1 / Calgary, AAFC | 213 steers | Feeding behaviour flagged BRD about 7 d before visual identification | Mechanism for early-detection benefit |
| Wolfger 2015b, doi:10.1016/j.prevetmed.2014.12.001 | Prev Vet Med / Q1 / Calgary | Cost model | Mortality RR 0.66 and retreatment RR 0.60 are **model inputs** taken from earlier literature, not results. System cost CAD 9.61/steer more than pen-checking; not cost-effective | Shows our 0.20 sits inside values others assumed; not evidence |

### (d) Receiving, preconditioning, transport and commingling
| Study | Journal / Q1 / institution | Design | Key finding | Bears on |
|---|---|---|---|---|
| Dewell 2019, doi:10.21423/bovine-vol53no1p71-80 | Bovine Pract / – / Iowa State | RCT, 136 calves, 4 pens | Acclimation/low-stress handling: **no morbidity difference (P=0.34)**; ADG 2.92 vs 2.70 lb/d | **receiving_protocol_morbidity_rrr**: null on morbidity |
| Ambs 2022, doi:10.18849/VE.V7I2.502 | Vet Evidence / – / Texas A&M | Knowledge summary of 2 RCTs | Delayed MLV vaccination may lower **retreatment** (Rogers: 37.05% vs 43.97% at closeout); first-treatment morbidity unchanged; evidence "weak" | Delayed processing acts on retreatment, not first pulls |
| Rogers 2016, doi:10.21423/bovine-vol50no2p154-164 | Bovine Pract / – / VRCS, K-State, industry | RCT, 5,179 auction heifers, 60 pens | Delayed vaccine reduced the share treated twice (P<0.05) | As above |
| Cheek 2026, doi:10.1093/tas/txag152 | Transl Anim Sci / – / Arkansas | MA of 56 studies (103 records), receiving cattle | Pooled lnRR 0.07 (P=0.09); heterogeneity Q=336; Egger P<0.0001; season and location are moderators. Sign convention not verified | Mitigation effects are heterogeneous and biased toward publication |
| Mijar 2023, doi:10.3389/fvets.2023.1137078 | Front Vet Sci / Q1 / Calgary | 500 calves, 5 pens of 100 | Morbidity 24% in all-preconditioned pens vs 50% in all-auction pens; auction calves 2.76× more likely to have BRD | Source/preconditioning is a **risk input** (pre-arrival) |
| Step 2008, doi:10.2527/jas.2008-0883 | JAS / Q1 / Oklahoma State | RCT, ranch vs market calves | Ranch-origin calves treated less than market calves (P<0.001); 45-d weaned calves treated less | Same |
| Cernicchiaro 2012, doi:10.2527/jas.2011-4599 | JAS / Q1 / K-State | 14,601 cohorts, 21 yards, 1997–2009 | Mean morbidity 4.9% (median 1.1%); distance associated with morbidity and mortality (P<0.05) | Transit term direction |
| Hay 2014 / 2016 / 2017, doi:10.1016/j.prevetmed.2014.07.001; .2016.01.013; .2017.03.001 | Prev Vet Med / Q1 / Queensland | 35,131 head, 170 cohorts, 14 Australian yards | Late mixing with ≥4 groups OR 3.7; ≥480 kg vs <400 kg OR 0.6 (0.5–0.7); summer vs spring OR 2.4 (1.4–3.8); PAF of shared pen water 0.70 | Directions for the DGP; different population (EST) |
| Wisnieski 2021, doi:10.2460/ajvr.82.8.644 | AJVR / – / K-State, LMU | 3,339 cohorts, 545,866 calves | Purchase-day weather associated with 60-d BRD mortality (qualitative) | Weather term |

### (e) Case fatality, retreatment and economics
| Study | Journal / Q1 / institution | Design | Key finding | Bears on |
|---|---|---|---|---|
| Neal 2024, doi:10.21423/bovine-vol58no2p1-8 | Bovine Pract / – / K-State, Elanco | 132,521 first treatments, 14 central-US yards, 2017–20 | **BRD case fatality 6.3%**, all-cause 10.0%, first-treatment success **67.8%**; treatment at DOF 0–20 had lower success (49–55%) | **case_fatality_rate**, **retreat_fraction** |
| Avra 2017, doi:10.2527/jas.2016.1254 | JAS / Q1 / K-State | 194,062 treated cattle, 10 yards, 2005–09 | **33.3%** treatment failure (retreated or died) | retreat_fraction |
| Smith 2023, doi:10.3390/vetsci10030204 | Vet Sci / – / K-State and yards | 25 yards, 301,721 first treatments, 19,332 BRD deaths | Crude BRD deaths per first treatment ≈ **6.4%** (our ratio; not all deaths were treated); 25% of deaths occur by day 3–4 after treatment; timing right-skewed and season-dependent | case_fatality_rate, case_onset_spread_days |
| Blakebrough-Hall 2022, doi:10.1093/jas/skab361 | JAS / Q1 / Sydney, UQ | 898 steers, Australia | Case fatality 13.2% (16/121) | Upper end of CFR range |
| Blakebrough-Hall 2020, doi:10.1093/jas/skaa005 | JAS / Q1 / Sydney, MLA | 898 steers | Morbidity 18%, BRD mortality 2.1%; treated ≥3× returned AUD 384.97 less | Retreatment cost |
| Cernicchiaro 2013, doi:10.2460/ajvr.74.2.300 | AJVR / – / K-State | 212,867 cattle, 2001–06 | Fall net return $39.41 never treated, $29.49 treated once, $16.56 twice, −$33.00 three or more times | Production loss rises with retreatment |
| Lhermie 2020, doi:10.1371/journal.pone.0239135 | PLoS One / ? / Cornell, Texas A&M, Toulouse | 100-head partial budget, high-risk cattle | Median loss $66/head without preventive/control antimicrobials | Stewardship trade-off context |
| Kaniyamattam 2021, doi:10.3389/fvets.2021.606810 | Front Vet Sci / Q1 / Cornell | Sector LP model | Antibiotic-free production costs +$367 M (0.90%) | Sector-level; **not** a farm input (same caveat as Johnson 2017) |

### (f) Within-pen transmission and clustering
| Study | Journal / Q1 / institution | Design | Key finding | Bears on |
|---|---|---|---|---|
| Picault 2022, doi:10.1186/s13567-022-01094-1 | Vet Res / Q1 / INRAE-Oniris, K-State | Mechanistic stochastic model, 12 scenarios | Trade-off between BRD duration and antimicrobial doses; treatment triggered by detected cases performs better (no numbers in abstract) | Supports a metaphylaxis/AMU trade-off frame |
| Sorin-Dupont 2023, doi:10.1016/j.prevetmed.2023.106009 | Prev Vet Med / Q1 / INRAE-Oniris | Multi-batch model | Splitting high- and low-risk animals reduced incidence, peaks and AMU (qualitative) | Pen allocation option (P2) |
| Sorin-Dupont 2024, arXiv:2408.16269 | preprint / INRAE-Oniris | 16 scenarios | A collective-treatment trigger based on speed of spread reduced incidence and AMU (qualitative) | Days 1–14 escalation logic |
| Merca 2025, doi:10.1186/s13567-025-01611-y | Vet Res / Q1 / Copenhagen, INRAE | DGLM plus mechanistic model, 48 synthetic scenarios | DGLM-triggered collective treatment gave lower or equal AMU | Sequential updating method |
| Snyder 2026, doi:10.1016/j.vetmic.2026.110955 | Vet Microbiol / Q1? / Saskatchewan | 400 steers, 4 pens, WGS | Contagious within-pen spread of resistant *M. haemolytica* clones | Clustering exists; no ICC reported |
| Sanderson 2008, Can Vet J 49:373 (PMC2275341) | ? / K-State, USDA | 122 pens, weekly counts | 0–27.7 cases/100 animal-weeks; incidence peaks in week 1 | Onset spread; between-pen variation |
| Johnson 2023, doi:10.3390/vetsci10020089 | Vet Sci / – / K-State | Cluster analysis of temporal curves | 6 clusters for high-morbidity cohorts (≥15%) and 7 for low | Epidemic curve shapes vary widely |

No located study reports a within-pen or within-lot ICC for BRD (searched explicitly).

### (g) ML for BRD and model-evaluation methods
| Study | Journal / Q1 / institution | Key point | Bears on |
|---|---|---|---|
| Narayanan 2020, arXiv:2007.12791 | preprint | Pathogen-signature classification, up to 89.7% accuracy on simulated data | Not arrival risk; shows the arXiv gap |
| Tedeschi 2006, doi:10.1016/j.agsy.2005.11.004 (metadata verified) | Agric Syst / Q1 / Texas A&M | MSEP decomposition, CCC and model-adequacy statistics | Model & evidence tab (already cited) |
| Lin 1989, doi:10.2307/2532051 (verified) | Biometrics / Q1 | Concordance correlation coefficient | Already cited; DOI confirmed |
| Van Calster 2019, doi:10.1186/s12916-019-1466-7 (metadata) | BMC Med / Q1 / KU Leuven, Leiden (STRATOS) | Calibration hierarchy (mean, weak, moderate) | Report calibration-in-the-large and slope, not only AUC |
| Vickers & Elkin 2006, doi:10.1177/0272989X06295361 (metadata) | Med Decis Making / Q1 / MSKCC | Decision-curve analysis (net benefit across thresholds) | Pairs naturally with the 15% threshold and Heinen 2024 |

### (h) Decision analysis under uncertainty
| Study | Journal / Q1 / institution | Key point | Bears on |
|---|---|---|---|
| Theurer 2015, doi:10.2527/jas.2014-8487 | JAS / Q1 / K-State | Monte Carlo value of diagnostic Se/Sp, at prevalence <15% vs ≥15% | Same threshold; value of better detection |
| Kristensen 2015, doi:10.1016/j.prevetmed.2014.11.019 | Prev Vet Med / Q1 / Copenhagen | Review: from biological models to economic optimisation (MDPs, decision graphs) | Framing for the MILP plus Monte Carlo |
| Van Boeckel 2015, doi:10.1073/pnas.1503141112 | PNAS / Q1 / **Princeton (HYPSM)** | Global livestock AMU projected +67% from 2010 to 2030; cattle ~45 mg/kg | Context for the AMU dimension only |
| Kopp 2024, Dennis 2020, Lhermie 2020, Heinen 2024 (above) | – | Cohort-level economic decision rules | Optimizer economics |

## 3. Implications for our parameters (proposals only; nothing applied)

| Parameter (current) | What the literature says | Proposal |
|---|---|---|
| **metaphylaxis_morbidity_rrr** 0.45 [0.25, 0.65], EST | Pooled RCT RR 0.42 (0.35–0.49), bias-adjusted 0.53 (0.43–0.64), i.e. RRR 0.36–0.65 (Baptiste 2017). NMA ≈1.5-fold reduction, RRR ≈0.33 (O'Connor 2020). Tier ORs 0.3–0.6 give RRR ≈0.32–0.67 at 15–30% baseline (our conversion, via Kopp 2024). One recent RCT in very high-risk heifers found only ≈14% at 60 d (Schupbach 2025). | **Keep default 0.45. Widen to [0.15, 0.70].** Replace the Rojas/Avra citation with Baptiste 2017 (primary) plus O'Connor 2020 and Abell 2017. Note: Rojas's "Avra et al." is probably Avra 2017 JAS, a treatment-failure paper, so the 0.5 looks like an assumption, not an Avra estimate (unverified; check Rojas ref. 35). Populations are trial cattle, mostly high-risk; EST. |
| *(new, optional)* metaphylaxis effect on case fatality | Schupbach 2025: deaths halved (7.1→3.3% BRD mortality) while morbidity fell only 14%. Pooled mortality RR 0.62 (0.54–0.72), mixed prophylaxis and metaphylaxis (Baptiste 2017). | Consider `metaphylaxis_cfr_rrr`, default **0.0**, range [0, 0.4], ASM, for **sensitivity only**. The current model (deaths = cases × CFR) may understate metaphylaxis's mortality benefit. |
| **enhanced_monitoring_cfr_rrr** 0.20 [0, 0.40], ASM | Technology vs pen riding: case fatality among treated unchanged (6.0% vs 6.0%), but BRD deaths per case ≈6.8% vs ≈10.5% at 60 d (our ratio, P=0.06 for mortality), driven by fewer untreated pen-deads (Schupbach 2025). Visual Se 64.5% (White 2016). Wolfger 2015's RR 0.66 is an assumption. | **Keep value and range; keep ASM.** Re-define as "relative reduction in BRD deaths per case, including untreated pen-deads". Cite Schupbach 2025 as the closest evidence and state that it compares technology with pen riding, not extra pen riding. |
| **receiving_protocol_morbidity_rrr** 0.10 [0, 0.25], ASM | Acclimation: no morbidity effect (P=0.34, n=136; Dewell 2019). Delayed MLV: retreatment lower in one RCT, first-treatment morbidity unchanged, evidence weak (Ambs 2022). Receiving-mitigation MA is heterogeneous with publication bias (Cheek 2026). | **Lower default to 0.05, range [0, 0.20], keep ASM.** Optionally move part of the effect to retreatment (e.g. retreat reduction 0–0.15, ASM). Preconditioning and source effects (Mijar 2023, Step 2008) stay as **risk inputs**, not feedlot actions. |
| **case_fatality_rate** 0.07 [0.04, 0.12], ASM | BRD CFR 6.3% (all-cause 10.0%), 14 US yards 2017–20 (Neal 2024). Crude 6.4% (Smith 2023). 6.0% at 60 d and 8.7–9.7% at closeout in high-risk heifers (Schupbach 2025). 13.2% in Australia (Blakebrough-Hall 2022). | **Keep 0.07 [0.04, 0.12]; retag EST**, source Neal 2024 plus Smith 2023. Note: feeding-period CFR, central-US yards, transfers reasonably to TX/KS. |
| **retreat_fraction** 0.25 [0.10, 0.40], ASM | Failure (retreat or death) 33.3% (Avra 2017) and 32.2% (100 − 67.8%, Neal 2024). Minus ~6% deaths gives ≈26% retreated at least once. Retreatment 37–44% in high-risk heifers (Rogers 2016). Extra regimens per case ≥ share retreated, because some cattle get 3+ treatments. | **Default 0.30, range [0.15, 0.50]; retag EST** (Neal 2024, Avra 2017). |
| **pen_clustering_icc** 0.0 [0, 0.08], ASM | Contagious within-pen spread is documented (Snyder 2026; Picault 2022; Sanderson 2008). **No located study reports an ICC.** | No change. Write: "No supplied or located evidence. Treated as ASM, range 0–0.08, flagged for sensitivity analysis." |
| **case_onset_spread_days** 21 [14, 30], ASM | Mean DOF to first Tx 14–16 d (60-d window, high-risk; Schupbach 2025). Incidence peaks in week 1 (Sanderson 2008). Timing right-skewed and later for Q2 arrivals (Smith 2023). Australian peak 15–30 d *(snippet, MLA report not verified)*. | Keep. Cite Schupbach 2025, Sanderson 2008 and Smith 2023 for direction; stays ASM. |
| dgp intercept (mean synthetic morbidity ~11%) | Non-metaphylaxis cohorts: median 5.1% (IQR 2.5–10%) (Kopp 2024). Cohort mean 4.9% (Cernicchiaro 2012). 6.3% (Rojas 2022). | No change required. State in the brief that synthetic morbidity sits above recent US medians, which makes interventions look more valuable. |

**Confidence & gaps.** The strongest evidence covers metaphylaxis efficacy on morbidity (several meta-analyses, but heterogeneous and from trial populations), case fatality and treatment failure (very large US multi-yard datasets, though all from one research network and partly industry-co-authored), and modest arrival-model discrimination (AUC ≈0.68–0.79). The weak points are enhanced monitoring (one sponsor-co-authored RCT comparing technology with pen riding; nothing on *extra* pen riding), the receiving protocol (mostly null), and pen clustering (no ICC anywhere). The optimizer's recommendation is probably most sensitive to `metaphylaxis_morbidity_rrr` (where the plausible range is now wider), `enhanced_monitoring_cfr_rrr` (deaths flip sign below ~0.04), and the base morbidity level. Kopp 2024 suggests most real cohorts sit below our synthetic mean, and at those levels blanket metaphylaxis seldom pays. Numbers marked *(secondary)*, *(snippet)* or "our conversion" need checking against the originals before they go into the brief.
