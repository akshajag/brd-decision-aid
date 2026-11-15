# Sources

These keys are referenced in `config/assumptions.yaml`. Every source in the first table was supplied with the challenge. Add any new external source below, with its licence, before citing it.

| Key | Citation | Licence / access | Used for |
|---|---|---|---|
| Rojas2022 | Rojas HA, White BJ, Amrine DE, Larson RL. Predicting Bovine Respiratory Disease Risk in Feedlot Cattle in the First 45 Days Post Arrival. *Pathogens* 2022;11:442. doi:10.3390/pathogens11040442 | CC BY 4.0 | 15% high-risk definition; first-treatment case definition; arrival predictors; AUC benchmark 0.682–0.789; costs $23.60 (false-positive metaphylaxis, the NAHMS 2011 treatment cost reused) and $151.18 (lost value per treated animal, including medicine; from 2000–01 Ranch to Rail data); **metaphylaxis efficacy 0.5** (a modelling input citing Avra et al., ref. 35) |
| Weber2024 | Weber KMD, Schroeder TC, Renter DG, Pendell DL, Tonsor GT. Evaluation of BRD Morbidity in the Feedlot and its Effect on Net Return Distributions. AAEA Annual Meeting poster, 2024 | Verbatim non-commercial copying permitted with notice | Losses larger for light calves; mortality by placement weight; retreatment is common; mixed-sex lots at higher risk |
| Johnson2017 | Johnson KK, Pendell DL. Market Impacts of Reducing the Prevalence of BRD in US Beef Cattle Feedlots. *Front Vet Sci* 2017;4:189. doi:10.3389/fvets.2017.00189 | CC BY 4.0 | NAHMS 2011 figures: 21.2% of placed cattle affected; $23.60 per case; BRD 45–55% of deaths |
| NAHMS2011 | USDA APHIS NAHMS Feedlot 2011 (as cited in Johnson2017) | Public domain | Treatment cost per case |
| NAHMS2019 | USDA APHIS VS NAHMS. Antimicrobial Use and Stewardship on U.S. Feedlots, 2017. Report #751.0419, May 2019 | Public domain (US Government work) | 2016 mortality 2.3% (<700 lb) vs 1.2% (≥700 lb); **15.6% of cattle group-treated** (shows our default baseline is generous); metaphylaxis definition; <700/≥700 lb split used for the default baseline rule |
| Adekunle2026a | Adekunle A, Kaniyamattam K. Bovine Respiratory Disease: Epidemiological Drivers, Transmission Dynamics, and Economic Implications in Beef Production Systems. *Agriculture* 2026;16:311. doi:10.3390/agriculture16030311 | CC BY | Direction of metaphylaxis and early-detection effects (no pooled magnitude); risk-factor context |
| Adekunle2026b | Adekunle A, et al. Mapping Variability in BRD Risk Factors in Beef Production Systems: A Scoping Review. *Animals* 2026;16:1726. doi:10.3390/ani16111726 | CC BY | Which risk factors to include (transport, commingling, nutrition, environment). The authors warn that how often a factor is studied is not its effect size. |

## Added by the team: methods (no data or parameter values taken from them)
| Key | Citation | Licence / access | Used for |
|---|---|---|---|
| Lin1989 | Lin LI-K. A concordance correlation coefficient to evaluate reproducibility. *Biometrics* 1989;45:255-268. doi:10.2307/2532051 | Journal article; cited, not reproduced | Concordance correlation coefficient in the Model & evidence tab |
| Tedeschi2006 | Tedeschi LO. Assessment of the adequacy of mathematical models. *Agricultural Systems* 2006;89:225-247. doi:10.1016/j.agsy.2005.11.004 | Journal article; cited, not reproduced | Model-adequacy approach: mean squared error of prediction split into mean bias, slope bias and random error |

Both were first cited from the team's background knowledge.

Lin1989 and Tedeschi2006: DOIs confirmed on Crossref, 2026-11-14.

## Literature screen (2026-11-14)
These sources were found in a rapid literature screen (see `docs/LITERATURE_REVIEW.md`). Two have since been applied to `config/assumptions.yaml` after the team verified the figures: **Baptiste2017** (metaphylaxis_morbidity_rrr, checked against the full text) and **Neal2024** (case_fatality_rate and retreat_fraction, checked against the abstract). The rest support proposals only. Access was checked on the Europe PMC record or publisher page. "OA" means open access. Where the licence is "not stated", we cite the work and do not reproduce it.

| Key | Citation | Licence / access | Proposed use |
|---|---|---|---|
| Baptiste2017 | Baptiste KE, Kyvsgaard NC. Do antimicrobial mass medications work? SR and MA of RCTs of prophylaxis/metaphylaxis against BRD. *Pathog Dis* 2017;75(7):ftx083. doi:10.1093/femspd/ftx083 | PMC OA subset; © FEMS | metaphylaxis_morbidity_rrr: team-verified pooled prophylaxis + metaphylaxis RR 0.49 (0.45–0.53), trim-and-fill 0.52 (0.48–0.57), mortality RR 0.62 (0.54–0.72) |
| OConnor2020 | O'Connor AM et al. SR and NMA of injectable antibiotic options for BRD control in the first 45 days post arrival. *Anim Health Res Rev* 2019;20(2):163–181 (online 2020). doi:10.1017/S1466252320000031 | CC BY 4.0 | metaphylaxis range |
| Abell2017 | Abell KM, Theurer ME, Larson RL, White BJ, Apley M. MTC meta-analysis of metaphylaxis treatments for BRD. *J Anim Sci* 2017;95(2):626–635. doi:10.2527/jas.2016.1062 | Subscription; abstract only read | Tiering; ORs cited second-hand |
| Avra2017 | Avra TD et al. Retrospective analysis of risk factors for BRD treatment failure in feedlot cattle. *J Anim Sci* 2017;95(4):1521–1527. doi:10.2527/jas.2016.1254 | Subscription; abstract read | retreat_fraction (33.3% failure). Possibly Rojas2022 ref. 35, not verified |
| Neal2024 | Neal KB, White BJ, Amrine DE, Lubbers BV, Tessman RK, Larson RL. Risk factors for case fatality and treatment success following initial BRD treatment in feedyard cattle. *Bovine Pract* 2024;58(2):1–8. doi:10.21423/bovine-vol58no2p1-8 | Licence not stated | case_fatality_rate (6.3%), retreat_fraction (FTS 67.8%) |
| Smith2023 | Smith KJ et al. First treatment timing, fatal disease onset and days from first treatment to death in feedlot cattle. *Vet Sci* 2023;10(3):204. doi:10.3390/vetsci10030204 | CC BY | CFR cross-check; onset timing |
| Schupbach2025 | Schupbach BS, Davis MS, Jennings TD, Dixon AL, Renter DG, Nickell JS. Comparison of a novel BRD prediction technology and an automated disease detection technology to traditional methods in a U.S. feedlot. *Transl Anim Sci* 2025;9:txaf067. doi:10.1093/tas/txaf067 (PMC12161071) | CC BY; Merck AH co-authors | Monitoring and metaphylaxis effects; full text read |
| Nickell2021 | Nickell JS, Hutcheson JP, Renter DG, Amrine DA. Traditional BRD control vs targeted program from Whisper On Arrival risk predictions. *Transl Anim Sci* 2021;5(2):txab081. doi:10.1093/tas/txab081 | CC BY; Merck AH co-authors | Targeted metaphylaxis AMU reduction |
| Kopp2024 | Kopp DJ, Larson RL, Lancaster PA, White BJ, Smith KJ, Pendell DL. Economically optimum metaphylactic strategy for cattle cohorts of varied demographics. *Animals* 2024;14(10):1423. doi:10.3390/ani14101423 | CC BY | Base morbidity (median 5.1%); 72% of cohorts do not benefit; 2018 prices |
| Dennis2020 | Dennis EJ, Schroeder TC, Renter DG. Net return distributions when metaphylaxis is used to control BRD in high health-risk cattle. *Transl Anim Sci* 2020;4(2):1091–1102. doi:10.1093/tas/txaa020 | CC BY (PMC7201164) | Economic context |
| White2016 | White BJ et al. Bayesian evaluation of visual observation and remote monitoring to diagnose BRD. *Prev Vet Med* 2016;126:74–80. doi:10.1016/j.prevetmed.2016.01.027 | Subscription; abstract read | Detection sensitivity |
| Timsit2016 | Timsit E, Dendukuri N, Schiller I, Buczinski S. Diagnostic accuracy of clinical illness for BRD in feedlot beef cattle: SR and latent-class MA. *Prev Vet Med* 2016;135:67–73. doi:10.1016/j.prevetmed.2016.11.006 | Subscription; abstract read | Case-definition caveat (Se 0.27) |
| Wolfger2015a | Wolfger B et al. Feeding behavior as an early predictor of BRD in North American feedlot systems. *J Anim Sci* 2015;93(1):377–385. doi:10.2527/jas.2013-8030 | Subscription; abstract read | Early detection mechanism |
| Wolfger2015b | Wolfger B et al. Cost implications of an RFID feeding system for early BRD detection. *Prev Vet Med* 2015;118(4):285–292. doi:10.1016/j.prevetmed.2014.12.001 | Subscription; abstract read | Assumed effect sizes (inputs, not results) |
| Dewell2019 | Dewell RD et al. Acclimation and low-stress handling: BRD and performance. *Bovine Pract* 2019;53(1):71–80. doi:10.21423/bovine-vol53no1p71-80 | Licence not stated | receiving_protocol_morbidity_rrr (null) |
| Ambs2022 | Ambs A, Moberly HK, Capik S. Delayed vs on-arrival MLV vaccine in feedlot cattle. *Vet Evidence* 2022;7(2). doi:10.18849/VE.V7I2.502 | OA, author copyright | Receiving protocol |
| Rogers2016 | Rogers KC, Miles DG, Renter DG, Sears JE, Woodruff JL. Delayed respiratory viral vaccine and/or immunostimulant in auction-derived heifers. *Bovine Pract* 2016;50(2):154–164. doi:10.21423/bovine-vol50no2p154-164 | Licence not stated; industry co-authors | Retreatment |
| Cheek2026 | Cheek RA, Kegley EB, Howe SM, Goswami A, Powell JG. Seasonality of BRD and mitigation techniques in receiving cattle: a meta-analysis. *Transl Anim Sci* 2026;10:txag152. doi:10.1093/tas/txag152 | CC BY | Heterogeneity of receiving interventions |
| Mijar2023 | Mijar S et al. Commingling preconditioned and auction-derived calves: BRD morbidity, mortality, gain. *Front Vet Sci* 2023;10:1137078. doi:10.3389/fvets.2023.1137078 | CC BY | Source as a risk input |
| Step2008 | Step DL et al. Commingling calves from different sources and weaning protocols during a 42-d receiving period. *J Anim Sci* 2008;86(11):3146–3158. doi:10.2527/jas.2008-0883 | Subscription; abstract read | Source as a risk input |
| Cernicchiaro2012 | Cernicchiaro N et al. Distance traveled from sale barns to US feedlots and BRD morbidity/mortality. *J Anim Sci* 2012;90(6):1929–1939. doi:10.2527/jas.2011-4599 | Subscription; abstract read | Transit direction; base morbidity |
| Cernicchiaro2013 | Cernicchiaro N, White BJ, Renter DG, Babcock AH. Economic and performance outcomes by number of BRD treatments. *Am J Vet Res* 2013;74(2):300–309. doi:10.2460/ajvr.74.2.300 | Subscription; abstract read | Production loss by retreatment |
| Hay2014/2016/2017 | Hay KE et al. *Prev Vet Med* 2014;117:160–169 (doi:10.1016/j.prevetmed.2014.07.001); 2016;125:66–74 (doi:10.1016/j.prevetmed.2016.01.013); 2017;140:78–86 (doi:10.1016/j.prevetmed.2017.03.001) | Subscription; abstracts read | DGP directions (Australian, EST) |
| Sanderson2008 | Sanderson MW, Dargatz DA, Wagner BA. Risk factors for initial respiratory disease in US feedlots based on producer-collected daily morbidity counts. *Can Vet J* 2008;49(4):373–378. PMC2275341 | Free full text (PMC) | Onset timing; between-pen variation |
| Babcock2009 | Babcock AH et al. Feedlot health and performance effects associated with the timing of respiratory disease treatment. *J Anim Sci* 2009;87(1):314–327. doi:10.2527/jas.2008-1201 | Subscription; abstract read | Timing × weight |
| Babcock2013 | Babcock AH, White BJ, Renter DG, Dubnicka SR, Scott HM. Predicting cumulative risk of BRDC using feedlot arrival data and daily morbidity and mortality counts. *Can J Vet Res* 2013;77(1):33–44 | Abstract via search summary only; verify | Days 1–14 update |
| Amrine2019 | Amrine DE et al. Three classification models predicting BRD risk class within 14 DOF. *Comput Electron Agric* 2019;156:439–446. doi:10.1016/j.compag.2018.11.035 | Subscription; abstract read | Arrival-only predictors |
| Heinen2023 / Heinen2024 | Heinen L et al. *Am J Vet Res* 2023;84(10) (doi:10.2460/ajvr.23.05.0094); 2024;85(12) (doi:10.2460/ajvr.24.06.0169) | 2024: CC BY-NC; 2023: not stated | AUC benchmark; threshold economics |
| BlakebroughHall2020 / 2022 | Blakebrough-Hall C et al. *J Anim Sci* 2020;98(2):skaa005 (doi:10.1093/jas/skaa005); 2022;100(1):skab361 (doi:10.1093/jas/skab361) | Free full text via PMC/Europe PMC | CFR upper end; retreatment cost (Australia) |
| Lhermie2020 | Lhermie G et al. Economic effects of policy options restricting antimicrobial use for high-risk cattle in US feedlots. *PLoS One* 2020;15(9):e0239135. doi:10.1371/journal.pone.0239135 | CC BY | Stewardship context |
| Kaniyamattam2021 | Kaniyamattam K, Tauer LW, Gröhn YT. System economic costs of antibiotic use elimination in the US beef supply chain. *Front Vet Sci* 2021;8:606810. doi:10.3389/fvets.2021.606810 | CC BY | Sector-level context only |
| Picault2022 | Picault S, Ezanno P, Smith K, Amrine D, White B, Assié S. Modelling the effects of antimicrobial metaphylaxis and pen size on BRD. *Vet Res* 2022;53:77. doi:10.1186/s13567-022-01094-1 | CC BY | Transmission/AMU framing |
| SorinDupont2023 / 2024 | Sorin-Dupont B et al. *Prev Vet Med* 2023;219:106009 (doi:10.1016/j.prevetmed.2023.106009); arXiv:2408.16269 (2024, v4 2025) | Licence not stated / arXiv | Allocation and collective-treatment triggers |
| Merca2025 | Merca C et al. Combining dynamic GLMs and mechanistic modelling to optimize BRD treatment strategies. *Vet Res* 2025;56:181. doi:10.1186/s13567-025-01611-y | CC BY | Sequential updating |
| Snyder2026 | Snyder ER, Younes JA, Bird EM, Lacoste SR, Waldner CL. Contagious transmission of *M. haemolytica* in feedlot calves. *Vet Microbiol* 2026;315:110955. doi:10.1016/j.vetmic.2026.110955 | Licence not stated | Clustering (qualitative) |
| Johnson2023 | Johnson B, White B, Lancaster P, Larson R. Temporal distributions of high, low and zero cohort morbidity of first-treatment BRD. *Vet Sci* 2023;10(2):89. doi:10.3390/vetsci10020089 | CC BY | Epidemic-curve heterogeneity |
| Wisnieski2021 | Wisnieski LC et al. Weather and BRDC deaths in high-risk auction calves. *Am J Vet Res* 2021;82(8):644–652. doi:10.2460/ajvr.82.8.644 | Licence not stated | Weather term |
| Theurer2015 | Theurer ME, White BJ, Larson RL, Schroeder TC. Stochastic model of the economic value of diagnostic test characteristics for BRD. *J Anim Sci* 2015;93(3):1398–1410. doi:10.2527/jas.2014-8487 | Subscription; abstract read | Decision analysis |
| Kristensen2015 | Kristensen AR. From biological models to economic optimization. *Prev Vet Med* 2015;118(2-3):226–237. doi:10.1016/j.prevetmed.2014.11.019 | Subscription | Optimisation framing |
| VanBoeckel2015 | Van Boeckel TP et al. Global trends in antimicrobial use in food animals. *PNAS* 2015;112(18):5649–5654. doi:10.1073/pnas.1503141112 | Free via Europe PMC; Princeton | AMU context |
| Narayanan2020 | Narayanan S et al. Genome sequence classification for animal diagnostics. arXiv:2007.12791 | arXiv | Shows the arXiv gap only |
| VanCalster2019 | Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles heel of predictive analytics. *BMC Med* 2019;17:230. doi:10.1186/s12916-019-1466-7 | CC BY (BMC); metadata checked | Calibration reporting |
| Vickers2006 | Vickers AJ, Elkin EB. Decision curve analysis. *Med Decis Making* 2006;26(6):565–574. doi:10.1177/0272989X06295361 | Subscription; metadata checked | Net-benefit evaluation |

## Images
| Asset | Source | Licence | Used in |
|---|---|---|---|
| `app/assets/cattle_herd.jpg` | "Cattle herd", photo by Keith Weller, USDA Agricultural Research Service (image K4328-8), via Wikimedia Commons (File:Cattle_herd.jpg), 960 px version | Public domain (work of the US federal government) | Dashboard header banner (decorative only, not data) |

## AI tools
Claude Code (Anthropic) was used to help write code, design the synthetic data and draft documentation. The team must check every number against the PDFs before submission.

## Verification still to do
- [ ] Recheck the Weber 2024 price table in the original PDF (the extracted layout was scrambled). It is not currently used for any parameter value.
- [ ] Have the consulting veterinarian review the defaults for `metaphylaxis_morbidity_rrr`, `enhanced_monitoring_cfr_rrr` and `receiving_protocol_morbidity_rrr`, or replace them with cited trial or meta-analysis estimates.
