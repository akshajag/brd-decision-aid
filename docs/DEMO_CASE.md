# Demonstration case: arrival week of 16–20 Nov 2026, four yards

Reproduce with `python -m src.data.generate --seed 42` followed by `streamlit run app/app.py`, using default inputs. All numbers come from the synthetic demo data and default assumptions; the pipeline is deterministic, so a rerun reproduces them.

## The decision
- **User:** the operations manager setting processing protocols for the week's arrivals. The consulting veterinarian reviews medication changes.
- **Decision unit:** 24 arriving lots, 3,406 head, yards A–D.
- **Decision time:** day 0, at arrival and processing, using arrival information only. A days 1–14 check-in tab then updates each lot's risk from pulls recorded so far (demo counts are synthetic).
- **Options per lot:** standard care (pull and treat); arrival metaphylaxis; enhanced monitoring with low-stress receiving; or both.
- **Current operating approach (assumed, editable):** metaphylaxis for every lot under 700 lb, standard care otherwise. This is a *generous* comparator: NAHMS 2019 reports that only 15.6% of US feedlot cattle were group-treated in 2016. Results against a stricter rule are shown below.
- **Constraints:**
  - Yard B rider hours (4 h/day extra, a hard limit).
  - Yard B hospital pen (22.5 free head places).
  - Yard D hospital pen (12.5 free head places).
- **Care floor:**
  - High-risk lots never get standard care only.
  - High-risk lots whose whole 90% risk range is above 15% keep metaphylaxis, unless the vet-override box is ticked.
  - Insufficient-data lots never lose metaphylaxis they would get under the current approach.

## Inputs that matter
- **Yards differ:** Yard B receives lightweight sale-barn calves (about 530 lb, long hauls). Yard A receives heavier ranch-direct yearlings.
- **Incomplete records:** 3 lots have 2 or more key records missing (C0010, C0012, C0017) and are labelled *insufficient data*. 6 more are *borderline*, meaning their range crosses 15%.

## Results (default assumptions, current approach = metaphylaxis under 700 lb)
| Policy | Cases | Deaths | Antimicrobial regimens | Expected BRD cost | Capacity shortfalls |
|---|---:|---:|---:|---:|---:|
| Current approach | 353 | 24.7 | 3,343 | $166,869 | 2 |
| **Recommended plan** | 363 (+10.6) | 22.0* | **2,504 (−25%)** | **$154,202** | 1 |
| Metaphylaxis for all lots | 336 | 23.5 | 3,826 | $173,381 | 1 |
| No metaphylaxis | 610 | 42.7 | 763 | $169,090 | 2 (fails care floor) |

\*Deaths fall only if extra riding cuts case fatality by more than about 4%, an unsourced team assumption (default 20%). The firmer results are 839 fewer regimens (100% of draws) and $12.7k lower expected cost (96% of draws), for about 11 more cases (10.6).

**What changes (17 of 24 lots):**
- **Withholds metaphylaxis from 6 lots** (852 head), each flagged "metaphylaxis change – requires veterinary review":
  - Low or Moderate risk lots C0015, C0016, C0018 and C0020 get standard care.
  - Borderline lots C0013 and C0014 get enhanced monitoring with low-stress receiving. Their 90% range reaches 15%, so the **borderline floor** rules out standard care only.
- **Adds enhanced monitoring with low-stress receiving** to 10 high-risk lots that keep metaphylaxis (including C0001 and the three insufficient-data lots), and to C0002 (Moderate, 737 lb). In Yard B this uses all 4 rider hours, so C0022 keeps metaphylaxis only.
- **No high-risk lot loses metaphylaxis.** If the vet-override box is ticked, the optimiser would move C0001 to monitoring only. It is then shown in red as a high-risk step-down that needs sign-off, with its trade-off: about +10 cases, +0.6 deaths and −102 regimens, for about $79 more in cash (all versus the recommended plan; the lot row on screen compares with the current approach: +8.8 cases, +0.30 deaths, −104 regimens, −$89). Only the stewardship weight on regimens makes this step-down attractive; for the default plan the weight barely matters.

**Under uncertainty** (200 paired Monte Carlo draws over every assumption range plus risk-model uncertainty), the recommended plan:
- is cheaper than the current approach in **96%** of draws (90% range of the saving: $0.9k–$27k);
- has no more deaths in **94%**;
- uses fewer regimens in **100%**.

**Verdict: favourable if assumed effects hold, not robust.** The tool also checks every end (8 corners) of the three unvalidated effect ranges. The plan loses at 3 of them:
- **Deaths:** when extra riding does nothing for case fatality *and* low-stress receiving does little, the plan has more deaths: +1.5 with metaphylaxis at 25%, +3.9 at 65%, and +0.3 at 65% with receiving at 25%. (The fourth no-riding corner, metaphylaxis 25% with receiving 25%, has 4.6 *fewer* deaths.)
- **Cost:** only at the corner where metaphylaxis works at 65% *and* extra riding and low-stress receiving do nothing. There it costs **$6.1k more** (and has the 3.9 extra deaths).

So deaths hinge mainly on the extra-riding effect (with receiving); cost rises only at the joint corner where metaphylaxis works better than assumed and the other effects are small. The plan only becomes robust if the veterinarian can rule out both extra riding having little effect on case fatality and metaphylaxis working better than about 50%. That is the most useful thing the tool tells the manager: where a pen-level trial would pay.

**What drives the death result:** the death advantage rests on `enhanced_monitoring_cfr_rrr`, a team assumption with no quantified source (default 0.20, range 0–0.40). Below about 0.04 the plan has *more* deaths than the current approach. The What-if tab's deaths tornado shows this.

**What would make it robust** (panel under the corner table): one effect at a time, metaphylaxis and low-stress receiving never flip the result inside their ranges; extra riding flips deaths below 0.04. Jointly, the plan wins at every corner once the three ranges narrow 65% toward their defaults (metaphylaxis 0.38–0.52, extra-riding case-fatality reduction 0.13–0.27, receiving 0.07–0.15). The narrowing is uniform across the three ranges, so it overstates what is needed for extra riding alone: the one-way flip is at 0.04.

**Partial budget (plan minus current):** −$20.1k metaphylaxis, −$4.1k death losses, +$9.8k extra riding and receiving, +$1.7k treatment and production losses from 11 more cases.

**Capacity:** Yard B's hospital pen still needs 22.7 head places against 22.5. The tool flags this as a **shortfall to escalate** and does not ration care. Adding 1 place clears it. Yard D's hospital is binding: one more place is worth about $2,700 in expected cost plus stewardship penalty. The processing-chute check (an assumed 120 head/h, reported, not optimized) uses 13–46% of chute time in every yard.

**Against a stricter current approach** (metaphylaxis only for auction lots under 700 lb, closer to NAHMS practice), the current approach uses 1,607 regimens, $151,496 and 31.9 deaths. The same recommended plan then uses **more** antimicrobials (+897 regimens) and has about **10 fewer deaths** (in 100% of draws), but the cost difference is uncertain (cheaper in only 47% of draws, 90% range −$14.7k to +$17.2k), so the tool shows **low confidence**. The honest message is that the tool reallocates antimicrobials toward the lots that need them. Whether that saves or adds antimicrobial use depends on what the yard does today.

## 3-minute live script
0. **Start here (20 s).** Read the one-minute summary aloud: 17 lots change; 839 fewer antimicrobial courses and $12.7k less, for about 11 more sick cattle; 2.7 fewer deaths only if extra riding cuts case fatality as assumed. Then read the verdict: *favourable if the assumed effects hold*. Point at "What needs a person" and the four-question strip (Risk, Action, Consequences, Feasibility).
1. **Lots & data (30 s).** Point out the synthetic banner, the 4 yards and the missing cells shown as "None". Rehearse this edit beforehand, because the grid needs a double-click and then Enter: in lot **C0017** (Yard B, auction, 77 head), enter `n_sources = 9` and `transit_hours = 22`. Its label changes from *insufficient data* to *confident*, with risk about 46%.
2. **Risk priorities (30 s).** Show the ranked chart with 90% ranges and the 15% line. Hover a Yard B lot to show its drivers: sale barn, light weight, fall arrival.
3. **Plan vs current approach (60 s).**
   - Show the comparison table, the trade-off chart and the **partial-budget waterfall**.
   - Show the red Yard B shortfall, and Yard D's binding hospital with its value of one more place.
   - Open the **corner table** under the verdict: the rows with a positive cost or deaths difference are highlighted; they are the scenarios to discuss with the veterinarian.
4. **What-if (45 s).**
   - Raise Yard B hospital head from 22.5 to 23.5. The shortfall clears.
   - Open the "Current approach" expander and switch to "sale-barn lots under the cut-off". The antimicrobial saving becomes an increase, but deaths fall. Say why.
   - Switch back. Show the deaths tornado: monitoring effectiveness is the assumption to verify.
5. **Decision summary (15 s).** Download it and point to the verdict, the "Changes that need veterinary sign-off" section and the review block.

## Questions to prepare for
- **"Isn't this just your assumptions?"** Yes for effect sizes, and the tool labels every one EST or ASM. The metaphylaxis default (0.45) matches meta-analysis (Baptiste and Kyvsgaard 2017: pooled RR 0.49) and sits just below the 0.5 Rojas et al. 2022 used. The monitoring effect has no quantified source. That is exactly why the verdict is *favourable if assumed effects hold* and not *robust*.
- **"Your baseline treats 85% of head; NAHMS says 15.6%."** Agreed, so the current approach is editable. Against the stricter rule the plan adds antimicrobials but saves about 10 deaths, and the tool says *low confidence* on cost.
- **"Why didn't you withhold metaphylaxis from high-risk lots?"** The care floor blocks it unless a veterinarian ticks the override. Even then, each step-down is flagged in red with its trade-off.
- **"Why is your AUC 0.82 when published values are 0.68–0.79?"** The generator includes predictors Rojas lacked (source, transit, vaccination history), and synthetic prevalence is 26% against Rojas's 10.85%. On a yard the model never saw, AUC is 0.71–0.83. Synthetic validation is circular, so 0.68–0.79 is the realistic expectation.
- **"Did you check the decisions, not just the predictions?"** Yes (`docs/VALIDATION_REPORT.md`). On fresh simulated weeks scored with the true risk, weekly regret against a clairvoyant oracle was $5.0k for the plan versus $21.9k for the current rule. No truly high-risk lot was left on standard care (current rule: 1.37 per week). If the effects are wrong, regret rises to $11.5k–$22.8k per week. The tool never called those weeks *robust*.
- **"How would a yard use this on day 7?"** Open the Days 1–14 check-in tab and type pulls to date per lot. Each lot's arrival risk is updated (Bayesian, binomial likelihood; cases assumed to start evenly over the first 21 days). With the synthetic demo counts (123 pulls against 121 expected; seed 150 is typical, the median over 200 seeds being 120):
  - **C0011** (Yard A, standard care) has 26 pulls against 2.1 expected. Its updated risk is 42% (36–46%), so it is **newly high risk – escalate**.
  - **C0024** (Yard D, metaphylaxis) has 11 against 3.1 expected and is also escalated.
  - C0002 and C0004 run above expectation and are already on enhanced monitoring, so they go to the veterinarian.
  - Yard D's hospital pen is projected short next week (14.4 head against 12.5 free).
  - A beta-binomial clustering input is offered as a sensitivity check only (default off).
- **"What if risk changes over time?"** The Model & evidence tab runs a drift check on the last 180 days. On the shipped history z = +1.0, so there is no alarm. In a simulated drifting world z = +4.4, so the alarm fires and the intercept is recalibrated.
- **"Is the model biased, or just noisy?"** Concordance correlation is 0.48 against a ceiling of 0.49, which is what a *perfect* expected-risk model would score against noisy observed morbidity. Mean and slope bias are each under 1% of prediction error (Tedeschi regression slope of observed on predicted 1.00; this is not the hold-out calibration slope of 1.08). Lots vary for reasons yards do not record. Methods: Lin 1989; Tedeschi 2006.

## Limitations
Day-0 allocation plus a days 1–14 check-in that suggests actions but does not re-optimise the whole plan. Synthetic data, including the demo pull counts. Intervention effects are assumptions, and the monitoring effect is unquantified. Price years are mixed: the production loss uses 2000–01 data, drug costs are 2011 dollars, and the feeder price is a 2026 placeholder. A metaphylaxis dose and a treatment course each count as one regimen.
