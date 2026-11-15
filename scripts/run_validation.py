"""Run the full pre-specified validation protocol (docs/DESIGN.md s5) and write the results.

Outputs:
  reports/validation_results.json   machine-readable results (read by the dashboard's Model & evidence tab)
  docs/VALIDATION_REPORT.md         human-readable report

Everything is SYNTHETIC. Seeds are listed in data/generator_manifest.json (fresh-evaluation seeds were never used
while the generator was tuned).  Usage:  python -m scripts.run_validation [--quick]
Runtime: about 10-15 minutes (about 2 minutes with --quick).
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import date

import numpy as np
import pandas as pd

from src.config import ROOT, load_assumptions
from src.data.generate import HIST_DATES, HIST_WEIGHTS, config_hash, simulate_world
from src.decision.policy_eval import regret_study
from src.model import risk as R
from src.model import stress as S
from src.model import validation as V

warnings.filterwarnings("ignore")
OUT_JSON = ROOT / "reports" / "validation_results.json"
OUT_MD = ROOT / "docs" / "VALIDATION_REPORT.md"
ARR_DROP = ["metaphylaxis_given", "brd_cases_45d", "morbidity_45d", "high_risk"]


def arrivals(df):
    return df.drop(columns=[c for c in df.columns if c.startswith("_")] + [c for c in ARR_DROP if c in df.columns])


def replicate_worlds(cfg, k: int, n_boot: int) -> pd.DataFrame:
    """Same synthetic world, k independent histories (plus the shipped CSV): is the model's error systematic or
    sampling noise? Coverage, P(high) calibration and band bias on 1,000 fresh untreated lots (seed 777)."""
    f = simulate_world(1000, 777, cfg, meta_probs=(0.0, 0.0))
    A = arrivals(f)
    mp = f["_marg_p0"].to_numpy()
    band = pd.cut(f["arrival_weight_lb"], V.BANDS, right=False)
    rows = []
    for s in [None] + list(range(k)):
        h = pd.read_csv(ROOT / "data" / "historical_cohorts.csv") if s is None else \
            simulate_world(1400, 9000 + s, cfg, dates=HIST_DATES, weights=HIST_WEIGHTS, prefix="H")
        m = R.fit(h, n_boot=n_boot, seed=0)
        r, _ = R.assess(m, A, cfg, n_draws=40)
        b = pd.DataFrame({"b": band, "d": m.predict_mean(f) - mp}).groupby("b", observed=True)["d"].mean()
        rows.append({"history": "shipped CSV (seed 42)" if s is None else f"fresh seed {9000 + s}",
                     "lot_sd": m.resid_sd,
                     "coverage_90": float(np.mean((r["risk_lo"] <= mp) & (mp <= r["risk_hi"]))),
                     "mean_p_high": float(r["p_high"].mean()),
                     "true_share_high": float((f["_true_p0"] >= cfg["decision"]["high_risk_morbidity"]).mean()),
                     **{f"bias_{str(i)}": float(v) for i, v in b.items()}})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    q = a.quick
    t0 = time.time()
    cfg = load_assumptions()
    hist = pd.read_csv(ROOT / "data" / "historical_cohorts.csv")
    res = {"generated": date.today().isoformat(), "config_hash": config_hash(), "quick": q}

    ev = R.evaluate(hist, cfg)
    res["temporal"] = {k: ev[k] for k in ("n_train", "n_test", "auc", "brier", "cal_in_the_large", "cal_slope",
                                          "resid_sd", "sensitivity", "specificity", "ppv", "npv", "prevalence")}
    res["temporal"]["ccc"] = ev["adequacy"]["ccc"]
    res["temporal"]["ccc_ceiling"] = ev["adequacy"]["ccc_ceiling"]
    tr, te = V.temporal_split(hist)
    res["baselines"] = V.baselines(tr, te, cfg)
    res["loyo"] = V.leave_one_yard_out(hist, cfg).to_dict("records")
    res["positivity"] = V.positivity(hist).to_dict("records")
    print(f"[{time.time() - t0:.0f}s] temporal, baselines, LOYO done", flush=True)

    model = R.fit(hist, n_boot=10 if q else 100, seed=0)
    res["drift_check"] = {k: v for k, v in model.drift.items()}
    res["oracle_recovery"] = V.oracle_recovery(model, cfg, n=1000, seed=4242).to_dict("records")
    res["replicates"] = replicate_worlds(cfg, k=2 if q else 8, n_boot=5 if q else 30).to_dict("records")
    print(f"[{time.time() - t0:.0f}s] oracle recovery and replicate worlds done", flush=True)

    rs = regret_study(model, cfg, n_weeks=10 if q else 60, seed=99, with_verdict=True)
    res["regret_base"] = {"summary": rs["summary"].to_dict("records"), "headline": rs["headline"]}
    print(f"[{time.time() - t0:.0f}s] regret study done", flush=True)

    st = S.run_stress(lambda h: R.fit(h, n_boot=5 if q else 10, seed=0), cfg,
                      n_weeks=6 if q else 25, seed=7, with_verdict=True)
    res["stress"] = st.to_dict("records")
    # drift: does the temporal split detect it?
    h, _ = S.generate_scenario("drift", seed=101)
    evd = R.evaluate(h, cfg)
    res["drift_temporal"] = {"cal_in_the_large": evd["cal_in_the_large"], "auc": evd["auc"]}
    res["drift_scenario_check"] = {k: v for k, v in R.fit(h, n_boot=0, seed=0).drift.items()}
    print(f"[{time.time() - t0:.0f}s] stress done", flush=True)

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8")
    print(f"wrote {OUT_JSON} and {OUT_MD} in {time.time() - t0:.0f}s")


def _tbl(rows: list[dict], cols: list[str], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(fmt.get(c, "{}").format(r[c]) if isinstance(r[c], (int, float)) and c in fmt
                                     else str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def render(r: dict) -> str:
    t = r["temporal"]
    b = r["baselines"]
    rb = r["regret_base"]["headline"]
    summ = {x["policy"]: x for x in r["regret_base"]["summary"]}
    rep = pd.DataFrame(r["replicates"])
    sims = rep.iloc[1:]
    worst = min(r["stress"], key=lambda x: x["realised_saving"])
    wdeath = max(r["stress"], key=lambda x: x["d_deaths"])   # worst on health, reported alongside (critic R3-1)
    stv = {x["scenario"]: x for x in r["stress"]}
    dc, dsc = r.get("drift_check", {}), r.get("drift_scenario_check", {})
    L = [
        "# Validation report (synthetic data)",
        "",
        f"Generated {r['generated']} by `python -m scripts.run_validation` · generator config hash `{r['config_hash']}`"
        + (" · **QUICK RUN (fewer weeks/replicates)**" if r["quick"] else ""),
        "",
        "> **Read this first.** No real data was provided. Every number below comes from data we simulated from our "
        "own assumptions. These checks show the pipeline is leakage-free, recovers a known process, fails in "
        "measured ways when that process is changed, and makes decisions with quantified regret. They are **not** "
        "evidence of real-world accuracy. On real feedlot data, Rojas et al. (2022) report cohort AUC 0.68-0.79 and "
        "PPV at most 25%.",
        "",
        "## 1. Headline findings",
        f"1. **Prediction (temporal hold-out, 2026 arrivals).** AUC {t['auc']:.2f}, Brier {t['brier']:.3f}, "
        f"calibration slope {t['cal_slope']:.2f}, calibration-in-the-large {100 * t['cal_in_the_large']:+.1f} pts. "
        f"The full model beats every simple baseline on Brier score (weight-only rule {b['B1']['brier']:.3f}, "
        f"yard × source {b['B2']['brier']:.3f}, full model {b['full']['brier']:.3f}).",
        f"2. **Honest transfer to a new yard (leave-one-yard-out).** AUC "
        + ", ".join(f"{x['yard']} {x['auc']:.2f}" for x in r["loyo"]) + ". This is the number to quote for a "
        "yard the model has never seen.",
        f"3. **Uncertainty is close to nominal on average, not in every dataset.** Over {len(sims)} fresh histories, the "
        f"90% range covered the true expected risk for {100 * sims['coverage_90'].mean():.0f}% of lots on average "
        f"(range {100 * sims['coverage_90'].min():.0f}-{100 * sims['coverage_90'].max():.0f}%). For the shipped "
        f"dataset it was {100 * rep.iloc[0]['coverage_90']:.0f}%: all lots share one fitted model, so a fit that is "
        "off by chance is off for everyone. Mean P(≥15%) was "
        f"{sims['mean_p_high'].mean():.2f} against a true share of {sims['true_share_high'].mean():.2f}.",
        f"4. **Decisions (regret study, {len(r['regret_base']['summary']) and 'fresh weeks'}).** Against a "
        f"clairvoyant oracle that knows each lot's true expected risk and the true effects, the recommended plan's "
        f"regret averaged ${summ['recommended']['regret']:,.0f}/week versus ${summ['current']['regret']:,.0f}/week for "
        f"the current approach. Expected saving under the simulated truth versus the current approach: "
        f"${rb['realised_saving_vs_current']:,.0f}/week (the tool estimated ${rb['estimated_saving_vs_current']:,.0f}); "
        f"it cost more in {100 * rb['p_loss_week']:.0f}% of weeks; deaths changed by {rb['d_deaths_vs_current']:+.2f}/week. "
        f"The care floor itself costs ${rb['cost_of_floor']:,.0f}/week relative to the oracle (the oracle ignores the "
        "floor and has unlimited hospital space). **Caveat: here the simulated truth equals the planner's own "
        "assumptions**, so this shows the risk model and optimizer work, not that the assumed effects are right; "
        "savings are expected values (lot-level chance is not added). When the effects are wrong the regret rises: "
        + ", ".join(f"{k.split('_')[0]} ${stv[k]['mean_regret']:,.0f}" for k in ("S6_no_enh_benefit", "S7_adverse", "S8_combined") if k in stv)
        + "/week.",
        f"5. **Most adverse scenario tested.** Under '{worst['label']}', the expected saving falls to "
        f"${worst['realised_saving']:,.0f}/week, with a loss in {100 * worst['p_loss_week']:.0f}% of weeks and "
        f"{worst['d_deaths']:+.2f} deaths/week versus the current approach. "
        + ("This is also the worst scenario for cattle health. " if wdeath is worst else
           f"**Worst for cattle health:** under '{wdeath['label']}', deaths change by **{wdeath['d_deaths']:+.2f}/week** "
           f"and the plan costs more in {100 * wdeath['p_loss_week']:.0f}% of weeks (expected saving "
           f"${wdeath['realised_saving']:,.0f}/week). ")
        +
        "**This is the headline risk:** the "
        "recommendation depends on assumed intervention effects that these data cannot validate.",
        f"6. **Does the verdict know when to be unsure?** The tool only says *robust* when the plan also wins at "
        "every end of the unvalidated effect ranges; otherwise it says *favourable if assumed effects hold* or "
        "*low confidence*. "
        + (lambda h, n: f"In the base regret study it said *robust* in {round(h['share_weeks_confident'] * n)} of {n} "
           f"weeks" + ("" if h['confident_and_truly_better'] != h['confident_and_truly_better'] else
                       f" ({round(h['confident_and_truly_better'] * h['share_weeks_confident'] * n)} of them truly better)"))
          (rb, 10 if r["quick"] else 60)
        + f"; in the stress worlds ({6 if r['quick'] else 25} weeks each) it said *robust* in "
        + ", ".join(f"{k.split('_')[0]} {round(stv[k]['share_weeks_confident'] * (6 if r['quick'] else 25))}"
                    for k in ("S6_no_enh_benefit", "S7_adverse", "S8_combined") if k in stv)
        + " weeks, but *favourable if assumed effects hold* is conditional: in the stress worlds the plan can still lose in most weeks while the tool says exactly that. In a typical week the plan loses at several corners of the "
        "declared effect ranges: **deaths rise when extra riding does not lower case fatality and low-stress "
        "receiving does little**, and **cost rises when metaphylaxis works better than assumed and the other effects are "
        "small**. Narrowing those two ranges (veterinary judgement, or a pen-level trial of extra riding) is what would "
        "make a robust call possible: in our simulations the decision was far more sensitive to these two effect sizes "
        "than to the risk-model misspecifications we tested (S1–S3).",
        f"7. **Drift alarm.** On the shipped history the recent {dc.get('window_days', 180)}-day check gives "
        f"z = {dc.get('z', float('nan')):+.2f} (alarm {'ON' if dc.get('alarm') else 'off'}). On the drift scenario "
        f"z = {dsc.get('z', float('nan')):+.2f} (alarm {'ON' if dsc.get('alarm') else 'off'}; intercept shifted by "
        f"{dsc.get('offset', 0):+.2f} logit when on). The critic's probe found the alarm detects a drift of 0.35 "
        "logit/year in 10 of 10 histories but 0.15 logit/year in only 1 of 10, with about 1 false alarm in 30 "
        "stable histories; it corrects the level of risk only, not changes in individual risk factors.",
        "",
        "## 2. Prediction",
        "### 2.1 Temporal split (train before 2026-01-01, test after)",
        f"n_train {t['n_train']}, n_test {t['n_test']}, observed high-risk prevalence {t['prevalence']:.2f}. "
        f"Sensitivity {t['sensitivity']:.2f}, specificity {t['specificity']:.2f}, PPV {t['ppv']:.2f}, NPV {t['npv']:.2f}. "
        f"Lot-to-lot logit SD estimated out of fold: {t['resid_sd']:.2f} (simulator: 1.30).",
        "",
        "The label is *observed* morbidity ≥15% after any metaphylaxis, while the decision uses *untreated* risk. "
        "Section 3 therefore checks untreated risk directly against the known truth.",
        "",
        f"Concordance (CCC) {t['ccc']:.2f} against a simulated ceiling of {t['ccc_ceiling']:.2f}: the ceiling is "
        "what a perfect expected-risk model scores against noisy observed morbidity, so CCC alone cannot separate a "
        "good model from a perfect one.",
        "",
        "### 2.2 Baselines (temporal split, observed high-risk label)",
        _tbl([{"model": k, **v} for k, v in b.items()], ["model", "brier", "auc"], {"brier": "{:.3f}", "auc": "{:.3f}"}),
        "",
        "B0 overall prevalence · B1 arrival weight only (the information the current rule uses) · B2 yard × source "
        "prevalence.",
        "",
        "### 2.3 Leave-one-yard-out",
        _tbl(r["loyo"], ["yard", "n_test", "auc", "brier", "cal_in_the_large", "cal_slope"],
             {"auc": "{:.3f}", "brier": "{:.3f}", "cal_in_the_large": "{:+.3f}", "cal_slope": "{:.2f}"}),
        "",
        "## 3. Recovery of the known truth (only possible with synthetic data)",
        "### 3.1 Oracle recovery: predicted untreated risk vs the true E[p | complete covariates], 1,000 fresh lots",
        _tbl(r["oracle_recovery"], ["band", "n", "bias", "rmse"], {"bias": "{:+.4f}", "rmse": "{:.4f}"}),
        "",
        "### 3.2 Positivity: how many untreated lots support the untreated-risk estimate",
        _tbl(r["positivity"], ["band", "lots", "treated_share", "untreated_lots"], {"treated_share": "{:.2f}"}),
        "",
        "Light lots were mostly treated, so their untreated risk rests on few untreated lots (confounding by "
        "indication). Their predictions lean on the adjustment for metaphylaxis.",
        "",
        "### 3.3 Is the error systematic or chance? Independent histories from the same world",
        _tbl(r["replicates"], ["history", "lot_sd", "coverage_90", "mean_p_high", "true_share_high"]
             + [c for c in rep.columns if c.startswith("bias_")],
             {**{c: "{:+.4f}" for c in rep.columns if c.startswith("bias_")},
              "lot_sd": "{:.2f}", "coverage_90": "{:.2f}", "mean_p_high": "{:.3f}", "true_share_high": "{:.3f}"}),
        "",
        "Bias columns are predicted minus true expected untreated risk by arrival-weight band (lb). The shipped "
        "dataset's fit sits at the low end of what independent histories produce; averaged over histories the "
        "bias is small. That is the behaviour expected of an approximately unbiased estimator with real sampling "
        "error, which the 90% range is meant to express.",
        "",
        "## 4. Decisions: regret against an oracle",
        "Each fresh week (24 lots) is planned from estimated risk and assumed effects, then scored with the true "
        "expected risk and true effects. *oracle* = best plan under the truth, within the rider-hour limits, "
        "ignoring the care floor (so regret ≥ 0 for every policy). *oracle_floor* = the same with the care floor "
        "built from true risk. Floor violations = truly-high lots on standard care only.",
        "",
        _tbl(r["regret_base"]["summary"], ["policy", "realised_cost", "cases", "deaths", "regimens",
                                           "floor_violations", "regret"],
             {"realised_cost": "${:,.0f}", "cases": "{:.1f}", "deaths": "{:.2f}", "regimens": "{:.0f}",
              "floor_violations": "{:.2f}", "regret": "${:,.0f}"}),
        "",
        f"Verdict calibration (critic D20): the tool said *robust* in "
        f"{100 * rb.get('share_weeks_confident', float('nan')):.0f}% of weeks; in those weeks the plan was truly "
        f"cheaper with no more deaths {100 * rb.get('confident_and_truly_better', float('nan')):.0f}% of the time. "
        "In this base world, truth equals the planner's assumptions, so this is the favourable case. See section 5.",
        "",
        "## 5. Stress tests (pipeline refit on each world; planner keeps its default assumptions; 25 weeks per world, so the base row differs from the 60-week study above)",
        _tbl(r["stress"], ["scenario", "auc", "cal_slope", "mean_regret", "estimated_saving", "realised_saving",
                           "p_loss_week", "d_deaths", "floor_violations_per_week"],
             {"auc": "{:.3f}", "cal_slope": "{:.2f}", "mean_regret": "${:,.0f}", "estimated_saving": "${:,.0f}",
              "realised_saving": "${:,.0f}", "p_loss_week": "{:.2f}", "d_deaths": "{:+.2f}",
              "floor_violations_per_week": "{:.2f}", "confident_and_truly_better": "{:.2f}"}),
        "",
        "\n".join(f"- **{x['scenario']}**: {x['label']}" for x in r["stress"]),
        "",
        f"Drift check: with risk rising 0.35 logit/year, the temporal split shows calibration-in-the-large "
        f"{100 * r['drift_temporal']['cal_in_the_large']:+.1f} pts (observed above predicted), so the temporal split "
        "*can* fail and does flag drift. In the tool, the drift check runs at every refit and recalibrates the intercept when it fires (headline 7).",
        "",
        "## 6. What this does and does not show",
        "- **Shows:** no leakage (tests in `tests/critic/test_leakage.py`); approximately unbiased recovery of a known "
        "process; calibrated P(≥15%); quantified regret; failure modes in both directions.",
        "- **Does not show:** real-world accuracy, real intervention effects, or savings for these yards. The benefit "
        "of extra riding and low-stress receiving can never be validated on these data, because the history contains "
        "no such interventions.",
    ]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
