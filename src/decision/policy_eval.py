"""Decision validation: policy value and regret against an oracle on fresh synthetic weeks (DESIGN s5.4).

For each simulated week of arrivals from a known synthetic world:
  * recommended = optimize() using the model's ESTIMATED risk and the planner's ASSUMED effects;
  * current     = the editable baseline rule (weight-rule metaphylaxis);
  * oracle      = argmin of the TRUE objective (true expected risk E[p|x], TRUE effects) subject to the same hard
                  rider-hour limits. It ignores the care floor, so it is a lower bound on every plan's true
                  objective and regret = true objective(plan) - true objective(oracle) >= 0 by construction
                  (critic D16: the old recipe gave planner and oracle different feasible sets).
  * oracle_floor = the same, but with the care floor built from TRUE risk (the cost of the floor itself =
                  oracle_floor - oracle).
Floor violations are counted against TRUE expected risk: a truly-high lot (E[p|x] >= 15%) on standard care only.
All scoring is in expectation given x (true E[p|x], true effects), so week-to-week noise from individual lot
effects does not enter regret; deaths/cases/costs are expected values under the truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import param_values
from src.data.generate import simulate_world
from src.decision.consequences import OPTION_IDS, outcome_table, plan_totals
from src.decision.optimize import _solve, allowed_matrix, baseline_plan, capacities, optimize
from src.model import risk as R

TRUE_KEYS = {"metaphylaxis_morbidity_rrr": "true_metaphylaxis_rrr",
             "enhanced_monitoring_cfr_rrr": "true_enhanced_monitoring_cfr_rrr",
             "receiving_protocol_morbidity_rrr": "true_receiving_protocol_morbidity_rrr"}
OUTCOMES = ["metaphylaxis_given", "brd_cases_45d", "morbidity_45d", "high_risk"]


def true_params(cfg: dict, overrides: dict | None = None) -> dict:
    """Planner parameters with the intervention effects replaced by the simulator's TRUE effects."""
    P = param_values(cfg)
    for k, tk in TRUE_KEYS.items():
        P[k] = float(cfg["dgp"][tk])
    for tk, v in (overrides or {}).items():
        for k, kk in TRUE_KEYS.items():
            if kk == tk:
                P[k] = float(v)
    return P


def _oracle(lots: pd.DataFrame, p: np.ndarray, P: dict, cfg: dict, floor: bool) -> list[str]:
    tab = outcome_table(lots, p, P)
    caps = capacities(cfg)
    big = {y: {"rider_h_day": c["rider_h_day"], "hosp_free": 1e9} for y, c in caps.items()}
    if floor:
        thr = cfg["decision"]["high_risk_morbidity"]
        risk = pd.DataFrame({"tier": np.where(p >= thr, "High", "Low"), "confidence": "confident", "risk_lo": p,
                             "risk_hi": p})
        allowed = allowed_matrix(risk, baseline_plan(lots, cfg), thr)
    else:
        allowed = np.ones((len(lots), len(OPTION_IDS)), bool)
    return _solve(lots, tab, allowed, big, None)[0]


def _arrivals(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if c.startswith("_")] + [c for c in OUTCOMES if c in df.columns])


def regret_study(model: R.RiskModel, cfg: dict, n_weeks: int = 200, lots_per_week: int = 24, seed: int = 99,
                 true_effects: dict | None = None, world: dict | None = None, n_draws: int = 20,
                 planner_params: dict | None = None, with_verdict: bool = False, n_mc: int = 100) -> dict:
    """Simulate `n_weeks` fresh weeks and score recommended / current / oracle plans under the truth."""
    Pp = planner_params or param_values(cfg)
    Pt = true_params(cfg, true_effects)
    thr = cfg["decision"]["high_risk_morbidity"]
    rows = []
    for w in range(n_weeks):
        f = simulate_world(lots_per_week, seed * 1000 + w, cfg, world=world, prefix=f"W{w}_")
        lots = _arrivals(f).reset_index(drop=True)
        p_true = f["_marg_p0"].to_numpy()
        risk, D = R.assess(model, lots, cfg, n_draws=n_draws)
        plans = {"recommended": optimize(lots, risk, Pp, cfg).choice,
                 "current": baseline_plan(lots, cfg),
                 "oracle": _oracle(lots, p_true, Pt, cfg, floor=False),
                 "oracle_floor": _oracle(lots, p_true, Pt, cfg, floor=True)}
        pe = risk["risk_mean"].to_numpy()
        est = plan_totals(lots, pe, plans["current"], Pp)["cash_cost"] - plan_totals(lots, pe, plans["recommended"], Pp)["cash_cost"]
        base_obj = plan_totals(lots, p_true, plans["oracle"], Pt)["objective"]
        hi = p_true >= thr
        conf = np.nan
        if with_verdict:   # did the tool's own Monte Carlo verdict say "confident"? (critic D20)
            from src.config import param_specs
            from src.decision import uncertainty as unc
            mc = unc.compare(lots, D, {"rec": plans["recommended"], "cur": plans["current"]}, param_specs(cfg), Pp,
                             n=n_mc, seed=w)
            corner = unc.corner_check(lots, risk["risk_mean"].to_numpy(), plans["recommended"], plans["current"],
                                      param_specs(cfg), Pp, cfg)
            conf = float(unc.robust_verdict(unc.summarize(mc, "rec", "cur"), corner, cfg)[0] == unc.ROBUST)
        for name, ch in plans.items():
            t = plan_totals(lots, p_true, ch, Pt)
            rows.append({"week": w, "policy": name, "realised_cost": t["cash_cost"], "objective": t["objective"],
                         "cases": t["cases"], "deaths": t["deaths"], "regimens": t["regimens"],
                         "floor_violations": int(sum(h and c == "STD" for h, c in zip(hi, ch))),
                         "regret": t["objective"] - base_obj,
                         "estimated_saving": est if name == "recommended" else np.nan,
                         "verdict_confident": conf if name == "recommended" else np.nan})
    pw = pd.DataFrame(rows)
    summ = pw.groupby("policy")[["realised_cost", "cases", "deaths", "regimens", "floor_violations", "regret"]].mean()
    rec = pw[pw.policy == "recommended"].set_index("week")
    cur = pw[pw.policy == "current"].set_index("week")
    extra = {"realised_saving_vs_current": float((cur.realised_cost - rec.realised_cost).mean()),
             "estimated_saving_vs_current": float(rec.estimated_saving.mean()),
             "p_loss_week": float((rec.realised_cost > cur.realised_cost).mean()),
             "d_deaths_vs_current": float((rec.deaths - cur.deaths).mean()),
             "d_regimens_vs_current": float((rec.regimens - cur.regimens).mean()),
             "cost_of_floor": float(summ.loc["oracle_floor", "regret"])}
    if with_verdict:
        c = rec.verdict_confident == 1
        good = (rec.realised_cost < cur.realised_cost) & (rec.deaths <= cur.deaths + 1e-9)
        extra["share_weeks_confident"] = float(c.mean())
        extra["confident_and_truly_better"] = float(good[c].mean()) if c.any() else float("nan")
    return {"per_week": pw, "summary": summ.reset_index(), "headline": extra}
