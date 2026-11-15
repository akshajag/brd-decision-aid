"""Constraint-aware allocation of management options to arrival lots.

Minimise expected cash cost + stewardship penalty on antimicrobial regimens, subject to:
  * one option per lot
  * per-yard rider-hour capacity for enhanced monitoring (HARD: dropping monitoring is always feasible)
  * per-yard hospital-pen peak capacity, and an optional operation-wide antimicrobial cap
    (penalised slack, so a plan is always returned; any slack is reported as a SHORTFALL to escalate,
    never as silently rationed care)
  * MINIMUM-CARE FLOOR (hard):
      - High-tier lots never get 'standard care only';
      - High-tier lots whose whole 90% risk range is >= the high-risk threshold keep metaphylaxis
        unless the veterinarian explicitly allows step-down (sidebar override);
      - lots with insufficient data never get less antimicrobial protection than the current approach.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from src.decision.consequences import OPTION_IDS, OPTIONS, outcome_table, plan_totals

def _slack_penalty() -> float:
    from src.config import load_assumptions
    return float(load_assumptions()["decision"].get("slack_penalty_usd", 1e4))


SLACK_PENALTY = _slack_penalty()   # from config (critic D19)
BASELINE_RULES = {
    "weight": "Metaphylaxis for every lot under the cut-off weight",
    "weight_auction": "Metaphylaxis for sale-barn (auction) lots under the cut-off weight",
}


def baseline_plan(lots: pd.DataFrame, cfg: dict, rule: str = "weight", cutoff: float | None = None) -> list[str]:
    """Current operating approach (ASM, editable in the sidebar)."""
    cut = cfg["decision"]["baseline_meta_weight_lb"] if cutoff is None else cutoff
    out = []
    for w, s in zip(lots["arrival_weight_lb"], lots["source_type"]):
        meta = w < cut and (rule == "weight" or s == "auction")
        out.append("META" if meta else "STD")
    return out


def capacities(cfg: dict, overrides: dict | None = None) -> dict[str, dict]:
    caps = {}
    for y, Y in cfg["yards"].items():
        caps[y] = {
            "rider_h_day": float(Y["rider_extra_hours_per_day"]),
            "hosp_free": float(Y["hospital_capacity_head"] * (1 - Y["hospital_baseline_occupancy"])),
        }
        P = cfg.get("processing")
        if P and y in P.get("chute_hours_per_day", {}):   # reported feasibility check (not an optimizer constraint)
            caps[y].update({"chute_h_week": float(P["chute_hours_per_day"][y]) * float(P["processing_days_per_week"]),
                            "chute_head_per_hour": float(P["chute_head_per_hour"]),
                            "meta_extra_frac": float(P["metaphylaxis_extra_handling_frac"])})
    for y, d in (overrides or {}).items():
        caps[y].update(d)
    return caps


def allowed_matrix(risk: pd.DataFrame, base: list[str], thr_high: float = 0.15,
                   allow_high_stepdown: bool = False, borderline_floor: bool = True) -> np.ndarray:
    """Care floor. High lots never get standard care only; neither do borderline lots whose 90% range reaches the
    high-risk cut-off (critic R2-13). Clearly high lots keep metaphylaxis unless a vet overrides. Lots with
    insufficient data never lose metaphylaxis they would get today."""
    allowed = np.ones((len(risk), len(OPTION_IDS)), bool)
    his = risk["risk_hi"] if "risk_hi" in risk else risk["risk_lo"]
    for i, (tier, conf, lo, hi) in enumerate(zip(risk["tier"], risk["confidence"], risk["risk_lo"], his)):
        for j, o in enumerate(OPTION_IDS):
            meta = OPTIONS[o][1] == 1
            if tier == "High" and o == "STD":
                allowed[i, j] = False
            if borderline_floor and hi >= thr_high and o == "STD":
                allowed[i, j] = False
            if tier == "High" and lo >= thr_high and not meta and not allow_high_stepdown:
                allowed[i, j] = False
            if conf == "insufficient data" and OPTIONS[base[i]][1] == 1 and not meta:
                allowed[i, j] = False
    return allowed


@dataclass
class PlanResult:
    choice: list[str]
    baseline: list[str]
    plan: pd.DataFrame
    constraints: pd.DataFrame
    totals: dict
    objective: float
    solver: str
    shortfalls: list[str]


def _solve(lots, tab, allowed, caps, amu_cap):
    """Returns (choice, slack_used) or None. Rider hours are hard; hospital and AMU cap have slack."""
    L, O = len(lots), len(OPTION_IDS)
    yards = sorted(lots["yard"].unique())
    Y = len(yards)
    n_var = L * O + Y + 1
    c = np.zeros(n_var)
    for j, o in enumerate(OPTION_IDS):
        c[j:L * O:O] = tab[o]["objective"]
    c[L * O:] = SLACK_PENALTY
    rows, lo, hi = [], [], []
    for i in range(L):
        r = np.zeros(n_var); r[i * O:(i + 1) * O] = 1
        rows.append(r); lo.append(1); hi.append(1)
    for k, y in enumerate(yards):
        in_y = (lots["yard"].to_numpy() == y)
        r = np.zeros(n_var)
        for j, o in enumerate(OPTION_IDS):
            r[j:L * O:O] = tab[o]["hosp_peak"] * in_y
        r[L * O + k] = -1
        rows.append(r); lo.append(-np.inf); hi.append(caps[y]["hosp_free"])
        r = np.zeros(n_var)
        for j, o in enumerate(OPTION_IDS):
            r[j:L * O:O] = tab[o]["rider_h_day"] * in_y
        rows.append(r); lo.append(-np.inf); hi.append(caps[y]["rider_h_day"] + 1e-9)
    if amu_cap is not None:
        r = np.zeros(n_var)
        for j, o in enumerate(OPTION_IDS):
            r[j:L * O:O] = tab[o]["regimens"]
        r[-1] = -1
        rows.append(r); lo.append(-np.inf); hi.append(amu_cap)
    ub = np.r_[allowed.ravel().astype(float), np.full(Y + 1, np.inf)]
    integrality = np.r_[np.ones(L * O), np.zeros(Y + 1)]
    res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi),
               integrality=integrality, bounds=Bounds(np.zeros(n_var), ub),
               options={"time_limit": 5})
    if res.x is None:
        return None
    X = res.x[:L * O].reshape(L, O)
    return [OPTION_IDS[j] for j in X.argmax(axis=1)], float(res.x[L * O:].sum())


def _greedy(lots, tab, allowed, caps):
    """Fallback: per-lot cheapest allowed option, then drop enhanced monitoring where riders overflow."""
    L = len(lots)
    obj = np.column_stack([tab[o]["objective"] for o in OPTION_IDS])
    obj[~allowed] = np.inf
    choice = [OPTION_IDS[j] for j in obj.argmin(axis=1)]
    for y in lots["yard"].unique():
        idx = [i for i in range(L) if lots["yard"].iloc[i] == y and OPTIONS[choice[i]][2]]
        used = sum(tab[choice[i]]["rider_h_day"][i] for i in idx)
        for i in idx:
            if used <= caps[y]["rider_h_day"]:
                break
            used -= tab[choice[i]]["rider_h_day"][i]
            choice[i] = "META" if (choice[i] == "ENH_META" or not allowed[i, 0]) else "STD"
    return choice, float("nan")


def _true_objective(tab, choice) -> float:
    return float(sum(tab[c]["objective"][i] for i, c in enumerate(choice)))


def usage(lots, tab, choice, caps, amu_cap) -> pd.DataFrame:
    rows = []
    for y in sorted(lots["yard"].unique()):
        in_y = [i for i in range(len(lots)) if lots["yard"].iloc[i] == y]
        for res, key, label, unit in (("hosp_peak", "hosp_free", "Hospital pen (peak head)", "head"),
                                      ("rider_h_day", "rider_h_day", "Extra rider hours/day", "h/day")):
            used = float(sum(tab[choice[i]][res][i] for i in in_y))
            cap = caps[y][key]
            rows.append({"yard": y, "resource": label, "unit": unit, "used": used, "capacity": cap,
                         "utilisation": used / cap if cap > 0 else (0.0 if used == 0 else np.inf)})
        if "chute_h_week" in caps[y]:
            c = caps[y]
            used = float(sum(lots["head_count"].iloc[i] / c["chute_head_per_hour"]
                             * (1 + c["meta_extra_frac"] * ("META" in choice[i])) for i in in_y))
            cap = c["chute_h_week"]
            rows.append({"yard": y, "resource": "Processing chute (h/week, check only)", "unit": "h/week",
                         "used": used, "capacity": cap,
                         "utilisation": used / cap if cap > 0 else (0.0 if used == 0 else np.inf)})
    if amu_cap is not None:
        used = float(sum(tab[c]["regimens"][i] for i, c in enumerate(choice)))
        rows.append({"yard": "All", "resource": "Antimicrobial regimens (cap)", "unit": "regimens",
                     "used": used, "capacity": amu_cap, "utilisation": used / amu_cap if amu_cap else np.inf})
    df = pd.DataFrame(rows)
    df["status"] = np.where(df["utilisation"] > 1.001, "SHORTFALL",
                            np.where(df["utilisation"] >= 0.95, "binding", "ok"))
    return df


def _reason(i, choice, base, risk, unconstrained, tab):
    r = risk.iloc[i]
    c, b = choice[i], base[i]
    pct = f"{100 * r['risk_mean']:.0f}% (90% range {100 * r['risk_lo']:.0f}-{100 * r['risk_hi']:.0f}%)"
    parts = [f"{r['tier']} risk, untreated expected morbidity {pct}."]
    if r["confidence"] == "insufficient data":
        parts.append("Key records missing: kept at least the current protection level.")
    elif r["confidence"] == "borderline":
        parts.append("Risk range crosses the 15% threshold: choice is sensitive to assumptions.")
    if c != unconstrained[i]:
        parts.append(f"Capacity-limited: preferred option was '{OPTIONS[unconstrained[i]][0]}'.")
    if OPTIONS[c][1] != OPTIONS[b][1]:
        d_cases = tab[c]["cases"][i] - tab[b]["cases"][i]
        d_deaths = tab[c]["deaths"][i] - tab[b]["deaths"][i]
        d_reg = tab[c]["regimens"][i] - tab[b]["regimens"][i]
        d_cost = tab[c]["cash_cost"][i] - tab[b]["cash_cost"][i]
        verb = "Adds" if OPTIONS[c][1] else "Withholds"
        parts.append(f"{verb} metaphylaxis vs current approach ({d_cases:+.1f} cases, {d_deaths:+.2f} deaths, "
                     f"{d_reg:+.0f} regimens, {'+' if d_cost >= 0 else '-'}${abs(d_cost):,.0f}).")
    if OPTIONS[c][1] != OPTIONS[b][1] or OPTIONS[c][1]:
        parts.append("Medication decision: requires veterinary review.")
    if r["tier"] == "High" and not OPTIONS[c][1]:
        parts.append("HIGH-RISK STEP-DOWN: veterinarian sign-off required.")
    return " ".join(parts)


def optimize(lots: pd.DataFrame, risk: pd.DataFrame, params: dict, cfg: dict,
             caps: dict | None = None, amu_cap: float | None = None,
             baseline_rule: str = "weight", baseline_cutoff: float | None = None,
             allow_high_stepdown: bool = False) -> PlanResult:
    lots = lots.reset_index(drop=True)
    risk = risk.reset_index(drop=True)
    caps = caps or capacities(cfg)
    p0 = risk["risk_mean"].to_numpy()
    tab = outcome_table(lots, p0, params)
    base = baseline_plan(lots, cfg, baseline_rule, baseline_cutoff)
    allowed = allowed_matrix(risk, base, cfg["decision"]["high_risk_morbidity"], allow_high_stepdown,
                             bool(cfg["decision"].get("borderline_floor", True)))
    solved = _solve(lots, tab, allowed, caps, amu_cap)
    solver = "MILP (scipy/HiGHS)"
    if solved is None:
        solved, solver = _greedy(lots, tab, allowed, caps), "greedy fallback"
    choice, slack = solved
    obj = _true_objective(tab, choice)
    big = {y: {"rider_h_day": 1e9, "hosp_free": 1e9} for y in caps}
    unc = _solve(lots, tab, allowed, big, None)
    unconstrained = unc[0] if unc else choice
    cons = usage(lots, tab, choice, caps, amu_cap)
    # marginal value of one more unit of a binding resource, in real $ (cost + stewardship penalty, no slack)
    mv, mv_note = [], []
    for _, row in cons.iterrows():
        if row["yard"] == "All" or row["status"] == "ok":
            mv.append(np.nan); mv_note.append("")
            continue
        if row["resource"].startswith("Processing chute"):   # check only: not an optimizer constraint
            mv.append(np.nan); mv_note.append("check only: stagger processing or add crew time")
            continue
        key = "hosp_free" if row["resource"].startswith("Hospital") else "rider_h_day"
        c2 = {y: dict(v) for y, v in caps.items()}
        c2[row["yard"]][key] += 1
        s2 = _solve(lots, tab, allowed, c2, amu_cap)
        if s2 is None:
            mv.append(np.nan); mv_note.append("")
        elif s2[1] < slack - 1e-6:
            mv.append(np.nan)
            mv_note.append("reduces shortfall")
        else:  # shortfall unchanged: the difference in real objective (no slack) is meaningful
            mv.append(max(obj - _true_objective(tab, s2[0]), 0.0)); mv_note.append("")
    cons["value_of_one_more_unit"] = mv
    cons["note"] = mv_note
    shortfalls = [f"Yard {r.yard}: {r.resource} needs {r.used:.1f} {r.unit} but capacity is {r.capacity:.1f}. "
                  f"Escalate: add staff or hospital space, stagger arrivals, or consult the veterinarian."
                  for r in cons.itertuples() if r.status == "SHORTFALL"]
    plan = risk[["lot_id", "yard", "head_count", "arrival_weight_lb", "tier", "confidence",
                 "risk_mean", "risk_lo", "risk_hi", "drivers"]].copy()
    plan["baseline_option"] = [OPTIONS[b][0] for b in base]
    plan["recommended_option"] = [OPTIONS[c][0] for c in choice]
    plan["changed"] = [c != b for c, b in zip(choice, base)]
    plan["vet_flag"] = [("HIGH-RISK STEP-DOWN" if t == "High" and not OPTIONS[c][1] else
                         ("metaphylaxis change" if OPTIONS[c][1] != OPTIONS[b][1] else ""))
                        for t, c, b in zip(risk["tier"], choice, base)]
    plan["expected_cases"] = [tab[c]["cases"][i] for i, c in enumerate(choice)]
    plan["expected_deaths"] = [tab[c]["deaths"][i] for i, c in enumerate(choice)]
    plan["regimens"] = [tab[c]["regimens"][i] for i, c in enumerate(choice)]
    plan["cash_cost"] = [tab[c]["cash_cost"][i] for i, c in enumerate(choice)]
    plan["reason"] = [_reason(i, choice, base, risk, unconstrained, tab) for i in range(len(lots))]
    policies = {
        "Current approach": base,
        "Recommended plan": choice,
        "Metaphylaxis for all lots": ["META"] * len(lots),
        "No metaphylaxis": ["STD"] * len(lots),
    }
    totals = {}
    for name, ch in policies.items():
        t = plan_totals(lots, p0, ch, params)
        u = usage(lots, tab, ch, caps, amu_cap)
        t["capacity_shortfalls"] = int((u["status"] == "SHORTFALL").sum())
        t["meets_care_floor"] = bool(all(allowed[i, OPTION_IDS.index(c)] for i, c in enumerate(ch)))
        totals[name] = t
    return PlanResult(choice, base, plan, cons, totals, obj, solver, shortfalls)
