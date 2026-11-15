"""Per-lot consequences of each management option.

All effect sizes and costs come from the `params` dict (config/assumptions.yaml values,
possibly changed by the manager in the What-if panel). Functions are vectorised over lots.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# id: (label, metaphylaxis, enhanced monitoring + low-stress receiving)
OPTIONS = {
    "STD": ("Standard care: pull and treat", 0, 0),
    "META": ("Arrival metaphylaxis + standard care", 1, 0),
    "ENH": ("Enhanced monitoring + low-stress receiving", 0, 1),
    "ENH_META": ("Metaphylaxis + enhanced monitoring + receiving", 1, 1),
}
OPTION_IDS = list(OPTIONS)


def pens(head: np.ndarray, params: dict) -> np.ndarray:
    return np.ceil(np.asarray(head, float) / params["head_per_pen"])


def lot_outcomes(head, weight, p0, option: str, params: dict) -> dict[str, np.ndarray]:
    """Expected outcomes for each lot under one option.

    head, weight, p0: arrays (p0 = untreated-equivalent expected 45-day morbidity).
    """
    _, meta, enh = OPTIONS[option]
    head = np.asarray(head, float)
    weight = np.asarray(weight, float)
    p0 = np.asarray(p0, float)
    morb = p0 * (1 - params["metaphylaxis_morbidity_rrr"] * meta) \
              * (1 - params["receiving_protocol_morbidity_rrr"] * enh)
    cases = head * morb
    cfr = params["case_fatality_rate"] * (1 - params["enhanced_monitoring_cfr_rrr"] * enh)
    deaths = cases * cfr
    regimens = head * meta + cases * (1 + params["retreat_fraction"])
    rider_h_day = enh * pens(head, params) * params["enhanced_monitoring_hours_per_pen_day"]
    hosp_peak = cases * params["hospital_days_per_case"] / params["case_onset_spread_days"]
    cost_treat = cases * params["treatment_cost_per_case"] * (1 + params["retreat_fraction"])
    cost_perf = cases * params["production_loss_per_case"]
    cost_death = deaths * (weight * params["feeder_price_per_lb"] + params["death_loss_extra_cost"])
    cost_meta = head * params["metaphylaxis_cost_per_head"] * meta
    cost_enh = enh * (rider_h_day * params["enhanced_monitoring_days"] * params["labor_cost_per_hour"]
                      + head * params["receiving_protocol_cost_per_head"])
    cash = cost_treat + cost_perf + cost_death + cost_meta + cost_enh
    return {
        "morbidity": morb, "cases": cases, "deaths": deaths, "regimens": regimens,
        "rider_h_day": rider_h_day, "hosp_peak": hosp_peak,
        "cost_treat": cost_treat, "cost_perf": cost_perf, "cost_death": cost_death,
        "cost_meta": cost_meta, "cost_enh": cost_enh, "cash_cost": cash,
        "objective": cash + regimens * params["amu_stewardship_weight"],
    }


def outcome_table(lots: pd.DataFrame, p0: np.ndarray, params: dict) -> dict[str, dict]:
    """{option: outcomes} for all lots."""
    return {o: lot_outcomes(lots["head_count"], lots["arrival_weight_lb"], p0, o, params)
            for o in OPTION_IDS}


COST_PARTS = {
    "cost_meta": "Metaphylaxis doses",
    "cost_treat": "Treating sick cattle",
    "cost_perf": "Lost performance of sick cattle",
    "cost_death": "Death losses",
    "cost_enh": "Extra riding + low-stress receiving",
}


def cost_breakdown(lots: pd.DataFrame, p0: np.ndarray, choice: list[str], params: dict) -> dict[str, float]:
    """Expected cost of a plan split into its parts (USD)."""
    tab = outcome_table(lots, p0, params)
    return {k: float(_pick(tab, choice, k).sum()) for k in COST_PARTS}


def _pick(tab: dict, choice: list[str], key: str) -> np.ndarray:
    """Per-lot value of `key` under each lot's chosen option (vectorised)."""
    M = np.vstack([np.asarray(tab[o][key], float) for o in OPTION_IDS])
    j = np.array([OPTION_IDS.index(c) for c in choice])
    return M[j, np.arange(len(choice))]


def plan_totals(lots: pd.DataFrame, p0: np.ndarray, choice: list[str], params: dict) -> dict:
    """Totals for a plan (one option id per lot)."""
    tab = outcome_table(lots, p0, params)
    keys = ["cases", "deaths", "regimens", "cash_cost", "objective", "cost_meta", "cost_enh"]
    out = {k: float(_pick(tab, choice, k).sum()) for k in keys}
    hc = lots["head_count"].to_numpy(float)
    head = float(hc.sum())
    out["head"] = head
    out["meta_head"] = float(hc[np.array([OPTIONS[c][1] == 1 for c in choice], bool)].sum())
    out["cost_per_head"] = out["cash_cost"] / head
    out["regimens_per_100hd"] = 100 * out["regimens"] / head
    out["morbidity"] = out["cases"] / head
    out["mortality"] = out["deaths"] / head
    return out
