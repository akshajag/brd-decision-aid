"""Misspecification and planner-assumption stress tests (DESIGN s5.3).

Each scenario changes the synthetic WORLD (how data are generated) and/or the TRUE intervention effects, while the
planner keeps its default assumptions. The unchanged pipeline is refit on the scenario's history and its decisions
are scored on the scenario's truth. Scenarios run in BOTH directions: some favour the recommended plan, others
(stronger true metaphylaxis effect, no true benefit from extra riding) work against it; the worst case is the headline.
"""
from __future__ import annotations

import copy

import pandas as pd

from src.data.generate import CUR_WEIGHTS, FUTURE_DATES, HIST_DATES, HIST_WEIGHTS, simulate_world

WORLD_KEYS = {"interaction_auction_highwx", "quad_transit", "lot_sd", "mnar_long_transit", "drift_per_year"}
EFFECT_KEYS = {"true_metaphylaxis_rrr", "true_enhanced_monitoring_cfr_rrr", "true_receiving_protocol_morbidity_rrr"}

SCENARIOS: dict[str, dict] = {
    "base": {"label": "Documented process (truth = planner's assumptions)"},
    "S1_interactions": {"label": "Unmodelled sale-barn x high-weather interaction and curved transit effect",
                        "interaction_auction_highwx": 0.5, "quad_transit": 0.0015},
    "S2_heterogeneity": {"label": "More unrecorded lot-to-lot variation (logit SD 1.8 vs 1.3)", "lot_sd": 1.8},
    "S3_mnar": {"label": "Transit time missing more often when the haul was long (not at random)",
                "mnar_long_transit": 0.5},
    "drift": {"label": "Risk rises over time (+0.35 logit per year): a temporal split must detect it",
              "drift_per_year": 0.35},
    "S4_weaker_meta": {"label": "True metaphylaxis effect 0.25 (planner assumes 0.45) - favours the plan",
                       "true_metaphylaxis_rrr": 0.25},
    "S5_stronger_meta": {"label": "True metaphylaxis effect 0.65 (planner assumes 0.45) - works against the plan",
                         "true_metaphylaxis_rrr": 0.65},
    "S6_no_enh_benefit": {"label": "Extra riding and low-stress receiving have NO true effect",
                          "true_enhanced_monitoring_cfr_rrr": 0.0, "true_receiving_protocol_morbidity_rrr": 0.0},
    "S7_adverse": {"label": "Adverse effects corner: metaphylaxis 0.65 AND no benefit from extra riding/receiving",
                   "true_metaphylaxis_rrr": 0.65, "true_enhanced_monitoring_cfr_rrr": 0.0,
                   "true_receiving_protocol_morbidity_rrr": 0.0},
    "S8_combined": {"label": "Most adverse tested: S7 effects + more lot variation (SD 1.8) + drift (+0.35 logit/yr)",
                    "true_metaphylaxis_rrr": 0.65, "true_enhanced_monitoring_cfr_rrr": 0.0,
                    "true_receiving_protocol_morbidity_rrr": 0.0, "lot_sd": 1.8, "drift_per_year": 0.35},
}


def split(sc: dict) -> tuple[dict, dict]:
    return ({k: v for k, v in sc.items() if k in WORLD_KEYS}, {k: v for k, v in sc.items() if k in EFFECT_KEYS})


def generate_scenario(name: str, seed: int, cfg: dict | None = None, n_hist: int = 1400,
                      n_eval: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(history with outcomes, fresh untreated evaluation lots) from the scenario's world.

    History uses the scenario's TRUE metaphylaxis effect; evaluation lots are future arrivals, nobody treated.
    """
    from src.config import load_assumptions
    cfg = copy.deepcopy(cfg or load_assumptions())
    world, eff = split(SCENARIOS[name])
    rrr = eff.get("true_metaphylaxis_rrr")
    h = simulate_world(n_hist, seed, cfg, world=world, dates=HIST_DATES, weights=HIST_WEIGHTS, prefix="H",
                       true_rrr=rrr)
    ev = simulate_world(n_eval, seed + 1, cfg, world=world, dates=FUTURE_DATES, weights=CUR_WEIGHTS, prefix="E",
                        meta_probs=(0.0, 0.0), true_rrr=rrr)
    return h, ev


def run_stress(model_fit_fn, cfg: dict, scenarios: list[str] | None = None, n_weeks: int = 20,
               seed: int = 7, with_verdict: bool = False) -> pd.DataFrame:
    """Refit the pipeline per scenario and score prediction and decisions under that scenario's truth."""
    from src.decision.policy_eval import regret_study
    from src.model.validation import _score
    rows = []
    for name in scenarios or list(SCENARIOS):
        world, eff = split(SCENARIOS[name])
        h, ev = generate_scenario(name, seed, cfg)
        m = model_fit_fn(h)
        sc = _score(m, ev, cfg)
        rs = regret_study(m, cfg, n_weeks=n_weeks, seed=seed + 11, true_effects=eff, world=world,
                          with_verdict=with_verdict)
        hd = rs["headline"]
        rows.append({"scenario": name, "label": SCENARIOS[name]["label"], "auc": sc["auc"],
                     "cal_slope": sc["cal_slope"], "cal_in_the_large": sc["cal_in_the_large"],
                     "mean_regret": float(rs["summary"].set_index("policy").loc["recommended", "regret"]),
                     "estimated_saving": hd["estimated_saving_vs_current"],
                     "realised_saving": hd["realised_saving_vs_current"], "p_loss_week": hd["p_loss_week"],
                     "d_deaths": hd["d_deaths_vs_current"], "d_regimens": hd["d_regimens_vs_current"],
                     "floor_violations_per_week": float(
                         rs["summary"].set_index("policy").loc["recommended", "floor_violations"]),
                     "cost_of_floor": hd["cost_of_floor"],
                     "share_weeks_confident": hd.get("share_weeks_confident", float("nan")),
                     "confident_and_truly_better": hd.get("confident_and_truly_better", float("nan"))})
    return pd.DataFrame(rows)
