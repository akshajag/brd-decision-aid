"""Decision-validation acceptance tests (critic rounds 1-2)."""
import copy

import numpy as np
import pandas as pd
import pytest

from src.config import load_assumptions, param_values
from src.data.generate import draw_lots, simulate_outcomes
from src.decision.optimize import optimize
from src.model import risk as R
from tests.critic.conftest import arrivals_only
from tests.critic.simworld import CUR_W, FUT_DATES, fresh_lots, yard_effects


def test_true_effect_is_not_the_planner_assumption(cfg):
    """D14 (BLOCKER): simulate_outcomes() reads cfg['interventions']['metaphylaxis_morbidity_rrr'], i.e. the
    synthetic 'truth' IS the planner's assumption, so any decision validation on default settings is circular.
    Acceptance: the generator's true effect lives in cfg['dgp'] (e.g. 'true_metaphylaxis_rrr') and changing the
    planner's assumption leaves simulated outcomes unchanged."""
    ye = yard_effects(cfg)
    lots = draw_lots(300, np.random.default_rng(5), cfg, FUT_DATES, CUR_W, "T")
    a = simulate_outcomes(lots, cfg, np.random.default_rng(6), ye)
    cfg2 = copy.deepcopy(cfg)
    cfg2["interventions"]["metaphylaxis_morbidity_rrr"]["value"] = 0.05
    b = simulate_outcomes(lots, cfg2, np.random.default_rng(6), ye)
    assert (a["brd_cases_45d"].to_numpy() == b["brd_cases_45d"].to_numpy()).all()


def test_regret_study_api(cfg):
    """D15: s5.4 (the 'primary result') is not implemented. Expected API:
    src.decision.policy_eval.regret_study(model, cfg, n_weeks=200, lots_per_week=24, seed=..., true_effects=None)
    -> {'per_week': DataFrame[week, policy, realised_cost, cases, deaths, regimens, floor_violations, regret],
        'summary': DataFrame by policy}; policies include 'recommended', 'current', 'oracle'.
    Oracle = argmin of the TRUE objective over the feasible set defined by TRUE risk, so regret >= 0 by
    construction; floor violations are counted against TRUE marginal p0."""
    from src.decision.policy_eval import regret_study
    hist = fresh_lots(600, seed=2024, cfg=cfg, dates=pd.date_range("2024-01-01", "2025-12-31"))
    m = R.fit(hist, n_boot=3)
    out = regret_study(m, cfg, n_weeks=4, lots_per_week=24, seed=99)
    pw = out["per_week"]
    for c in ("week", "policy", "realised_cost", "cases", "deaths", "regimens", "floor_violations", "regret"):
        assert c in pw.columns
    assert {"recommended", "current", "oracle"} <= set(pw["policy"])
    assert (pw["regret"] >= -1e-6).all()



def test_regret_definitions_policy_eval(model, cfg):
    """D16 (rewritten in round 2; builder's rebuttal ACCEPTED). The round-1 version fed optimize() a risk table with
    risk_lo = risk_hi = true p, so the rule-6 clause 'High lots with risk_lo >= 15% keep metaphylaxis' bound for
    every truly-high lot and the 'oracle' paid the floor; negative 'regret' was the cost of the floor, which rule 6
    forbids removing. policy_eval now scores every plan against a floor-free oracle and reports the floor's cost
    separately. Acceptance (definitions, checked on fresh weeks):
      * regret >= 0 for every policy and week (oracle = argmin of the true objective within rider-hour limits);
      * oracle_floor regret >= 0 and equals headline cost_of_floor (mean);
      * oracle_floor leaves no truly-high lot on standard care (floor built from TRUE risk);
      * policies include recommended, current, oracle, oracle_floor."""
    from src.decision.policy_eval import regret_study
    out = regret_study(model, cfg, n_weeks=5, lots_per_week=24, seed=4321)
    pw = out["per_week"]
    assert {"recommended", "current", "oracle", "oracle_floor"} <= set(pw["policy"])
    assert (pw["regret"] >= -1e-6).all(), pw.loc[pw.regret < -1e-6, ["week", "policy", "regret"]]
    assert (pw.loc[pw.policy == "oracle", "regret"].abs() < 1e-6).all()
    of = pw[pw.policy == "oracle_floor"]
    assert (of["floor_violations"] == 0).all()
    assert abs(of["regret"].mean() - out["headline"]["cost_of_floor"]) < 1e-6


def test_truly_high_lots_not_left_on_standard_care(model, cfg):
    """D17 (rule 6): lots whose TRUE expected untreated risk is >= 15% must not get STD. The floor uses the
    (downward-biased, see D8) point estimate, so truly-high lots can slip through (probe: ~0.1-0.2 per week).
    Passes today at the 1% bar on these seeds: REGRESSION GUARD for any change to the floor or the model.
    Acceptance: <= 1% of truly-high lots get STD over 12 fresh weeks."""
    P = param_values(cfg)
    thr = cfg["decision"]["high_risk_morbidity"]
    n_hi = n_bad = 0
    for w in range(12):
        f = fresh_lots(24, seed=60_000 + w, cfg=cfg)
        lots = arrivals_only(f).reset_index(drop=True)
        r, _ = R.assess(model, lots, cfg, n_draws=12)
        ch = optimize(lots, r, P, cfg).choice
        hi = f["_marg_p0"].to_numpy() >= thr
        n_hi += int(hi.sum())
        n_bad += int(sum(h and c == "STD" for h, c in zip(hi, ch)))
    assert n_bad <= 0.01 * n_hi, (n_bad, n_hi)


def test_stress_includes_adverse_effect_direction():
    """D18: S4 only stresses a SMALLER true metaphylaxis effect (0.25), which FAVOURS a plan that withdraws
    metaphylaxis (probe: realised saving rises to ~21k USD/week). The adverse case (true RRR 0.65, ENH effects 0)
    cuts realised saving from ~15k to ~0.6k USD/week, loses money in ~half of weeks, +1.9 deaths/week.
    Expected API: src.model.stress.SCENARIOS: dict name -> dict of overrides; must contain a scenario with
    true metaphylaxis RRR above the planner's and one with zero true ENH effects; and
    src.model.stress.run_stress(model_fit_fn, cfg, scenarios, n_weeks, seed) -> DataFrame
    [scenario, auc, cal_slope, mean_regret, realised_saving, p_loss_week, d_deaths]."""
    from src.model.stress import SCENARIOS
    planner = load_assumptions()["interventions"]["metaphylaxis_morbidity_rrr"]["value"]
    assert any(s.get("true_metaphylaxis_rrr", 0) > planner for s in SCENARIOS.values())
    assert any(s.get("true_enhanced_monitoring_cfr_rrr", None) == 0 for s in SCENARIOS.values())


def test_verdict_thresholds_in_config(cfg):
    """D19 (rule 2, forking paths): MIN_P_CHEAPER / MIN_P_NO_MORE_DEATHS and the confidence/escalation cut-offs
    (>=2 missing critical fields, 5/95 percentiles, 1.5x expected + 3 pulls) are magic numbers in code.
    Acceptance: cfg['decision'] contains 'verdict_min_p_cheaper' and 'verdict_min_p_no_more_deaths'."""
    assert {"verdict_min_p_cheaper", "verdict_min_p_no_more_deaths"} <= set(cfg["decision"])


@pytest.mark.slow
def test_verdict_not_confidently_wrong_inside_declared_ranges(model, cfg):
    """R2-1 (D20 follow-up, MAJOR). The Monte Carlo verdict samples each effect from triangular(low, value, high), so
    'confident' means confident under a prior centred on the planner's assumption. In S6 (no true benefit from extra
    riding / receiving; both values are the LOW END of the planner's own declared ranges, so inside what the tool
    claims to cover) the validation report shows the tool said 'confident' in 72% of weeks but was truly better
    (cheaper AND no more deaths) in only 6% of those weeks. A verdict labelled confident must be right most of the time
    whenever the truth lies inside the declared ranges.
    Acceptance: in the S6 world, share of 'confident' weeks that are truly better >= 0.8, OR 'confident' is
    issued in <= 20% of weeks. Fix options: require the verdict criteria to hold at the unfavourable end of each
    unvalidated effect range (corner / one-at-a-time check), or split the verdict into 'robust' vs 'depends on
    assumed effects' and show the latter for any plan whose death benefit rests on enhanced-monitoring effects."""
    from src.decision.policy_eval import regret_study
    from src.model.stress import SCENARIOS, split
    _, eff = split(SCENARIOS["S6_no_enh_benefit"])
    hd = regret_study(model, cfg, n_weeks=8, seed=4322, true_effects=eff, with_verdict=True, n_mc=100)["headline"]
    ok = hd["share_weeks_confident"] <= 0.2 or hd["confident_and_truly_better"] >= 0.8
    assert ok, {k: round(hd[k], 2) for k in ("share_weeks_confident", "confident_and_truly_better",
                                             "d_deaths_vs_current", "realised_saving_vs_current")}
