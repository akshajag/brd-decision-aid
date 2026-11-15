"""Tests for the days 1-14 pull-based risk update."""
import numpy as np
import pandas as pd
import pytest

from src.config import load_assumptions, param_values
from src.decision import followup as fu
from src.decision.optimize import capacities, optimize
from src.model import risk as riskmod

ROOT_DATA = "data"


@pytest.fixture(scope="module")
def world():
    cfg = load_assumptions()
    params = param_values(cfg)
    hist = pd.read_csv(f"{ROOT_DATA}/historical_cohorts.csv")
    lots = riskmod.prepare(pd.read_csv(f"{ROOT_DATA}/current_arrivals.csv")).reset_index(drop=True)
    model = riskmod.fit(hist, n_boot=8)
    risk, draws = riskmod.assess(model, lots, cfg, n_draws=16)
    res = optimize(lots, risk, params, cfg)
    return cfg, params, lots, model, draws, res


def test_onset_fraction_bounds(world):
    _, params, *_ = world
    assert fu.onset_fraction(0, params) == 0
    assert fu.onset_fraction(7, params) == pytest.approx(7 / params["case_onset_spread_days"])
    assert fu.onset_fraction(100, params) == 1


def test_no_pulls_means_no_update(world):
    _, params, lots, model, draws, res = world
    u = fu.update(lots, draws, model.resid_sd, res.choice, params, 7, np.full(len(lots), np.nan))
    assert (u["status"] == "No pulls recorded").all()
    np.testing.assert_allclose(u["updated_risk"], u["prior_risk"], rtol=1e-9)


def test_more_pulls_raise_risk_monotonically(world):
    _, params, lots, model, draws, res = world
    risks = []
    for k in (0, 5, 15, 30):
        u = fu.update(lots, draws, model.resid_sd, res.choice, params, 10, np.full(len(lots), float(k)))
        risks.append(u["updated_risk"].to_numpy())
    for a, b in zip(risks, risks[1:]):
        assert (b >= a - 1e-9).all()
    u = fu.update(lots, draws, model.resid_sd, res.choice, params, 10, np.full(len(lots), 30.0))
    assert ((u["updated_lo"] <= u["updated_risk"] + 1e-9) & (u["updated_risk"] <= u["updated_hi"] + 1e-9)).all()


def test_extreme_pulls_flag_escalation_and_actions_avoid_drug_advice(world):
    cfg, params, lots, model, draws, res = world
    pulls = np.zeros(len(lots))
    pulls[0] = lots["head_count"].iloc[0] * 0.3          # 30% pulled by day 7: far above any plan
    u = fu.update(lots, draws, model.resid_sd, res.choice, params, 7, pulls)
    assert u.loc[0, "status"] in fu.ESCALATE
    caps = capacities(cfg)
    rider = res.constraints[res.constraints["resource"].str.contains("rider")].set_index("yard")["used"].to_dict()
    acts = fu.suggested_actions(lots, u, params, caps, rider)
    assert acts and all("Veterinarian" in a for a in acts)
    assert not any(w in " ".join(acts).lower() for w in ("mg/", "dose", "tulathromycin", "florfenicol"))
    out = fu.hospital_outlook(u, params, caps, 7)
    assert set(out["yard"]) == set(lots["yard"]) and (out["projected_hospital_load"] >= 0).all()


def test_synthetic_pulls_are_deterministic_and_bounded(world):
    _, params, lots, model, draws, res = world
    a = fu.simulate_pulls(lots, draws, model.resid_sd, res.choice, params, 7)
    b = fu.simulate_pulls(lots, draws, model.resid_sd, res.choice, params, 7)
    np.testing.assert_array_equal(a, b)
    assert (a >= 0).all() and (a <= lots["head_count"].to_numpy()).all()


def test_prior_mean_matches_plan_morbidity(world):
    _, params, lots, model, draws, res = world
    u = fu.update(lots, draws, model.resid_sd, res.choice, params, 21, np.full(len(lots), np.nan))
    p0 = draws.mean(axis=1)
    plan = np.array([fu.option_morbidity(p0[i:i + 1], c, params)[0] for i, c in enumerate(res.choice)])
    np.testing.assert_allclose(u["prior_risk"], plan, rtol=0.01)


def test_demo_seed_is_typical(world):
    """Same model settings as the dashboard, so the check is about the demo the judges see."""
    cfg, params, lots, *_ = world
    model = riskmod.fit(pd.read_csv(f"{ROOT_DATA}/historical_cohorts.csv"))
    risk, draws = riskmod.assess(model, lots, cfg)
    res = optimize(lots, risk, params, cfg)
    tot = [fu.simulate_pulls(lots, draws, model.resid_sd, res.choice, params, 7, seed=s).sum() for s in range(60)]
    demo = fu.simulate_pulls(lots, draws, model.resid_sd, res.choice, params, 7).sum()
    lo, hi = np.percentile(tot, [20, 80])
    assert lo <= demo <= hi


def test_borderline_not_called_high(world):
    _, params, lots, model, draws, res = world
    pulls = np.full(len(lots), np.nan)
    u = fu.update(lots, draws, model.resid_sd, res.choice, params, 7,
                  np.where(np.isnan(pulls), lots["head_count"].to_numpy() * 0.08, pulls))
    hi = u[u["status"] == fu.NEW_HIGH]
    assert (hi["updated_lo"] >= 0.15).all()


def test_cost_breakdown_sums_to_plan_cost(world):
    from src.decision.consequences import cost_breakdown, plan_totals
    _, params, lots, model, draws, res = world
    p0 = draws.mean(axis=1)
    for plan in (res.baseline, res.choice):
        assert sum(cost_breakdown(lots, p0, plan, params).values()) == pytest.approx(
            plan_totals(lots, p0, plan, params)["cash_cost"], rel=1e-9)


def test_adequacy_statistics_behave():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.3, 300)
    perfect = riskmod.adequacy(p, p)
    assert perfect["ccc"] == pytest.approx(1.0) and perfect["msep"] == pytest.approx(0.0)
    biased = riskmod.adequacy(p + 0.05, p)                     # constant offset = pure mean bias
    assert biased["share_mean_bias"] > 0.99
    noisy = riskmod.adequacy(p + rng.normal(0, 0.05, 300), p)  # unbiased noise = mostly random error
    assert noisy["share_random"] > 0.9
    s = noisy["share_mean_bias"] + noisy["share_slope_bias"] + noisy["share_random"]
    assert s == pytest.approx(1.0, abs=0.02)


def test_contagion_clustering_widens_ranges(world):
    _, params, lots, model, draws, res = world
    pulls = lots["head_count"].to_numpy() * 0.06
    w = {}
    for icc in (0.0, 0.05):
        u = fu.update(lots, draws, model.resid_sd, res.choice, dict(params, pen_clustering_icc=icc), 10, pulls)
        w[icc] = (u["updated_hi"] - u["updated_lo"]).mean()
    assert w[0.05] > w[0.0]
