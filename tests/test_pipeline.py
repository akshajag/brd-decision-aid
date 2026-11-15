"""Unit tests for data generation, risk model and decision layer."""
import numpy as np
import pandas as pd
import pytest

from src.config import load_assumptions, param_values
from src.data.generate import ARRIVAL_COLUMNS, generate
from src.decision.consequences import OPTION_IDS, lot_outcomes
from src.decision.optimize import allowed_matrix, baseline_plan, capacities, optimize
from src.model import risk as riskmod


@pytest.fixture(scope="module")
def world():
    hist, cur, cfg = generate(seed=42)
    model = riskmod.fit(hist, n_boot=8)
    r, draws = riskmod.assess(model, cur, cfg, n_draws=16)
    return hist, cur, cfg, model, r, draws


# ---------- data ----------
def test_generator_is_deterministic():
    a, b, _ = generate(seed=3, n_hist=200, n_current=10)
    c, d, _ = generate(seed=3, n_hist=200, n_current=10)
    pd.testing.assert_frame_equal(a, c)
    pd.testing.assert_frame_equal(b, d)


def test_current_arrivals_have_no_outcomes(world):
    _, cur, *_ = world
    assert list(cur.columns) == ARRIVAL_COLUMNS
    assert not {"brd_cases_45d", "morbidity_45d", "high_risk"} & set(cur.columns)


def test_missingness_is_present_and_worse_for_auction(world):
    hist, *_ = world
    auc = hist[hist.source_type == "auction"]["transit_hours"].isna().mean()
    oth = hist[hist.source_type != "auction"]["transit_hours"].isna().mean()
    assert auc > oth > 0


def test_morbidity_is_plausible(world):
    hist, *_ = world
    # between Rojas 2022 (6.3%) and NAHMS 2011 (21.2%)
    assert 0.06 <= hist["morbidity_45d"].mean() <= 0.21
    assert hist.groupby("yard")["morbidity_45d"].mean().agg(np.ptp) > 0.05  # yards differ


# ---------- risk ----------
def test_no_outcome_leakage_in_design(world):
    _, _, _, model, *_ = world
    leaks = {"brd_cases_45d", "morbidity_45d", "high_risk"}
    assert not leaks & set(model.mu.index)


def test_risk_outputs_valid(world):
    *_, r, draws = world
    assert ((r.risk_mean > 0) & (r.risk_mean < 1)).all()
    assert (r.risk_lo <= r.risk_mean + 1e-9).all() and (r.risk_mean <= r.risk_hi + 1e-9).all()
    assert draws.shape[0] == len(r)


def test_missing_critical_fields_trigger_insufficient(world):
    _, cur, cfg, model, *_ = world
    lots = cur.copy()
    lots.loc[0, ["transit_hours", "n_sources", "shrink_pct"]] = np.nan
    r, _ = riskmod.assess(model, lots, cfg, n_draws=8)
    assert r.loc[0, "confidence"] == "insufficient data"


def test_validation_catches_bad_upload(world):
    _, cur, *_ = world
    assert riskmod.validate_lots(cur) == []
    bad = cur.drop(columns=["arrival_weight_lb"])
    assert any("arrival_weight_lb" in p for p in riskmod.validate_lots(bad))
    bad2 = cur.copy(); bad2.loc[0, "yard"] = "Z"
    assert riskmod.validate_lots(bad2)


def test_synthetic_auc_not_overclaimed():
    hist, _, cfg = generate(seed=42)
    ev = riskmod.evaluate(hist, cfg)
    assert 0.65 <= ev["auc"] <= 0.88  # near published 0.68-0.79; far above would signal an unrealistic DGP


# ---------- decision ----------
def test_zero_effect_gives_no_benefit():
    p = param_values(load_assumptions())
    p.update(metaphylaxis_morbidity_rrr=0.0)
    h, w, p0 = np.array([100.0]), np.array([600.0]), np.array([0.2])
    std = lot_outcomes(h, w, p0, "STD", p)
    meta = lot_outcomes(h, w, p0, "META", p)
    assert meta["cases"][0] == pytest.approx(std["cases"][0])
    assert meta["cash_cost"][0] > std["cash_cost"][0]


def test_baseline_reproduces_weight_rule(world):
    _, cur, cfg, *_ = world
    base = baseline_plan(cur, cfg)
    assert all((b == "META") == (w < 700) for b, w in zip(base, cur.arrival_weight_lb))


def test_care_floor_never_violated(world):
    _, cur, cfg, _, r, _ = world
    res = optimize(cur, r, param_values(cfg), cfg)
    allowed = allowed_matrix(r, res.baseline)
    for i, c in enumerate(res.choice):
        assert allowed[i, OPTION_IDS.index(c)]
    assert all(c != "STD" for c, t in zip(res.choice, r.tier) if t == "High")


def test_capacity_respected_when_feasible(world):
    _, cur, cfg, _, r, _ = world
    caps = {y: {"rider_h_day": 100.0, "hosp_free": 1000.0} for y in cfg["yards"]}
    res = optimize(cur, r, param_values(cfg), cfg, caps=caps)
    assert (res.constraints["status"] != "SHORTFALL").all()
    tight = {y: {"rider_h_day": 0.0, "hosp_free": 1000.0} for y in cfg["yards"]}
    res2 = optimize(cur, r, param_values(cfg), cfg, caps=tight)
    assert not any("ENH" in c for c in res2.choice)


def test_monotonic_care_with_risk(world):
    _, cur, cfg, _, r, _ = world
    p = param_values(cfg)
    lots = pd.concat([cur.iloc[[0]]] * 2, ignore_index=True)
    lots["lot_id"] = ["X1", "X2"]
    rr = pd.concat([r.iloc[[0]]] * 2, ignore_index=True)
    rr["lot_id"] = ["X1", "X2"]
    rr["risk_mean"] = [0.03, 0.40]
    rr["tier"] = ["Low", "High"]
    rr["confidence"] = "confident"
    big = {y: {"rider_h_day": 100.0, "hosp_free": 1000.0} for y in cfg["yards"]}
    res = optimize(lots, rr, p, cfg, caps=big)
    rank = {"STD": 0, "ENH": 1, "META": 1, "ENH_META": 2}
    assert rank[res.choice[1]] >= rank[res.choice[0]]


def test_high_risk_confident_lots_keep_metaphylaxis_without_vet_override(world):
    _, cur, cfg, _, r, _ = world
    p = dict(param_values(cfg), amu_stewardship_weight=40.0)  # push hard toward withholding
    res = optimize(cur, r, p, cfg)
    strict = (r.tier == "High") & (r.risk_lo >= cfg["decision"]["high_risk_morbidity"])
    assert all("META" in c for c, s in zip(res.choice, strict) if s)
    res2 = optimize(cur, r, p, cfg, allow_high_stepdown=True)
    # any step-down allowed by the override must be flagged for sign-off
    for c, t, flag in zip(res2.choice, r.tier, res2.plan.vet_flag):
        if t == "High" and "META" not in c:
            assert flag == "HIGH-RISK STEP-DOWN"


def test_rider_hours_are_hard_even_when_capacity_zero(world):
    _, cur, cfg, _, r, _ = world
    zero = {y: {"rider_h_day": 0.0, "hosp_free": 0.0} for y in cfg["yards"]}
    res = optimize(cur, r, param_values(cfg), cfg, caps=zero)
    assert not any("ENH" in c for c in res.choice)
    rid = res.constraints[res.constraints.resource.str.startswith("Extra rider")]
    assert (rid["used"] <= rid["capacity"] + 1e-9).all()


def test_marginal_value_excludes_slack_penalty(world):
    _, cur, cfg, _, r, _ = world
    res = optimize(cur, r, param_values(cfg), cfg)
    mv = res.constraints["value_of_one_more_unit"].dropna()
    assert (mv < 5000).all()  # never the $10,000 slack penalty
    short = res.constraints[res.constraints.status == "SHORTFALL"]
    assert short["value_of_one_more_unit"].isna().all()  # a shortfall gets a note, not a fake dollar value


def test_text_in_numeric_column_is_reported_not_crashing(world):
    _, cur, *_ = world
    bad = cur.copy().astype({"arrival_weight_lb": object})
    bad.loc[0, "arrival_weight_lb"] = "720 lb"
    probs = riskmod.validate_lots(bad)
    assert any("non-numeric" in p for p in probs)


def test_editable_baseline_rules(world):
    _, cur, cfg, *_ = world
    a = baseline_plan(cur, cfg, "weight", 600)
    b = baseline_plan(cur, cfg, "weight_auction", 700)
    assert all((x == "META") == (w < 600) for x, w in zip(a, cur.arrival_weight_lb))
    assert all((x == "META") == (w < 700 and s == "auction")
               for x, w, s in zip(b, cur.arrival_weight_lb, cur.source_type))


def test_amu_cap_binds(world):
    _, cur, cfg, _, r, _ = world
    p = param_values(cfg)
    free = optimize(cur, r, p, cfg)
    cap = free.totals["Recommended plan"]["regimens"] * 0.9
    res = optimize(cur, r, p, cfg, amu_cap=cap)
    assert res.totals["Recommended plan"]["regimens"] <= cap + 1 or res.constraints["status"].eq("SHORTFALL").any()


# ---------- robustness thresholds ----------
def test_break_even_threshold_is_where_the_sign_flips(world):
    """At a reported deaths threshold the plan-minus-current difference changes sign (threshold analysis)."""
    from src.config import param_specs
    from src.decision import uncertainty as unc
    from src.decision.consequences import plan_totals
    _, cur, cfg, _, r, _ = world
    params, specs = param_values(cfg), param_specs(cfg)
    res = optimize(cur, r, params, cfg)
    p0 = r["risk_mean"].to_numpy()
    be = unc.break_even(cur, p0, res.choice, res.baseline, specs, params, cfg)
    assert set(be["key"]) == set(cfg["decision"]["robust_effect_keys"])
    for _, row in be.iterrows():
        thr = row["deaths_threshold"]
        if thr is None or thr != thr:
            continue
        width = row["high"] - row["low"]
        diffs = []
        for v in (thr - 0.05 * width, thr + 0.05 * width):
            prm = dict(params); prm[row["key"]] = v
            diffs.append(plan_totals(cur, p0, res.choice, prm)["deaths"] - plan_totals(cur, p0, res.baseline, prm)["deaths"])
        assert diffs[0] * diffs[1] < 0
        assert (diffs[1] < 0) == (row["deaths_plan_wins_when"] == "above")


def test_narrowing_result_is_robust_at_its_ranges(world):
    from src.config import param_specs
    from src.decision import uncertainty as unc
    _, cur, cfg, _, r, _ = world
    params, specs = param_values(cfg), param_specs(cfg)
    res = optimize(cur, r, params, cfg)
    p0 = r["risk_mean"].to_numpy()
    nar = unc.narrowing_needed(cur, p0, res.choice, res.baseline, specs, params, cfg)
    if nar["shrink"] is None:
        pytest.skip("plan not robust even at defaults in this fixture")
    sp = {**specs, **{k: {**specs[k], "low": lo, "high": hi} for k, (lo, hi) in nar["ranges"].items()}}
    assert unc.corner_check(cur, p0, res.choice, res.baseline, sp, params, cfg)["robust"]
    for k, (lo, hi) in nar["ranges"].items():
        assert specs[k]["low"] - 1e-9 <= lo <= params[k] <= hi <= specs[k]["high"] + 1e-9


def test_chute_check_reports_shortfall_without_changing_the_plan(world):
    """The processing-chute row is a reported feasibility check: a tiny chute flags a shortfall but the
    optimizer's choice is unchanged (it is not a constraint)."""
    import copy
    _, cur, cfg, _, r, _ = world
    params = param_values(cfg)
    a = optimize(cur, r, params, cfg)
    assert (a.constraints["resource"].str.startswith("Processing chute")).sum() == cur["yard"].nunique()
    cfg2 = copy.deepcopy(cfg)
    cfg2["processing"]["chute_hours_per_day"] = {y: 0.1 for y in cfg2["processing"]["chute_hours_per_day"]}
    b = optimize(cur, r, params, cfg2)
    chute = b.constraints[b.constraints["resource"].str.startswith("Processing chute")]
    assert (chute["status"] == "SHORTFALL").all()
    assert a.choice == b.choice
    assert chute["value_of_one_more_unit"].isna().all()


# ---------- individual-animal secondary view ----------
def test_animal_view_preserves_lot_risk_and_ranks_by_weight(world):
    from src.model.animals import animal_view
    _, cur, cfg, model, r, _ = world
    for i in range(3):
        lot = cur.iloc[i]
        a = animal_view(model, lot, float(r.iloc[i]["risk_mean"]), cfg)
        assert len(a) == int(lot["head_count"])
        assert abs(a["risk"].mean() - r.iloc[i]["risk_mean"]) < 1e-6          # lot expected cases unchanged
        assert a["risk"].between(0, 1).all()
        assert np.corrcoef(a["arrival_weight_lb"], a["risk"])[0, 1] < 0      # lighter head rank higher (sign-constrained)
        assert a["watch_first"].sum() == int(np.ceil(cfg["individual_view"]["watch_fraction"] * len(a)))
        b = animal_view(model, lot, float(r.iloc[i]["risk_mean"]), cfg)
        pd.testing.assert_frame_equal(a, b)                                     # deterministic per lot


def test_animal_view_handles_missing_weight(world):
    from src.model.animals import animal_view
    _, cur, cfg, model, r, _ = world
    lot = cur.iloc[0].copy()
    lot["arrival_weight_lb"] = np.nan
    assert animal_view(model, lot, 0.1, cfg) is None
