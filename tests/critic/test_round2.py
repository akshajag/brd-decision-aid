"""Critic round 2: leakage canaries, reproducibility, invariance, monotonicity and stability tests on the shipped data.
Fault IDs refer to docs/CRITIC_LOG.md (Round 2)."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.config import ROOT, param_values
from src.decision.optimize import optimize
from src.model import risk as R
from src.model.validation import _score, temporal_split


def test_evaluate_ignores_test_outcomes(hist, cfg):
    """Leakage canary (guard). Scrambling the OUTCOMES of the 2026 test lots must not change a single prediction
    that evaluate() makes for them (fit, medians, donors and the out-of-fold lot SD use training rows only)."""
    a = R.evaluate(hist, cfg)["obs_pred"]["pred"].to_numpy()
    h2 = hist.copy()
    te = (pd.to_datetime(h2["arrival_date"]) >= pd.Timestamp("2026-01-01")).to_numpy()
    rng = np.random.default_rng(0)
    cases = rng.permutation(h2.loc[te, "brd_cases_45d"].to_numpy())
    h2.loc[te, "brd_cases_45d"] = np.minimum(cases, h2.loc[te, "head_count"].to_numpy())
    h2.loc[te, "morbidity_45d"] = h2.loc[te, "brd_cases_45d"] / h2.loc[te, "head_count"]
    h2.loc[te, "high_risk"] = (h2.loc[te, "morbidity_45d"] >= 0.15).astype(int)
    b = R.evaluate(h2, cfg)["obs_pred"]["pred"].to_numpy()
    assert np.allclose(a, b)


def test_permutation_canary_auc_collapses(hist, cfg):
    """Leakage canary (guard). Break the link between arrival data and outcome (permute morbidity across lots, then
    rebuild cases from each lot's own head count). The temporal hold-out AUC must fall to chance (0.5 +/- 0.1);
    any pipeline path that lets outcome information reach predictions would keep it high."""
    h = hist.copy()
    morb = np.random.default_rng(1).permutation(h["morbidity_45d"].to_numpy())
    h["brd_cases_45d"] = np.round(morb * h["head_count"]).astype(int)
    h["morbidity_45d"] = h["brd_cases_45d"] / h["head_count"]
    h["high_risk"] = (h["morbidity_45d"] >= 0.15).astype(int)
    tr, te = temporal_split(h)
    auc = _score(R.fit(tr, n_boot=0), te, cfg)["auc"]
    assert abs(auc - 0.5) < 0.1, auc


_DET = ("import hashlib,warnings;warnings.filterwarnings('ignore');import numpy as np,pandas as pd;"
        "from src.config import ROOT,load_assumptions;from src.model import risk as R;"
        "h=pd.read_csv(ROOT/'data/historical_cohorts.csv');c=pd.read_csv(ROOT/'data/current_arrivals.csv');"
        "m=R.fit(h,n_boot=2,seed=0);r,D=R.assess(m,c,load_assumptions(),n_draws=6);"
        "print(hashlib.sha256(np.round(np.r_[D.ravel(),m.resid_sd],10).tobytes()).hexdigest())")


def test_fit_is_deterministic_across_hash_seeds():
    """R2-5 (builder-reported bug, fixed by sorting the (yard, source) groups in _donor_fill). Same inputs in two
    fresh interpreters with different PYTHONHASHSEED must give bit-identical risk draws and lot SD."""
    outs = []
    for hs in ("1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hs}
        r = subprocess.run([sys.executable, "-c", _DET], cwd=ROOT, env=env, capture_output=True, text=True,
                           timeout=300)
        assert r.returncode == 0, r.stderr[-2000:]
        outs.append(r.stdout.strip().splitlines()[-1])
    assert outs[0] == outs[1], outs


def test_recommendation_invariant_to_row_order_and_lot_id(model, cur, cfg):
    """R2-4. A lot's risk and option must not depend on where it sits in the upload or what it is called. Today the
    donor-imputation RNG is consumed row by row, so permuting rows moves a lot's risk_mean by up to ~0.6 pp (option
    choices happened not to change on the demo week). Acceptance: max |delta risk_mean| < 0.2 pp and identical options.
    Fix: per-lot RNG streams keyed on the lot's covariates (not lot_id), or enough imputations that MC error < 0.1 pp."""
    P = param_values(cfg)
    r0, _ = R.assess(model, cur, cfg)
    ch0 = optimize(cur, r0, P, cfg).choice
    perm = np.random.default_rng(1).permutation(len(cur))
    cp = cur.iloc[perm].reset_index(drop=True).copy()
    cp["lot_id"] = [f"Z{i:03d}" for i in range(len(cp))]
    rp, _ = R.assess(model, cp, cfg)
    chp = optimize(cp, rp, P, cfg).choice
    d = np.abs(rp["risk_mean"].to_numpy() - r0["risk_mean"].to_numpy()[perm]).max()
    assert [ch0[i] for i in perm] == list(chp)
    assert d < 0.002, d


def test_risk_monotone_in_literature_signed_factors(model, cur):
    """R2-3 (face validity). Literature and the simulator agree on direction: more commingled sources, longer transit,
    more shrink, less pen space and lighter weight never LOWER BRD risk. On the shipped model +5 sources lowers risk
    for some lots (fitted coefficient -0.009/source vs DGP +0.06; 35% of bootstrap members negative) and -5 m2/head of
    pen space lowers risk (positive coefficient in 95% of members vs the DGP crowding penalty). A judge who types
    'more sources' in the live demo sees risk fall.
    Acceptance: each perturbation changes predicted risk by >= -0.1 pp for every demo lot.
    Fix: sign-constrained coefficients for literature-signed factors (or drop factors whose bootstrap CI spans 0 from
    the model and from the 'drivers' text), and never show a driver whose sign contradicts the literature."""
    for col, delta in (("arrival_weight_lb", -100), ("transit_hours", 10), ("n_sources", 5), ("shrink_pct", 3),
                       ("pen_area_m2_per_head", -5)):
        c2 = cur.copy()
        c2[col] = pd.to_numeric(c2[col]) + delta
        d = model.predict_mean(c2) - model.predict_mean(cur)
        assert np.nanmin(d) >= -0.001, (col, delta, round(float(np.nanmin(d)), 4))


def test_interval_uses_distinct_bootstrap_members(model, cur, cfg):
    """R2-6. assess() takes 60 draws by default but the ensemble has 30 members, cycled; for a lot with complete records
    the 60 draws are 30 values each repeated twice. The Weibull 5-95% range of that duplicated set is about
    [2nd, 29th] of 30 members, whose expected coverage is ~0.87, not 0.90 (simulated: 0.871). Acceptance: for
    complete-record lots, the number of distinct draws equals n_draws (n_draws <= n_boot, or members not reused),
    so the interval level is what the label says."""
    _, D = R.assess(model, cur, cfg)
    comp = ~cur[R.IMPUTABLE].isna().any(axis=1).to_numpy()
    n_unique = [len(np.unique(np.round(D[i], 12))) for i in np.flatnonzero(comp)]
    assert min(n_unique) == D.shape[1], (min(n_unique), D.shape[1])


def test_borderline_lots_not_left_on_standard_care(model, cfg):
    """R2-13 (rule 6 + rule 7). The care floor protects only lots whose point estimate is High. Lots flagged
    'borderline' (90% range crosses 15%) can still get standard care only, and they make up most truly-high lots left
    on STD: over 40 fresh weeks 7 of 436 truly-high lots got STD (1.6%, the D17 guard now fails), and 5 of the 7
    were flagged borderline by the tool itself. Acceptance: on 12 fresh weeks no lot with risk_hi >= 15% gets STD
    (at least enhanced monitoring, or escalation if rider hours bind). Fix: extend the floor to borderline lots."""
    P = param_values(cfg)
    from tests.critic.conftest import arrivals_only
    from tests.critic.simworld import fresh_lots
    thr = cfg["decision"]["high_risk_morbidity"]
    bad = []
    for w in range(12):
        f = fresh_lots(24, seed=60_000 + w, cfg=cfg)
        lots = arrivals_only(f).reset_index(drop=True)
        r, _ = R.assess(model, lots, cfg, n_draws=12)
        ch = optimize(lots, r, P, cfg).choice
        bad += [(w, round(float(r.risk_mean[i]), 3), round(float(r.risk_hi[i]), 3)) for i in range(len(lots))
                if r.risk_hi[i] >= thr and ch[i] == "STD"]
    assert not bad, bad


@pytest.mark.slow
def test_recommendation_stable_across_bootstrap_seeds(hist, cur, cfg):
    """R2-7. With app defaults (n_boot=30, 60 draws), changing only the bootstrap seed should not change the demo
    plan materially. Round-2 probe (n_boot=20): 4, 3 and 1 of 24 lots changed option and up to 4 lots changed
    confidence status for seeds 1-3 vs 0; one lot's risk moved 3.5 pp. Acceptance: for seeds 1-3 vs 0, at most 1 lot
    changes option and max |delta risk_mean| <= 1 pp. Fix: more bootstrap members (>= 200 as asked in round 1) or
    report the Monte Carlo error and treat options within it as ties."""
    P = param_values(cfg)
    base = None
    res = []
    for s in (0, 1, 2, 3):
        m = R.fit(hist, seed=s)
        r, _ = R.assess(m, cur, cfg)
        ch = optimize(cur, r, P, cfg).choice
        if base is None:
            base = (r, ch)
            continue
        res.append((sum(a != b for a, b in zip(base[1], ch)),
                    round(float(np.abs(r["risk_mean"] - base[0]["risk_mean"]).max()), 4)))
    assert all(f <= 1 and d <= 0.01 for f, d in res), res


@pytest.mark.slow
def test_drift_check_operating_characteristics(cfg):
    """Round 3 (R2-8 fix). check_drift must (a) rarely alarm on histories WITHOUT drift and (b) alarm on the drift world
    (+0.35 logit/yr). Round-3 probe (scripts/critic/probe_r3_drift.py): false-alarm 1/30 base, 0/15 S2, 0/15 S1;
    detection 10/10 at 0.35/yr but only 1/10 at 0.15/yr (see R3-2). Acceptance here: <= 1 alarm in 8 null histories
    and alarms in all 3 drift histories; the recalibration offset is positive under drift."""
    from src.data.generate import HIST_DATES, HIST_WEIGHTS, simulate_world
    from src.model.stress import SCENARIOS, split

    def run(world, seeds):
        out = []
        for s in seeds:
            h = simulate_world(1400, s, cfg, world=world, dates=HIST_DATES, weights=HIST_WEIGHTS, prefix="H")
            m = R.fit(h, n_boot=0, n_point=1, drift_check=False)
            out.append(R.check_drift(m, h, np.random.default_rng(s)))
        return out
    null = run({}, range(78_000, 78_008))
    drift = run(split(SCENARIOS["drift"])[0], range(79_000, 79_003))
    assert sum(d["alarm"] for d in null) <= 1, [round(d["z"], 2) for d in null]
    assert all(d["alarm"] and d["offset"] > 0 for d in drift), [(round(d["z"], 2), round(d["offset"], 3)) for d in drift]


def test_headline_worst_case_reports_worst_health_outcome():
    """R3-1. The report (scripts/run_validation.render) and the app pick the 'most adverse scenario' by the lowest saving
    only. On the round-3 results that is S8 (saving -$4,880/week but deaths -2.34/week), while S7 has +4.49 deaths/week
    and a loss in 80% of weeks, and S7 is absent from the headline. Health comes first in this brief (rule 6, consequence
    dimensions reported together). Acceptance: the headline section of the rendered report mentions the d_deaths value
    of the scenario with the MOST extra deaths. (The app's v4 metric and warning use the same min-saving rule: fix both.)"""
    import json
    from scripts.run_validation import render
    r = json.loads((ROOT / "reports" / "validation_results.json").read_text(encoding="utf-8"))
    head = render(r).split("## 2.")[0]
    wd = max(r["stress"], key=lambda x: x["d_deaths"])
    assert f"{wd['d_deaths']:+.2f}" in head, (wd["scenario"], wd["d_deaths"])


def test_duplicate_key_lots_invariant_to_row_order(model, cur, cfg):
    """R4-2. R3-4's fix gives duplicate-key lots an occurrence index (first = bare key, second = key#1, ...). Lots that
    share the hashed fields (yard, source, sex, vaccination, quarter, weather, head count) but differ elsewhere (weight,
    which fields are missing) then swap random streams when their rows swap, so a lot's risk depends on row order again
    (the R2-4 property) for exactly these lots. Acceptance: two such lots keep their risk_mean (within 0.2 pp) when
    their order is reversed. Fix: break ties with the lot's own observed (non-what-if) numeric fields, e.g. arrival
    weight and the missingness pattern, instead of an occurrence index; only fully identical rows need an index."""
    worst = (0.0, "")   # builder fix: (0.0, None) raised TypeError on exact ties
    for lid in cur.loc[cur[R.CRITICAL].isna().any(axis=1), "lot_id"]:
        a = cur[cur["lot_id"] == lid].copy()
        b = a.copy()
        b["arrival_weight_lb"] = b["arrival_weight_lb"] + 150
        b["lot_id"] = "DUP2"
        ra, _ = R.assess(model, pd.concat([a, b], ignore_index=True), cfg)
        rb, _ = R.assess(model, pd.concat([b, a], ignore_index=True), cfg)
        d = max(abs(ra.risk_mean[0] - rb.risk_mean[1]), abs(ra.risk_mean[1] - rb.risk_mean[0]))
        worst = max(worst, (float(d), lid))
    # round-4 probe (n_boot=30 model): C0017 moves 2.3 pp, C0007 1.0 pp, C0010 0.6 pp
    assert worst[0] < 0.002, worst


def test_docs_corner_claim_matches_corner_table(model, cur, cfg):
    """R4-1. TECHNICAL_BRIEF ('The plan loses only at one corner'), DEMO_CASE ('At one corner the plan loses') and the
    validation report ('the plan always loses at one corner') say the demo plan fails only where metaphylaxis works
    better than assumed AND extra riding does nothing. The app's own corner table for the demo week shows the plan has
    MORE DEATHS at 3 of 8 corners: (0.25, 0, 0) +1.5, (0.65, 0, 0) +3.9 and (0.65, 0, 0.25) +0.3; only the cost loss
    is unique to one corner. So the death risk does not need metaphylaxis to work better than assumed: it appears
    whenever extra riding has no effect on case fatality. A judge can read this off the What-if tab.
    Acceptance: if more than one corner loses on cost or deaths, none of the single-corner phrases appear."""
    from src.config import param_specs, param_values
    from src.decision import uncertainty as unc
    P = param_values(cfg)
    lots = R.prepare(cur).reset_index(drop=True)
    r, _ = R.assess(model, lots, cfg)
    res = optimize(lots, r, P, cfg)
    c = unc.corner_check(lots, r["risk_mean"].to_numpy(), res.choice, res.baseline, param_specs(cfg), P, cfg)["corners"]
    n_lose = int(((c["cost_diff"] > 0) | (c["deaths_diff"] > 1e-9)).sum())
    phrases = ["loses only at one corner", "At one corner the plan loses", "always loses at one corner"]
    texts = {f: (ROOT / f).read_text(encoding="utf-8") for f in
             ("docs/TECHNICAL_BRIEF.md", "docs/DEMO_CASE.md", "docs/VALIDATION_REPORT.md")}
    found = [(f, ph) for f, t in texts.items() for ph in phrases if ph in t]
    assert n_lose <= 1 or not found, (n_lose, found)


def test_docs_death_corner_claim_has_no_counterexample(model, cur, cfg):
    """R5-1. After R4-1 the documents say deaths rise 'whenever extra riding does nothing ... whatever metaphylaxis
    does'. The demo corner table has a counterexample: metaphylaxis 0.25, extra riding 0, receiving 0.25 gives
    deaths -4.7 (fewer deaths). Deaths rise when the enhanced-monitoring BUNDLE (riding + low-stress receiving) does
    little, not whenever riding alone does nothing. Acceptance: if any corner with zero riding effect has fewer deaths,
    the 'whenever / whatever metaphylaxis does' phrasing is absent from the brief, DEMO_CASE, report and overview."""
    from src.config import param_specs, param_values
    from src.decision import uncertainty as unc
    P = param_values(cfg)
    lots = R.prepare(cur).reset_index(drop=True)
    r, _ = R.assess(model, lots, cfg)
    res = optimize(lots, r, P, cfg)
    c = unc.corner_check(lots, r["risk_mean"].to_numpy(), res.choice, res.baseline, param_specs(cfg), P, cfg)["corners"]
    counter = bool(((c["enhanced_monitoring_cfr_rrr"] == c["enhanced_monitoring_cfr_rrr"].min())
                    & (c["deaths_diff"] < 0)).any())
    phrases = ["whatever metaphylaxis does", "whenever extra riding does nothing",
               "whenever extra riding does not lower case fatality"]
    files = ("docs/TECHNICAL_BRIEF.md", "docs/DEMO_CASE.md", "docs/VALIDATION_REPORT.md", "SUBMISSION_OVERVIEW.md")
    found = [(f, ph) for f in files for ph in phrases if ph.lower() in (ROOT / f).read_text(encoding="utf-8").lower()]
    assert not counter or not found, found


def test_lot_risk_independent_of_other_lots_in_upload(model, cur, cfg):
    """R5-2. After R4-2, a lot's imputation key gains weight + missing pattern only when ANOTHER lot in the upload shares
    its base key. Adding an unrelated lot can therefore change an existing lot's imputations and risk (batch
    dependence). Acceptance: adding a colliding lot (same categorical fields and head count, +150 lb) leaves the
    original lot's risk_mean unchanged within 0.2 pp, for every demo lot with missing critical fields.
    Fix: always build the key from the lot's own fields (categoricals, head count, weight, missing pattern); this
    changes demo numbers slightly, so regenerate them, or accept and document the rare batch dependence."""
    base, _ = R.assess(model, cur, cfg)
    worst = (0.0, "")
    for lid in cur.loc[cur[R.CRITICAL].isna().any(axis=1), "lot_id"]:
        extra = cur[cur["lot_id"] == lid].copy()
        extra["arrival_weight_lb"] = extra["arrival_weight_lb"] + 150
        extra["lot_id"] = "EXTRA"
        r2, _ = R.assess(model, pd.concat([cur, extra], ignore_index=True), cfg)
        i = int(np.flatnonzero(cur["lot_id"].to_numpy() == lid)[0])
        worst = max(worst, (abs(float(r2.risk_mean[i] - base.risk_mean[i])), lid))
    assert worst[0] < 0.002, worst
