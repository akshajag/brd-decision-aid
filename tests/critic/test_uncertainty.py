"""Uncertainty-quantification acceptance tests (critic round 1)."""
import numpy as np

from src.config import param_values
from src.decision import followup as fu
from src.model import risk as R


def test_p_high_is_calibrated(fresh_assessed, fresh_untreated, cfg):
    """D1: P(>=15%) centres a logit-normal on logit(population-averaged mean); for p < 0.5 the mean of a
    logit-normal exceeds its median, so P(high) is inflated (today ~0.45 vs ~0.35 true on fresh lots).
    Acceptance: |mean p_high - freq(lot true p0 >= 15%)| < 0.03 on 1,000 fresh untreated lots."""
    r, _ = fresh_assessed
    thr = cfg["decision"]["high_risk_morbidity"]
    true_freq = float((fresh_untreated["_true_p0"] >= thr).mean())
    assert abs(r["p_high"].mean() - true_freq) < 0.03, (r["p_high"].mean(), true_freq)


# D2 single-dataset coverage test retired in round 2 (rebuttal accepted): see tests/critic/test_replicates.py
# (multi-history mean coverage and bootstrap-vs-sampling spread).


def test_point_estimate_is_the_mean(fresh_assessed):
    """D11: s6 says the objective uses the median of epistemic draws and that it differs from the mean by
    < 1 pp. Outcomes are linear in p, so the mean is the correct plug-in. Acceptance: max |median-mean| < 1 pp
    (or switch assess() to the mean)."""
    r, D = fresh_assessed
    assert np.max(np.abs(r["risk_mean"].to_numpy() - D.mean(axis=1))) < 0.01


def test_day0_and_checkin_agree(model, cur, cfg):
    """D12: the arrival P(>=15%) (risk.prob_high: p is the logit-normal CENTRE) and the check-in prior
    (followup._prior_particles: p is the particle MEAN) use incompatible definitions; with no pulls at day 0
    they should be identical. Today they differ by up to ~0.14. Acceptance: max |diff| < 0.02."""
    r, D = R.assess(model, cur, cfg, n_draws=40)
    upd = fu.update(cur, D, model.resid_sd, ["STD"] * len(cur), param_values(cfg), day=0,
                    pulls=np.full(len(cur), np.nan), n_particles=2000)
    assert np.max(np.abs(upd["p_high"].to_numpy() - r["p_high"].to_numpy())) < 0.02


def test_aleatoric_sd_not_understated(model, cfg):
    """D13: resid_sd is computed in-sample on the training fit (optimistic) and from a population-averaged
    model; it should be >= the DGP lot SD (the fitted model is misspecified, so residual spread can only be
    larger). Today 1.14 < 1.3. Acceptance: resid_sd >= 0.95 * dgp.lot_unobserved_sd (or estimate it by a
    proper random-intercept / beta-binomial fit, out of fold)."""
    assert model.resid_sd >= 0.95 * cfg["dgp"]["lot_unobserved_sd"], model.resid_sd
