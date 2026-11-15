"""Validation-protocol acceptance tests (critic round 1)."""
import json

import pandas as pd

from src.config import ROOT
from src.model import risk as R
from tests.critic.simworld import fresh_lots


def test_evaluate_reports_calibration_slope_and_itl(hist, cfg):
    """D5: s5.1 promises calibration-in-the-large and calibration slope; evaluate() returns neither and uses a
    random split. Expected keys in R.evaluate(hist, cfg): 'cal_in_the_large', 'cal_slope', 'split'=='temporal'."""
    ev = R.evaluate(hist, cfg)
    for k in ("cal_in_the_large", "cal_slope", "split"):
        assert k in ev, f"missing {k}"
    assert ev["split"] == "temporal"


def test_baselines_reported_and_beaten(hist, cfg):
    """D6: B0/B1/B2 baselines promised, absent. Expected API: src.model.validation.baselines(train, test, cfg)
    -> {'B0': {'brier','auc'}, 'B1': {...}, 'B2': {...}, 'full': {...}}."""
    from src.model.validation import baselines, temporal_split
    tr, te = temporal_split(hist, cutoff="2026-01-01")
    b = baselines(tr, te, cfg)
    assert {"B0", "B1", "B2", "full"} <= set(b)
    assert b["full"]["brier"] < b["B1"]["brier"]


def test_oracle_recovery_api(model, cfg):
    """D7: s5.2 oracle recovery absent. Expected API: src.model.validation.oracle_recovery(model, cfg, n, seed)
    -> DataFrame by weight band [band, n, bias, rmse], computed against the MARGINAL p0 = E[p | x] (not the
    lot-conditional true p0, whose RMSE floor is ~0.16 because of the N(0, 1.3^2) lot effect)."""
    from src.model.validation import oracle_recovery
    out = oracle_recovery(model, cfg, n=1000, seed=4242)
    assert {"band", "n", "bias", "rmse"} <= set(out.columns)
    assert out["rmse"].max() < 0.10


# D8 single-dataset band-bias test retired in round 2 (rebuttal accepted): see
# tests/critic/test_replicates.py::test_untreated_risk_unbiased_across_histories.


def test_ccc_reported_with_noise_ceiling(hist, cfg):
    """D9: CCC/MSEP of observed morbidity vs predicted mean cannot reach 1 even for the TRUE model (lot effect +
    binomial noise), so it is uninterpretable without a ceiling. Expected: evaluate()['adequacy']['ccc_ceiling']
    (CCC of a perfect E[p|x] predictor, simulated) and head-count-weighted versions."""
    ev = R.evaluate(hist, cfg)
    assert "ccc_ceiling" in ev["adequacy"]
    assert ev["adequacy"]["ccc"] <= ev["adequacy"]["ccc_ceiling"] + 0.05


def test_ccc_ceiling_is_far_below_one(cfg):
    """D9 evidence (passes today): even the true marginal p0 has CCC well below 1 against observed morbidity."""
    f = fresh_lots(3000, seed=31337, cfg=cfg, meta_probs=(0.0, 0.0))
    ad = R.adequacy(f["morbidity_45d"].to_numpy(), f["_marg_p0"].to_numpy())
    print(f"CCC ceiling (true marginal p0 vs observed): {ad['ccc']:.3f}; random share {ad['share_random']:.2f}")
    assert ad["ccc"] < 0.75


def test_generator_frozen_with_manifest():
    """D10: s2 R2 promises a frozen generator, a recorded config hash and fresh-seed metrics.
    Expected: data/generator_manifest.json with keys {seed, config_hash, tuning_seeds, fresh_eval_seeds}
    and src.data.generate.config_hash() -> str equal to the manifest hash."""
    from src.data.generate import config_hash
    man = json.loads((ROOT / "data" / "generator_manifest.json").read_text())
    assert man["config_hash"] == config_hash()
    assert not set(man["tuning_seeds"]) & set(man["fresh_eval_seeds"])
