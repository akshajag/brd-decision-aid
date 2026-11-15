"""Leakage and split-protocol acceptance tests (critic round 1)."""
import numpy as np
import pandas as pd

from src.model import risk as R


def test_prediction_path_forces_meta_zero(model, cur):
    """D0 (regression guard, passes today): a stray metaphylaxis column in uploads must not change risk."""
    a = model.predict_mean(cur)
    b = model.predict_mean(cur.assign(metaphylaxis_given=1))
    assert np.allclose(a, b)


def test_no_id_or_date_features(model):
    """D0 (regression guard): no lot_id / arrival_date / outcome columns in the design matrix."""
    bad = {"lot_id", "arrival_date", "brd_cases_45d", "morbidity_45d", "high_risk"}
    assert not bad & set(model.mu.index)


def test_preprocessing_is_fit_on_train_only(hist):
    """D0 (regression guard): medians/scaler depend only on the rows passed to fit()."""
    tr = hist[hist.arrival_date < "2026-01-01"]
    te = hist[hist.arrival_date >= "2026-01-01"].assign(transit_hours=999.0)
    m1 = R.fit(tr, n_boot=0)
    m2 = R.fit(pd.concat([tr, te]).iloc[: len(tr)], n_boot=0)
    assert m1.medians == m2.medians and np.allclose(m1.mu, m2.mu)


def test_temporal_split_exists_and_is_strict(hist):
    """D3: design s3/s5.1 make a temporal split PRIMARY; code only has a random split.
    Expected API: src.model.validation.temporal_split(hist, cutoff="2026-01-01") -> (train_df, test_df)."""
    from src.model.validation import temporal_split
    tr, te = temporal_split(hist, cutoff="2026-01-01")
    assert pd.to_datetime(tr.arrival_date).max() < pd.Timestamp("2026-01-01") <= pd.to_datetime(te.arrival_date).min()
    assert len(tr) + len(te) == len(hist)


def test_leave_one_yard_out(hist, cfg):
    """D3: LOYO promised as primary. Expected API:
    src.model.validation.leave_one_yard_out(hist, cfg, n_boot=0) -> DataFrame, one row per yard, columns
    [yard, n_test, auc, brier, cal_in_the_large, cal_slope]; all preprocessing refit inside each fold."""
    from src.model.validation import leave_one_yard_out
    res = leave_one_yard_out(hist, cfg, n_boot=0)
    assert set(res["yard"]) == {"A", "B", "C", "D"}
    for c in ("n_test", "auc", "brier", "cal_in_the_large", "cal_slope"):
        assert c in res.columns
    assert res["n_test"].sum() == len(hist)


def test_temporal_validation_can_fail():
    """D4: the DGP has no time term, so 'temporal validation' is a random split in disguise and cannot fail.
    Expected API: src.model.stress.generate_scenario(name, seed) -> (hist_df, eval_df), with a 'drift'
    scenario whose 2026 mean morbidity differs from 2024 by >= 3 percentage points."""
    from src.model.stress import generate_scenario
    h, _ = generate_scenario("drift", seed=101)
    yr = pd.to_datetime(h.arrival_date).dt.year
    assert abs(h.loc[yr == 2026, "morbidity_45d"].mean() - h.loc[yr == 2024, "morbidity_45d"].mean()) >= 0.03
