"""Validation protocol of docs/DESIGN.md s5 (temporal and leave-one-yard-out splits, baselines, oracle recovery).

Everything here runs on SYNTHETIC data. Temporal and leave-one-yard-out numbers show the pipeline works and how it
degrades when it must extrapolate; they are NOT evidence of real-world accuracy (compare Rojas et al. 2022:
cohort AUC 0.68-0.79 on real feedlot data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.model import risk as R

BANDS = [0, 500, 600, 700, 800, 2000]


def temporal_split(hist: pd.DataFrame, cutoff: str = "2026-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train = arrivals before `cutoff`, test = arrivals on or after it. Strict and complete."""
    d = pd.to_datetime(hist["arrival_date"])
    c = pd.Timestamp(cutoff)
    return hist[d < c].copy(), hist[d >= c].copy()


def _score(model: R.RiskModel, te: pd.DataFrame, cfg: dict) -> dict:
    """Score a fitted model on held-out lots (observed post-treatment label, actual metaphylaxis status)."""
    te = R.prepare(te).reset_index(drop=True)
    thr = cfg["decision"]["high_risk_morbidity"]
    head = te["head_count"].to_numpy(float)
    cases = te["brd_cases_45d"].to_numpy(float)
    p = model.predict_mean(te, meta=te["metaphylaxis_given"].to_numpy(float))
    ph = R.prob_obs_high(p, head, model.resid_sd, thr)
    y = te["high_risk"].to_numpy()
    slope, _ = R._cal_slope(cases, head, p)
    return {"n_test": len(te),
            "auc": float(roc_auc_score(y, ph)) if 0 < y.mean() < 1 else float("nan"),
            "brier": float(brier_score_loss(y, ph)),
            "cal_in_the_large": float(cases.sum() / head.sum() - np.sum(p * head) / head.sum()),
            "cal_slope": slope}


def leave_one_yard_out(hist: pd.DataFrame, cfg: dict, n_boot: int = 0) -> pd.DataFrame:
    """Hold out each yard in turn; everything (medians, scaler, imputation donors, lot SD) is refit on the
    other three yards. The held-out yard's own effect is unknown to the model, as for a newly added yard."""
    rows = []
    for y in sorted(hist["yard"].unique()):
        tr, te = hist[hist["yard"] != y], hist[hist["yard"] == y]
        m = R.fit(tr, n_boot=n_boot)
        rows.append({"yard": y, **_score(m, te, cfg)})
    return pd.DataFrame(rows)


def baselines(train: pd.DataFrame, test: pd.DataFrame, cfg: dict) -> dict[str, dict]:
    """Brier/AUC for the observed high-risk label from simple baselines the full model must beat.

    B0 overall training prevalence; B1 logistic on arrival weight only (the current rule's information);
    B2 training prevalence by yard x source type; full = the decision aid's model.
    """
    y_tr = train["high_risk"].to_numpy()
    y = test["high_risk"].to_numpy()

    def sc(p):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        return {"brier": float(brier_score_loss(y, p)),
                "auc": float(roc_auc_score(y, p)) if np.ptp(p) > 0 else 0.5}
    out = {"B0": sc(np.full(len(test), y_tr.mean()))}
    lr = LogisticRegression(max_iter=1000).fit(train[["arrival_weight_lb"]].to_numpy(float) / 100, y_tr)
    out["B1"] = sc(lr.predict_proba(test[["arrival_weight_lb"]].to_numpy(float) / 100)[:, 1])
    grp = train.groupby(["yard", "source_type"])["high_risk"].mean()
    out["B2"] = sc([grp.get((a, b), y_tr.mean()) for a, b in zip(test["yard"], test["source_type"])])
    m = R.fit(train, n_boot=0)
    out["full"] = {k: v for k, v in _score(m, test, cfg).items() if k in ("brier", "auc")}
    return out


def oracle_recovery(model: R.RiskModel, cfg: dict, n: int = 1000, seed: int = 4242,
                    world: dict | None = None) -> pd.DataFrame:
    """Predicted untreated risk vs the TRUE E[p | recorded covariates] on fresh lots from the same synthetic world,
    by weight band and overall. Possible only because the data are synthetic (DESIGN s5.2)."""
    from src.data.generate import simulate_world
    f = simulate_world(n, seed, cfg, world=world, meta_probs=(0.0, 0.0))
    d = model.predict_mean(f) - f["_marg_p0"].to_numpy()
    band = pd.cut(f["arrival_weight_lb"], BANDS, right=False).astype(str)
    g = pd.DataFrame({"band": band, "d": d})
    rows = [{"band": b, "n": len(x), "bias": x["d"].mean(), "rmse": float(np.sqrt((x["d"] ** 2).mean()))}
            for b, x in g.groupby("band", sort=False)]
    rows.append({"band": "all", "n": len(g), "bias": g["d"].mean(), "rmse": float(np.sqrt((g["d"] ** 2).mean()))})
    order = {str(pd.Interval(a, b, closed="left")): i for i, (a, b) in enumerate(zip(BANDS[:-1], BANDS[1:]))}
    return pd.DataFrame(rows).sort_values("band", key=lambda s: s.map(lambda v: order.get(v, 99))).reset_index(drop=True)


def positivity(hist: pd.DataFrame) -> pd.DataFrame:
    """Treated share and untreated lot count by weight band (support for the untreated-equivalent prediction)."""
    band = pd.cut(hist["arrival_weight_lb"], BANDS, right=False).astype(str)
    g = hist.assign(band=band).groupby("band", sort=False)["metaphylaxis_given"]
    return pd.DataFrame({"lots": g.size(), "treated_share": g.mean().round(3),
                         "untreated_lots": g.apply(lambda s: int((s == 0).sum()))}).reset_index()
