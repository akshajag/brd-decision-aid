"""Critic probe: split protocols, stationarity, yard confounding, baselines (real CSV).

Run from repo root:  python -m scripts.critic.probe_leakage
"""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.config import ROOT, load_assumptions
from src.model import risk as R

cfg = load_assumptions()
thr = cfg["decision"]["high_risk_morbidity"]
h = pd.read_csv(ROOT / "data/historical_cohorts.csv", parse_dates=["arrival_date"])
print(f"n={len(h)}  dates {h.arrival_date.min().date()}..{h.arrival_date.max().date()}")

print("\n[stationarity] morbidity / high-risk by year (no time term in DGP => temporal split == random split)")
print(h.groupby(h.arrival_date.dt.year)[["morbidity_45d", "high_risk", "metaphylaxis_given"]].mean().round(3))


def score(tr, te, label):
    m = R.fit(tr, n_boot=0)
    p = m.predict_mean(te, meta=te["metaphylaxis_given"].to_numpy(float))
    ph = m.prob_high(p, thr)
    y = te["high_risk"].to_numpy()
    auc = roc_auc_score(y, ph) if 0 < y.mean() < 1 else np.nan
    return dict(split=label, n_tr=len(tr), n_te=len(te), auc=round(auc, 3),
                brier=round(brier_score_loss(y, ph), 4), cal_itl=round(te["morbidity_45d"].mean() - p.mean(), 4))


rows = [score(h[h.arrival_date < "2026-01-01"], h[h.arrival_date >= "2026-01-01"], "temporal")]
for y in "ABCD":
    rows.append(score(h[h.yard != y], h[h.yard == y], f"LOYO-{y}"))
rng = np.random.default_rng(1)
te = rng.random(len(h)) < 0.25
rows.append(score(h[~te], h[te], "random"))
print("\n[splits] same pipeline, different protocols")
print(pd.DataFrame(rows).to_string(index=False))

print("\n[yard x weight confounding] share of light (<700 lb) lots and treated share by yard")
print(h.assign(light=h.arrival_weight_lb < 700).groupby("yard")[["light", "metaphylaxis_given"]].mean().round(3))

print("\n[baselines, temporal split, Brier on high_risk]")
tr, te = h[h.arrival_date < "2026-01-01"], h[h.arrival_date >= "2026-01-01"]
y = te.high_risk.to_numpy()
b0 = np.full(len(te), tr.high_risk.mean())
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression().fit(tr[["arrival_weight_lb"]], tr.high_risk)
b1 = lr.predict_proba(te[["arrival_weight_lb"]])[:, 1]
g = tr.groupby(["yard", "source_type"]).high_risk.mean()
b2 = te.set_index(["yard", "source_type"]).index.map(g).to_numpy(float)
b2 = np.where(np.isnan(b2), tr.high_risk.mean(), b2)
m = R.fit(tr, n_boot=0)
full = m.prob_high(m.predict_mean(te, meta=te.metaphylaxis_given.to_numpy(float)), thr)
for k, v in dict(B0=b0, B1_weight=b1, B2_yard_source=b2, full=full).items():
    print(f"  {k:15s} Brier {brier_score_loss(y, v):.4f}  AUC {roc_auc_score(y, v) if v.std() > 0 else float('nan'):.3f}")

print("\n[label/treatment] high_risk label is OBSERVED (post-metaphylaxis) morbidity >= 15%")
print(h.groupby("metaphylaxis_given")[["morbidity_45d", "high_risk"]].mean().round(3))
