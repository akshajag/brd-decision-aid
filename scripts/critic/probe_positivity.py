"""Critic probe: positivity, confounding adjustment, oracle recovery of p0 by weight band.

Run from repo root:  python -m scripts.critic.probe_positivity
"""
import numpy as np
import pandas as pd

from src.config import ROOT, load_assumptions
from src.model import risk as R
from tests.critic.simworld import fresh_lots

cfg = load_assumptions()
h = pd.read_csv(ROOT / "data/historical_cohorts.csv")
bands = pd.cut(h.arrival_weight_lb, [0, 500, 600, 700, 800, 2000], right=False)
t = h.groupby(bands, observed=True).agg(n=("lot_id", "size"), treated=("metaphylaxis_given", "mean"),
                                        untreated_n=("metaphylaxis_given", lambda s: int((s == 0).sum())))
print("[positivity] treated share by weight band (historical CSV)")
print(t.round(3))
print(f"  yard B untreated lots: {int(((h.yard=='B') & (h.metaphylaxis_given==0)).sum())} of {int((h.yard=='B').sum())}")

m = R.fit(h, n_boot=0)
cols = list(m.mu.index)
b_meta = m.coefs[0][cols.index("metaphylaxis_given")] / m.sd["metaphylaxis_given"]
rrr = cfg["interventions"]["metaphylaxis_morbidity_rrr"]["value"]
print(f"\n[adjustment] fitted metaphylaxis logit coef {b_meta:.3f}; DGP is multiplicative RRR={rrr} "
      f"(logit-equivalent at p=0.10: {np.log(0.10*(1-rrr)/(1-0.10*(1-rrr))) - np.log(0.10/0.90):.3f}, "
      f"at p=0.30: {np.log(0.30*(1-rrr)/(1-0.30*(1-rrr))) - np.log(0.30/0.70):.3f})")
print(f"  implied RRR at mean p: {1 - np.exp(b_meta):.3f}  (planner assumes {rrr}; the generator uses the SAME config value)")

f = fresh_lots(4000, seed=9001, cfg=cfg, meta_probs=(0.0, 0.0), dates=pd.date_range('2024-01-01','2026-10-31'),
               weights={"A": 0.32, "B": 0.26, "C": 0.24, "D": 0.18})
p = m.predict_mean(f)  # meta=0: untreated-equivalent
fb = pd.cut(f.arrival_weight_lb, [0, 500, 600, 700, 800, 2000], right=False)
rec = pd.DataFrame({"band": fb, "pred": p, "marg_p0": f._marg_p0, "cond_p0": f._true_p0, "obs": f.morbidity_45d})
g = rec.groupby("band", observed=True).agg(n=("pred", "size"), pred=("pred", "mean"), marg_p0=("marg_p0", "mean"),
                                            obs_untreated=("obs", "mean"))
g["bias_pp"] = 100 * (g.pred - g.marg_p0)
print("\n[oracle recovery] fresh untreated lots (seed 9001, n=4000): predicted p0 vs true marginal p0")
print(g.round(4))
print(f"  RMSE vs marginal p0: {np.sqrt(np.mean((p - f._marg_p0)**2)):.4f};  vs conditional p0 (incl. lot effect): "
      f"{np.sqrt(np.mean((p - f._true_p0)**2)):.4f}")
hi_true = f._marg_p0 >= 0.15
hi_pred = p >= 0.15
print(f"  High (p0>=15%) prevalence: true-marginal {hi_true.mean():.3f}, predicted {hi_pred.mean():.3f}; "
      f"misclassified {np.mean(hi_true != hi_pred):.3f}")
