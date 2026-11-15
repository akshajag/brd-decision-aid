"""Critic probe: calibration of P(>=15%), epistemic-interval coverage, resid_sd, day-0 vs check-in coherence.

Run from repo root:  python -m scripts.critic.probe_uncertainty
"""
import time

import numpy as np
import pandas as pd

from src.config import ROOT, load_assumptions, param_values
from src.decision import followup as fu
from src.model import risk as R
from tests.critic.simworld import fresh_lots

t0 = time.time()
cfg = load_assumptions()
thr = cfg["decision"]["high_risk_morbidity"]
h = pd.read_csv(ROOT / "data/historical_cohorts.csv")
m = R.fit(h, n_boot=30)
print(f"[aleatoric] fitted resid_sd {m.resid_sd:.3f} vs DGP lot_unobserved_sd {cfg['dgp']['lot_unobserved_sd']} "
      f"(attenuated: population-averaged fit + residual misspecification)")

f = fresh_lots(1500, seed=777, cfg=cfg, meta_probs=(0.0, 0.0))
r, D = R.assess(m, f.drop(columns=[c for c in f.columns if c.startswith("_")] +
                          ["metaphylaxis_given", "brd_cases_45d", "morbidity_45d", "high_risk"]), cfg, n_draws=60)
ph = r.p_high.to_numpy()
cond_hi = (f._true_p0.to_numpy() >= thr)
obs_hi = (f.morbidity_45d.to_numpy() >= thr)
print(f"\n[P(>=15%) calibration] fresh untreated lots n={len(f)}: mean p_high {ph.mean():.3f}; "
      f"freq(true lot p0>=15%) {cond_hi.mean():.3f}; freq(observed morbidity>=15%) {obs_hi.mean():.3f}")
bins = pd.cut(ph, [0, .1, .3, .5, .7, .9, 1.0], include_lowest=True)
print(pd.DataFrame({"bin": bins, "p_high": ph, "true": cond_hi, "obs": obs_hi}).groupby("bin", observed=True)
      .agg(n=("p_high", "size"), p_high=("p_high", "mean"), true=("true", "mean"), obs=("obs", "mean")).round(3))

mp = f._marg_p0.to_numpy()
cover = np.mean((r.risk_lo.to_numpy() <= mp) & (mp <= r.risk_hi.to_numpy()))
width = np.median(r.risk_hi - r.risk_lo)
print(f"\n[epistemic 90% range] coverage of true marginal p0: {cover:.3f} (nominal 0.90); median width {100*width:.1f} pp")
miss = r.missing_critical.to_numpy() > 0
print(f"  width with missing critical fields {100*np.median((r.risk_hi-r.risk_lo)[miss]):.1f} pp vs complete "
      f"{100*np.median((r.risk_hi-r.risk_lo)[~miss]):.1f} pp")
print(f"  median(draws) - mean(draws): max |diff| {100*np.max(np.abs(np.median(D,1)-D.mean(1))):.2f} pp")
print(f"  confidence mix: {r.confidence.value_counts().to_dict()}")
print(f"  'confident' lots misclassified on High (vs marginal p0): "
      f"{np.mean(((r.risk_mean>=thr) != (mp>=thr))[r.confidence=='confident']):.3f}")

# day-0 P(high) vs check-in prior P(high) for the same lots, no pulls recorded
cur = pd.read_csv(ROOT / "data/current_arrivals.csv")
rc, Dc = R.assess(m, cur, cfg, n_draws=60)
upd = fu.update(cur, Dc, m.resid_sd, ["STD"] * len(cur), param_values(cfg), day=0,
                pulls=np.full(len(cur), np.nan))
diff = upd.p_high.to_numpy() - rc.p_high.to_numpy()
print(f"\n[coherence] P(>=15%) day-0 (assess) vs check-in prior at day 0, STD, no pulls: "
      f"max |diff| {np.max(np.abs(diff)):.3f}, mean diff {diff.mean():+.3f}")
print(pd.DataFrame({"lot": cur.lot_id, "assess_p_high": rc.p_high.round(3), "checkin_p_high": upd.p_high.round(3)}).head(8).to_string(index=False))
print(f"\n({time.time()-t0:.0f}s)")
