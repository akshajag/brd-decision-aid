"""Critic probe: optimizer's curse, regret vs oracle, effect-size misspecification (S4), MC verdict optimism.

Run from repo root:  python -m scripts.critic.probe_decision_regret [n_weeks]
"""
import sys
import time

import numpy as np
import pandas as pd

from src.config import ROOT, load_assumptions, param_specs, param_values
from src.decision.consequences import plan_totals
from src.decision.optimize import optimize
from src.decision import uncertainty as U
from src.model import risk as R
from tests.critic.simworld import fresh_lots

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
t0 = time.time()
cfg = load_assumptions()
P = param_values(cfg)
thr = cfg["decision"]["high_risk_morbidity"]
h = pd.read_csv(ROOT / "data/historical_cohorts.csv")
m = R.fit(h, n_boot=10)
OUT = ["metaphylaxis_given", "brd_cases_45d", "morbidity_45d", "high_risk"]


def oracle_risk(lots, p):
    return pd.DataFrame({"lot_id": lots.lot_id, "yard": lots.yard, "head_count": lots.head_count,
                         "arrival_weight_lb": lots.arrival_weight_lb, "tier": np.where(p >= thr, "High", "Low"),
                         "confidence": "confident", "risk_mean": p, "risk_lo": p, "risk_hi": p, "drivers": ""})


rows, verdict_rows = [], []
for w in range(N):
    f = fresh_lots(24, seed=50_000 + w, cfg=cfg)
    lots = f.drop(columns=[c for c in f.columns if c.startswith("_")] + OUT).reset_index(drop=True)
    r, D = R.assess(m, lots, cfg, n_draws=20)
    res = optimize(lots, r, P, cfg)
    est_p, true_p, marg_p = r.risk_mean.to_numpy(), f._true_p0.to_numpy(), f._marg_p0.to_numpy()
    orc = optimize(lots, oracle_risk(lots, true_p), P, cfg).choice
    bay = optimize(lots, oracle_risk(lots, marg_p), P, cfg).choice
    P4 = dict(P, metaphylaxis_morbidity_rrr=0.25)
    P6 = dict(P, metaphylaxis_morbidity_rrr=0.65, enhanced_monitoring_cfr_rrr=0.0, receiving_protocol_morbidity_rrr=0.0)
    for name, prm in (("base", P), ("S4_true_rrr_0.25", P4), ("S4b_rrr0.65_noENHeffect", P6)):
        tb = lambda ch, p: plan_totals(lots, p, ch, prm)
        rec_est = plan_totals(lots, est_p, res.choice, P)["objective"]
        base_est = plan_totals(lots, est_p, res.baseline, P)["objective"]
        rows.append(dict(week=w, world=name,
                         est_saving=base_est - rec_est,
                         real_saving=tb(res.baseline, true_p)["objective"] - tb(res.choice, true_p)["objective"],
                         regret_vs_clairvoyant=tb(res.choice, true_p)["objective"] - tb(orc, true_p)["objective"],
                         regret_vs_bayes=tb(res.choice, marg_p)["objective"] - tb(bay, marg_p)["objective"],
                         real_d_deaths=tb(res.choice, true_p)["deaths"] - tb(res.baseline, true_p)["deaths"],
                         floor_viol=int(sum((mp >= thr) and c == "STD" for mp, c in zip(marg_p, res.choice)))))
    if w < 10:
        mc = U.compare(lots, D, {"rec": res.choice, "base": res.baseline}, param_specs(cfg), P, n=150)
        s = U.summarize(mc, "rec", "base")
        real_cheaper = plan_totals(lots, true_p, res.choice, P)["cash_cost"] < plan_totals(lots, true_p, res.baseline, P)["cash_cost"]
        verdict_rows.append(dict(p_cheaper=s["p_cheaper"], confident=U.verdict(s)[0], real_cheaper=real_cheaper))

df = pd.DataFrame(rows)
print(f"[regret study] {N} fresh weeks x 24 lots (seeds 50000+), model fit on historical CSV")
print(df.groupby("world")[["est_saving", "real_saving", "regret_vs_clairvoyant", "regret_vs_bayes",
                            "real_d_deaths", "floor_viol"]].mean().round(1).to_string())
b = df[df.world == "base"]
print(f"\n  optimizer's curse (base): estimated saving {b.est_saving.mean():.0f} vs realised {b.real_saving.mean():.0f} USD/week "
      f"(ratio {b.real_saving.mean()/b.est_saving.mean():.2f}); weeks where realised saving < 0: {(b.real_saving<0).mean():.2f}")
s4 = df[df.world == "S4_true_rrr_0.25"]
print(f"  S4 (true RRR 0.25, planner 0.45): realised saving {s4.real_saving.mean():.0f}; weeks < 0: {(s4.real_saving<0).mean():.2f}; "
      f"extra deaths/week vs baseline {s4.real_d_deaths.mean():+.2f}")
v = pd.DataFrame(verdict_rows)
print(f"\n[MC verdict] first 10 weeks: mean P(cheaper) {v.p_cheaper.mean():.2f}; 'confident' {v.confident.mean():.2f}; "
      f"actually cheaper on truth {v.real_cheaper.mean():.2f}")
print(f"({time.time()-t0:.0f}s)")
s6 = df[df.world == "S4b_rrr0.65_noENHeffect"]
print(f"  S4b (true RRR 0.65, ENH effects 0): realised saving {s6.real_saving.mean():.0f}; weeks < 0: {(s6.real_saving<0).mean():.2f}; "
      f"extra deaths/week vs baseline {s6.real_d_deaths.mean():+.2f}")
