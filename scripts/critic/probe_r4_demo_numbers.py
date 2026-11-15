"""Critic round 4 probe: recompute the demo-case numbers quoted in TECHNICAL_BRIEF / DEMO_CASE from the default app path.
Run:  python -m scripts.critic.probe_r4_demo_numbers
"""
import warnings

import numpy as np
import pandas as pd

from src.config import ROOT, load_assumptions, param_specs, param_values
from src.decision import uncertainty as unc
from src.decision.consequences import plan_totals
from src.decision.optimize import optimize
from src.model import risk as R

warnings.filterwarnings("ignore")
cfg = load_assumptions()
P, specs = param_values(cfg), param_specs(cfg)
hist = pd.read_csv(ROOT / "data/historical_cohorts.csv")
cur = pd.read_csv(ROOT / "data/current_arrivals.csv")
m = R.fit(hist)
lots = R.prepare(cur).reset_index(drop=True)
risk, D = R.assess(m, lots, cfg)
print("head", int(lots.head_count.sum()), "| confidence", risk.confidence.value_counts().to_dict(),
      "| tiers", risk.tier.value_counts().to_dict(), "| drift", {k: m.drift.get(k) for k in ("z", "alarm")})
res = optimize(lots, risk, P, cfg)
pe = risk["risk_mean"].to_numpy()
pol = {"current": res.baseline, "recommended": res.choice, "meta_all": ["META"] * len(lots), "none": ["STD"] * len(lots)}
for k, ch in pol.items():
    t = plan_totals(lots, pe, ch, P)
    print(f"{k:12s} cases {t['cases']:.0f} deaths {t['deaths']:.1f} regimens {t['regimens']:.0f} cash ${t['cash_cost']:,.0f}")
chg = [i for i, (a, b) in enumerate(zip(res.baseline, res.choice)) if a != b]
from src.decision.consequences import OPTIONS
wd = [i for i in chg if OPTIONS[res.baseline[i]][1] == 1 and OPTIONS[res.choice[i]][1] == 0]
enh = [lots.lot_id[i] for i in range(len(lots)) if OPTIONS[res.choice[i]][2] == 1]
print("ENH lots", len(enh), enh)
print("choices", dict(zip(lots.lot_id, res.choice)))
print("changed lots", len(chg), "| meta withdrawn", [lots.lot_id[i] for i in wd], "head", int(lots.head_count[wd].sum()))
print("shortfalls", res.shortfalls)
mc = unc.compare(lots, D, {"rec": res.choice, "base": res.baseline}, specs, P, n=200)
s = unc.summarize(mc, "rec", "base")
print("p_cheaper %.2f p_no_more_deaths %.2f p_fewer_reg %.2f cost q %s" % (s["p_cheaper"], s["p_fewer_deaths"],
      s["p_fewer_regimens"], [round(x) for x in s["cost_diff_q"]]))
c = unc.corner_check(lots, pe, res.choice, res.baseline, specs, P, cfg)
print("verdict", unc.robust_verdict(s, c, cfg)[0])
print(c["corners"].round(1).to_string())
# CFR break-even
for v in (0.0, 0.03, 0.05, 0.07, 0.10):
    Q = dict(P, enhanced_monitoring_cfr_rrr=v)
    print(f"cfr_rrr {v:.2f}: deaths diff {plan_totals(lots, pe, res.choice, Q)['deaths'] - plan_totals(lots, pe, res.baseline, Q)['deaths']:+.2f}")
