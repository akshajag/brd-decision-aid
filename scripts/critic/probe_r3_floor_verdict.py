"""Critic round 3 probe: cost of the borderline floor (R2-13 fix) and how often the three-state verdict says 'robust'.

For fresh base-world weeks: plan with and without the borderline floor (true-world expected cost, deaths, regimens,
truly-high lots on STD), and the robust_verdict state of the recommended plan vs the current approach.
Run:  python -m scripts.critic.probe_r3_floor_verdict [n_weeks]
"""
import copy
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from src.config import ROOT, load_assumptions, param_specs, param_values
from src.data.generate import simulate_world
from src.decision import uncertainty as unc
from src.decision.consequences import plan_totals
from src.decision.optimize import baseline_plan, optimize
from src.decision.policy_eval import _arrivals, true_params
from src.model import risk as R

warnings.filterwarnings("ignore")
NW = int(sys.argv[1]) if len(sys.argv) > 1 else 10
cfg = load_assumptions()
cfg_off = copy.deepcopy(cfg)
cfg_off["decision"]["borderline_floor"] = False
P, Pt, specs = param_values(cfg), true_params(cfg), param_specs(cfg)
m = R.fit(pd.read_csv(ROOT / "data/historical_cohorts.csv"), n_boot=30, seed=0)
rows, states = [], Counter()
for w in range(NW):
    f = simulate_world(24, 88_000 + w, cfg, prefix=f"W{w}_")
    lots = _arrivals(f).reset_index(drop=True)
    pt = f["_marg_p0"].to_numpy()
    r, D = R.assess(m, lots, cfg)
    on, off = optimize(lots, r, P, cfg).choice, optimize(lots, r, P, cfg_off).choice
    cur = baseline_plan(lots, cfg)
    for name, ch in (("floor_on", on), ("floor_off", off)):
        t = plan_totals(lots, pt, ch, Pt)
        rows.append({"week": w, "plan": name, "cost": t["cash_cost"], "deaths": t["deaths"], "regimens": t["regimens"],
                     "hi_on_std": int(sum(h and c == "STD" for h, c in zip(pt >= 0.15, ch))),
                     "n_STD": ch.count("STD"), "n_META": sum(c.startswith("META") for c in ch)})
    mc = unc.compare(lots, D, {"rec": on, "cur": cur}, specs, P, n=100, seed=w)
    corner = unc.corner_check(lots, r["risk_mean"].to_numpy(), on, cur, specs, P, cfg)
    states[unc.robust_verdict(unc.summarize(mc, "rec", "cur"), corner, cfg)[0]] += 1
T = pd.DataFrame(rows)
print(T.groupby("plan")[["cost", "deaths", "regimens", "hi_on_std", "n_STD", "n_META"]].mean().round(2).to_string())
print("verdict states over", NW, "weeks:", dict(states))
