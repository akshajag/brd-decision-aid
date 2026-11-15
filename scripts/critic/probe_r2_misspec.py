"""Critic round 2 probe: where does the lot-level (misspecification) bias come from?
Mean prediction over K histories (member-0 fits) minus true E[p|x], on complete-data fresh lots, by covariate bins.
Run:  python -m scripts.critic.probe_r2_misspec [K]
"""
import sys
import warnings

import numpy as np
import pandas as pd

from src.config import load_assumptions
from src.data.generate import HIST_DATES, HIST_WEIGHTS, simulate_world
from src.model import risk as R

warnings.filterwarnings("ignore")
K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
cfg = load_assumptions()
f = simulate_world(2000, 779, cfg, meta_probs=(0.0, 0.0), mask=False)
P = np.mean([R.fit(simulate_world(1400, 72_000 + s, cfg, dates=HIST_DATES, weights=HIST_WEIGHTS, prefix="H"),
                   n_boot=0, seed=0).predict_mean(f) for s in range(K)], axis=0)
f["bias_pp"] = 100 * (P - f["_marg_p0"].to_numpy())
f["truth_pct"] = 100 * f["_marg_p0"]
print(f"overall RMS of mean bias {np.sqrt((f.bias_pp ** 2).mean()):.2f} pp")
for col, bins in (("arrival_weight_lb", [0, 500, 600, 700, 800, 900, 2000]), ("shrink_pct", [0, 2, 3, 4, 5, 6, 13]),
                  ("pen_area_m2_per_head", [0, 12, 15, 18, 22, 40]), ("transit_hours", [0, 5, 10, 20, 41]),
                  ("truth_pct", [0, 5, 10, 15, 20, 30, 50, 100])):
    print(f.groupby(pd.cut(f[col], bins, right=False), observed=True)["bias_pp"].agg(["mean", "size"]).round(2).T.to_string(), "\n")
for col in ("source_type", "weather_stress", "yard", "sex"):
    print(f.groupby(col)["bias_pp"].agg(["mean", "size"]).round(2).T.to_string(), "\n")
