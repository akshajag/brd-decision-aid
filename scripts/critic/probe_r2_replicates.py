"""Critic round 2 probe: is the D8 band bias / D2 coverage shortfall sampling error or systematic?

K independent histories from the seed-42 world (same yard effects). For each: band bias vs true E[p|x] on fixed
fresh untreated lots, 90% coverage, and the bootstrap SD of each lot's risk. Across histories: mean bias with SE, and
the ratio of the across-history SD of each lot's prediction (the TRUE sampling SD) to the mean bootstrap SD
(what the 90% range claims). Run:  python -m scripts.critic.probe_r2_replicates [K] [n_boot]
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd

from src.config import ROOT, load_assumptions
from src.data.generate import HIST_DATES, HIST_WEIGHTS, simulate_world
from src.model import risk as R
from src.model.validation import BANDS

warnings.filterwarnings("ignore")
K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 20
t0 = time.time()
cfg = load_assumptions()
f = simulate_world(600, 777, cfg, meta_probs=(0.0, 0.0))
A = f.drop(columns=[c for c in f.columns if c.startswith("_")] + ["metaphylaxis_given", "brd_cases_45d",
                                                                  "morbidity_45d", "high_risk"])
mp = f["_marg_p0"].to_numpy()
band = pd.cut(f["arrival_weight_lb"], BANDS, right=False)
preds, sds, rows = [], [], []
for s in [None] + list(range(K)):
    h = pd.read_csv(ROOT / "data/historical_cohorts.csv") if s is None else \
        simulate_world(1400, 70_000 + s, cfg, dates=HIST_DATES, weights=HIST_WEIGHTS, prefix="H")
    m = R.fit(h, n_boot=NB, seed=0)
    r, D = R.assess(m, A, cfg, n_draws=2 * NB)
    p = r["risk_mean"].to_numpy()
    b = pd.Series(p - mp).groupby(band, observed=True).mean()
    rows.append({"hist": "CSV" if s is None else 70_000 + s, "sd": round(m.resid_sd, 3),
                 "cov90": float(np.mean((r.risk_lo <= mp) & (mp <= r.risk_hi))),
                 **{str(k): round(1e2 * v, 2) for k, v in b.items()}})
    if s is not None:
        preds.append(p)
        sds.append(D.std(axis=1, ddof=1))
    print(rows[-1], f"[{time.time() - t0:.0f}s]", flush=True)
T = pd.DataFrame(rows)
fr = T[T["hist"] != "CSV"]
bc = [c for c in T.columns if c.startswith("[")]
print("\nfresh histories: mean band bias (pp) and SE")
print(pd.DataFrame({"mean": fr[bc].mean(), "se": fr[bc].std(ddof=1) / np.sqrt(len(fr)), "sd": fr[bc].std(ddof=1)}).round(2))
print(f"mean coverage {fr.cov90.mean():.3f} (min {fr.cov90.min():.2f}, max {fr.cov90.max():.2f})")
P, S = np.array(preds), np.array(sds)
emp = P.std(axis=0, ddof=1)
boot = S.mean(axis=0)
print(f"across-history SD / mean bootstrap SD per lot: median ratio {np.median(emp / boot):.2f}, "
      f"head-mean ratio {emp.mean() / boot.mean():.2f}")
# coverage of the MEAN-over-histories prediction: is there a common bias not explained by sampling?
z = (P.mean(axis=0) - mp) / (emp / np.sqrt(len(P)))
print(f"lots whose across-history mean prediction is >2 SE from truth: {np.mean(np.abs(z) > 2):.2f} (~0.05 if unbiased)")
