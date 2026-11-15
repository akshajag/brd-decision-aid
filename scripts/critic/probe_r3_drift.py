"""Critic round 3 probe: operating characteristics of risk.check_drift (R2-8 fix).

False-alarm rate on histories WITHOUT drift (base world and misspecified worlds S1/S2) and power on the drift world.
Uses cheap models (n_point=1, n_boot=0): check_drift's z depends only on the pre-window refit, not on the ensemble.
Run:  python -m scripts.critic.probe_r3_drift [n_null] [n_drift]
"""
import sys
import warnings

import numpy as np

from src.config import load_assumptions
from src.data.generate import HIST_DATES, HIST_WEIGHTS, simulate_world
from src.model import risk as R
from src.model.stress import SCENARIOS, split

warnings.filterwarnings("ignore")
NN = int(sys.argv[1]) if len(sys.argv) > 1 else 30
ND = int(sys.argv[2]) if len(sys.argv) > 2 else 10
cfg = load_assumptions()
zc = cfg["decision"].get("drift_alarm_z", 2.5)


def zs(world, seeds):
    out = []
    for s in seeds:
        h = simulate_world(1400, s, cfg, world=world, dates=HIST_DATES, weights=HIST_WEIGHTS, prefix="H")
        m = R.fit(h, n_boot=0, n_point=1, drift_check=False)
        d = R.check_drift(m, h, np.random.default_rng(s))
        out.append((d["z"], d["offset"]))
    return np.array(out)


for name, world, seeds in (("base (no drift)", {}, range(73_000, 73_000 + NN)),
                           ("S2 heterogeneity (no drift)", split(SCENARIOS["S2_heterogeneity"])[0],
                            range(74_000, 74_000 + NN // 2)),
                           ("S1 interactions (no drift)", split(SCENARIOS["S1_interactions"])[0],
                            range(75_000, 75_000 + NN // 2)),
                           ("drift 0.35/yr", split(SCENARIOS["drift"])[0], range(76_000, 76_000 + ND)),
                           ("drift 0.15/yr", {"drift_per_year": 0.15}, range(77_000, 77_000 + ND))):
    Z = zs(world, seeds)
    print(f"{name:30s} n={len(Z):3d}  z mean {Z[:, 0].mean():+.2f} sd {Z[:, 0].std(ddof=1):.2f}  "
          f"alarm rate {np.mean(np.abs(Z[:, 0]) > zc):.2f}  mean offset {Z[:, 1].mean():+.3f}", flush=True)
