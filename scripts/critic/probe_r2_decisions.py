"""Critic round 2 probe: decision-layer robustness.

1. 'p_loss_week' is computed on EXPECTED cost given E[p|x]; what is it when each lot's own outcome varies
   (realised lot effect, as a yard would actually experience a week)?
2. Seed stability: how many of the 24 demo lots change option when only the bootstrap seed changes?
3. Invariance of the recommendation to row order and lot_id.
4. Permutation canary: shuffle outcomes across historical lots; temporal AUC must collapse to ~0.5.
5. Monotonicity of predicted risk in known risk factors.
Run:  python -m scripts.critic.probe_r2_decisions
"""
import time
import warnings

import numpy as np
import pandas as pd

from src.config import ROOT, load_assumptions, param_values
from src.data.generate import simulate_world
from src.decision.consequences import plan_totals
from src.decision.optimize import baseline_plan, optimize
from src.decision.policy_eval import _arrivals, true_params
from src.model import risk as R

warnings.filterwarnings("ignore")
t0 = time.time()
cfg = load_assumptions()
P = param_values(cfg)
Pt = true_params(cfg)
hist = pd.read_csv(ROOT / "data/historical_cohorts.csv")
cur = pd.read_csv(ROOT / "data/current_arrivals.csv")
m0 = R.fit(hist, n_boot=20, seed=0)
print(f"fit [{time.time() - t0:.0f}s]")

# 1. aleatoric loss weeks
exp_loss, real_loss, dd_real = [], [], []
for w in range(30):
    f = simulate_world(24, 99_000 + w, cfg, prefix=f"W{w}_")
    lots = _arrivals(f).reset_index(drop=True)
    r, _ = R.assess(m0, lots, cfg, n_draws=20)
    rec, cu = optimize(lots, r, P, cfg).choice, baseline_plan(lots, cfg)
    pe, pr = f["_marg_p0"].to_numpy(), f["_true_p0"].to_numpy()
    exp_loss.append(plan_totals(lots, pe, rec, Pt)["cash_cost"] > plan_totals(lots, pe, cu, Pt)["cash_cost"])
    a, b = plan_totals(lots, pr, rec, Pt), plan_totals(lots, pr, cu, Pt)
    real_loss.append(a["cash_cost"] > b["cash_cost"])
    dd_real.append(a["deaths"] - b["deaths"])
print(f"[1] loss weeks: expected-value scoring {np.mean(exp_loss):.2f}; with realised lot effects {np.mean(real_loss):.2f}; "
      f"weeks with more deaths (realised) {np.mean(np.array(dd_real) > 0):.2f}  [{time.time() - t0:.0f}s]")

# 2. seed stability of the demo recommendation
r0, _ = R.assess(m0, cur, cfg)
ch0 = optimize(cur, r0, P, cfg).choice
flips = []
for s in (1, 2, 3):
    ms = R.fit(hist, n_boot=20, seed=s)
    rs, _ = R.assess(ms, cur, cfg)
    chs = optimize(cur, rs, P, cfg).choice
    flips.append(sum(a != b for a, b in zip(ch0, chs)))
    tiers = (rs.tier != r0.tier).sum(), (rs.confidence != r0.confidence).sum()
    print(f"[2] seed {s}: {flips[-1]} of {len(cur)} lots change option; tier changes {tiers[0]}, confidence changes {tiers[1]}; "
          f"max |dp| {np.abs(rs.risk_mean - r0.risk_mean).max():.4f}")

# 3. invariance to row order and lot_id
perm = np.random.default_rng(1).permutation(len(cur))
cp = cur.iloc[perm].reset_index(drop=True).copy()
cp["lot_id"] = [f"Z{i:03d}" for i in range(len(cp))]
rp, _ = R.assess(m0, cp, cfg)
chp = optimize(cp, rp, P, cfg).choice
back = dict(zip(cur.lot_id.iloc[perm], chp))
print(f"[3] row-order/lot_id permutation: {sum(back[l] != c for l, c in zip(cur.lot_id, ch0))} option changes; "
      f"max |dp| {np.abs(rp.risk_mean.to_numpy() - r0.risk_mean.to_numpy()[perm]).max():.5f}")

# 4. permutation canary
from src.model.validation import temporal_split, _score
hp = hist.copy()
idx = np.random.default_rng(0).permutation(len(hp))
for c in ("brd_cases_45d",):
    hp[c] = hp[c].to_numpy()[idx]
hp["head_count"] = hp["head_count"].to_numpy()  # keep head; clip cases to head
hp["brd_cases_45d"] = np.minimum(hp["brd_cases_45d"], hp["head_count"])
hp["morbidity_45d"] = hp["brd_cases_45d"] / hp["head_count"]
hp["high_risk"] = (hp["morbidity_45d"] >= 0.15).astype(int)
tr, te = temporal_split(hp)
print(f"[4] permuted outcomes temporal AUC {_score(R.fit(tr, n_boot=0), te, cfg)['auc']:.3f}")

# 5. monotonicity
base = cur.copy()
for col, delta, sign in (("arrival_weight_lb", -100, +1), ("transit_hours", +10, +1), ("n_sources", +5, +1),
                         ("shrink_pct", +3, +1), ("pen_area_m2_per_head", -5, +1), ("bunk_cm_per_head", -10, 0)):
    b2 = base.copy()
    b2[col] = pd.to_numeric(b2[col]) + delta
    d = m0.predict_mean(b2) - m0.predict_mean(base)
    print(f"[5] {col} {delta:+}: min dp {np.nanmin(d):+.4f}, max dp {np.nanmax(d):+.4f}")
print(f"done [{time.time() - t0:.0f}s]")
