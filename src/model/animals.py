"""Individual-animal secondary view (P2).

The decision unit is the lot. Pen riders still ask "which animals in this pen do I look at first?". No
animal-level records exist (the data are synthetic and lot-level), so this view is a *ranking aid*, not a
diagnosis:

* each head gets a synthetic arrival weight around the lot mean (within-lot SD is an assumption,
  config `individual_view.weight_sd_within_lot_lb`, tag ASM);
* its risk is the lot model evaluated at that weight, with the lot's other fields (and the same imputations of any
  missing lot records) held fixed, so only weight moves the ranking;
* the animal risks are then shifted on the logit scale so their mean equals the lot's risk, keeping the lot's
  expected case count unchanged.

Caveat (ecological inference): the model was fitted to lot averages, so applying its weight effect to individuals
assumes the within-lot weight effect matches the between-lot one. Real use needs animal records (e.g. arrival
weight per head, temperature, clinical scores).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from src.model import risk as riskmod


def _rng_for(lot_id: str) -> np.random.Generator:
    h = int.from_bytes(hashlib.blake2b(str(lot_id).encode(), digest_size=8).digest(), "little")
    return np.random.default_rng([h, 20261115])


def animal_view(model: riskmod.RiskModel, lot: pd.Series, lot_risk: float, cfg: dict) -> pd.DataFrame | None:
    """One row per head: synthetic arrival weight, relative risk ranking, watch-list flag.

    Returns None when the lot has no arrival weight or no head count (nothing to vary)."""
    iv = cfg.get("individual_view", {})
    sd = float(iv.get("weight_sd_within_lot_lb", 45))
    watch_frac = float(iv.get("watch_fraction", 0.10))
    n = lot.get("head_count")
    w0 = lot.get("arrival_weight_lb")
    if n is None or w0 is None or pd.isna(n) or pd.isna(w0) or int(n) < 1:
        return None
    n = int(n)
    rng = _rng_for(lot["lot_id"])
    w = np.clip(rng.normal(float(w0), sd, n), float(w0) - 3 * sd, float(w0) + 3 * sd).round()

    # impute the lot's missing fields ONCE (from the lot's own stream) and share across its animals
    one = riskmod.prepare(pd.DataFrame([lot])).reset_index(drop=True)
    has_miss = one[riskmod.IMPUTABLE].isna().any().any()
    m = max(model.n_impute, model.n_point) if has_miss else model.n_point
    imp1 = riskmod._lot_imputations(one, model.donors, m, 0) if has_miss else {}
    rows = pd.concat([one] * n, ignore_index=True)
    rows["arrival_weight_lb"] = w
    imp = {c: np.repeat(a, n, axis=0) for c, a in imp1.items()}
    acc = np.zeros(n)
    for j in range(m):
        k = j % model.n_point
        X = riskmod._design_mi(rows, riskmod._fill_from(rows, imp, j), model.pre[k][0], None)
        acc += riskmod._sig(model._logit(X, k))
    p = acc / m
    p = riskmod.recentre(p[None, :], np.array([float(lot_risk)]), iters=8)[0]

    out = pd.DataFrame({"animal_id": [f"{lot['lot_id']}-{i + 1:03d}" for i in range(n)],
                        "arrival_weight_lb": w.astype(int), "risk": p})
    out = out.sort_values("risk", ascending=False, kind="mergesort").reset_index(drop=True)
    out["rank"] = np.arange(1, n + 1)
    out["watch_first"] = out["rank"] <= max(1, int(np.ceil(watch_frac * n)))
    return out
