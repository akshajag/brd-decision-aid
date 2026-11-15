"""Critic helpers: rebuild the synthetic 'world' of seed 42 and draw FRESH lots from it.

generate() draws the four yard effects first from default_rng(seed), so the same effects can be
rebuilt here and reused with new lot seeds (fresh, never-tuned evaluation data from the same world).
Columns added (in memory only): _true_p0 (conditional on the unobserved lot effect),
_true_lp (covariate-only logit), _marg_p0 (E over the lot effect = the design's estimand p0(lot)).
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from src.config import load_assumptions
from src.data.generate import draw_lots, mask_missing, simulate_outcomes, untreated_logit

HIST_DATES = pd.date_range("2024-01-01", "2026-10-31", freq="D")
FUT_DATES = pd.date_range("2026-11-01", "2027-10-31", freq="D")
HIST_W = {"A": 0.32, "B": 0.26, "C": 0.24, "D": 0.18}
CUR_W = {"A": 0.25, "B": 0.30, "C": 0.25, "D": 0.20}
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(40)


def yard_effects(cfg: dict, world_seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(world_seed)
    return {y: float(rng.normal(0, cfg["dgp"]["yard_effect_sd"])) for y in cfg["yards"]}


def marginal_p(lp: np.ndarray, sd: float) -> np.ndarray:
    """E[logistic(lp + u)], u ~ N(0, sd^2), by Gauss-Hermite quadrature."""
    z = lp[:, None] + sd * _GH_X[None, :]
    return (1 / (1 + np.exp(-z))) @ (_GH_W / _GH_W.sum())


def fresh_lots(n: int, seed: int, cfg: dict | None = None, *, meta_probs: tuple[float, float] | None = None,
               rrr: float | None = None, dates=FUT_DATES, weights=CUR_W, world_seed: int = 42,
               lot_sd: float | None = None, mask: bool = True) -> pd.DataFrame:
    cfg = copy.deepcopy(cfg or load_assumptions())
    if meta_probs is not None:
        cfg["dgp"]["hist_meta_prob_light"], cfg["dgp"]["hist_meta_prob_heavy"] = meta_probs
    if rrr is not None:
        cfg["dgp"]["true_metaphylaxis_rrr"] = rrr   # round 2: the simulator's truth now lives in dgp (D14)
    if lot_sd is not None:
        cfg["dgp"]["lot_unobserved_sd"] = lot_sd
    ye = yard_effects(cfg, world_seed)
    rng = np.random.default_rng(seed)
    lots = draw_lots(n, rng, cfg, dates, weights, "F")
    out = simulate_outcomes(lots, cfg, rng, ye)
    out["_true_lp"] = untreated_logit(lots, cfg["dgp"], ye)
    out["_marg_p0"] = marginal_p(out["_true_lp"].to_numpy(), cfg["dgp"]["lot_unobserved_sd"])
    return mask_missing(out, rng) if mask else out
