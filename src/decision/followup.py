"""Days 1-14 check-in: update each lot's 45-day risk from BRD pulls recorded so far.

Method (transparent, no new data needed):
* Prior for each lot's 45-day morbidity under its assigned option = arrival risk-model draws
  (bootstrap x imputation) plus the model's lot-level logit dispersion (resid_sd), then the
  option's assumed effects. This is the same uncertainty the arrival decision used.
* Onset timing: first-treatment cases are assumed spread evenly over the first
  `case_onset_spread_days` (ASM, same assumption as the hospital-load calculation), so the
  fraction expected by day d is F(d) = min(d / spread, 1).
* Likelihood: pulls_to_date ~ Binomial(head, m * F(d)) by default; optionally beta-binomial
  (pen_clustering_icc > 0, a sensitivity check). Importance weights give the
  posterior over m; we report its mean and 90% range.

Everything is an *estimate under stated assumptions*. Medication changes still require
veterinary review; the tool only suggests monitoring/capacity actions and escalation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import betabinom, binom, norm

from .consequences import OPTIONS, pens

NEW_HIGH = "Newly high risk - escalate"
BORDERLINE = "Possibly high (borderline) - review"
STATUS_ORDER = [NEW_HIGH, BORDERLINE, "Above expected", "On track", "Below expected", "No pulls recorded"]
ESCALATE = (NEW_HIGH, BORDERLINE, "Above expected")
DEMO_SEED = 150  # day-7 total 123 pulls vs a median of 120 over 200 seeds (typical, not favourable); disclosed in DEMO_CASE


def onset_fraction(day: float, params: dict) -> float:
    return float(min(max(day, 0.0) / params["case_onset_spread_days"], 1.0))


def _prior_particles(draws: np.ndarray, resid_sd: float, n: int, seed: int, random: bool = False) -> np.ndarray:
    """(n_lots, n) untreated-morbidity particles: each epistemic draw of the lot's EXPECTED risk, spread by the
    lot-to-lot logit SD with the logit-normal centred so its mean equals that draw.

    This is exactly the distribution behind the arrival table's P(>=15%) (risk.prob_high_from_draws), so with no
    pulls the check-in reproduces the arrival numbers (critic D12), and the particle mean equals the plan's
    expected risk (mean of the draws). Deterministic stratified spread by default; random=True for simulation.
    """
    from src.model.risk import lot_centre
    rng = np.random.default_rng(seed)
    d = np.clip(np.asarray(draws, float), 1e-6, 1 - 1e-6)
    n_lots, n_d = d.shape
    if random:
        cols = rng.integers(0, n_d, n)
        z = rng.normal(0.0, 1.0, (n_lots, n))
        return 1 / (1 + np.exp(-(lot_centre(d[:, cols], resid_sd) + resid_sd * z)))
    m = max(int(np.ceil(n / n_d)), 1)
    zq = norm.ppf((np.arange(m) + 0.5) / m)
    c = lot_centre(d, resid_sd)                              # (n_lots, n_d)
    return 1 / (1 + np.exp(-(c[:, :, None] + resid_sd * zq[None, None, :]).reshape(n_lots, n_d * m)))


def option_morbidity(p0: np.ndarray, option: str, params: dict) -> np.ndarray:
    _, meta, enh = OPTIONS[option]
    return p0 * (1 - params["metaphylaxis_morbidity_rrr"] * meta) * (1 - params["receiving_protocol_morbidity_rrr"] * enh)


def pulls_pmf(k: int, n: int, q: np.ndarray, icc: float) -> np.ndarray:
    """P(k pulls of n) when each animal's chance is q; icc > 0 lets cases cluster (beta-binomial)."""
    if icc <= 1e-9:
        return binom.pmf(k, n, q)
    s = (1 - icc) / icc
    return betabinom.pmf(k, n, np.clip(q, 1e-9, 1) * s, np.clip(1 - q, 1e-9, 1) * s)


def _wquantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, q * cw[-1])])


def simulate_pulls(lots: pd.DataFrame, draws: np.ndarray, resid_sd: float, choice: list[str],
                   params: dict, day: int, seed: int = DEMO_SEED) -> np.ndarray:
    """SYNTHETIC pulls-to-date for the demo: one plausible 'true' morbidity per lot from the prior."""
    rng = np.random.default_rng(seed)
    p0 = _prior_particles(draws, resid_sd, 1, seed + 7919, random=True)[:, 0]   # separate stream from the pull draws
    m = np.array([option_morbidity(p0[i:i + 1], c, params)[0] for i, c in enumerate(choice)])
    q = m * onset_fraction(day, params)
    icc = params.get("pen_clustering_icc", 0.0)
    if icc > 1e-9:                            # clustered outbreaks: lot-specific attack rate ~ Beta
        s = (1 - icc) / icc
        q = rng.beta(np.clip(q, 1e-9, 1) * s, np.clip(1 - q, 1e-9, 1) * s)
    return rng.binomial(lots["head_count"].to_numpy(int), q).astype(float)


def update(lots: pd.DataFrame, draws: np.ndarray, resid_sd: float, choice: list[str], params: dict,
           day: int, pulls: np.ndarray, thr_high: float = 0.15, n_particles: int = 4000,
           seed: int = 11, escalate_ratio: float | None = None, escalate_excess: float | None = None) -> pd.DataFrame:
    """Posterior 45-day morbidity per lot given pulls to date (NaN = not recorded).

    p_high = P(45-day morbidity under the lot's assigned option >= threshold); for standard care this is the
    same quantity as the arrival table's P(>=15%).
    """
    if escalate_ratio is None or escalate_excess is None:
        from src.config import load_assumptions
        dec = load_assumptions()["decision"]
        escalate_ratio = dec.get("escalate_ratio", 1.5) if escalate_ratio is None else escalate_ratio
        escalate_excess = dec.get("escalate_excess", 3) if escalate_excess is None else escalate_excess
    head = lots["head_count"].to_numpy(float)
    F = onset_fraction(day, params)
    P0 = _prior_particles(draws, resid_sd, n_particles, seed)
    rows = []
    for i, opt in enumerate(choice):
        m = np.clip(option_morbidity(P0[i], opt, params), 1e-6, 1 - 1e-6)
        prior_mean = float(m.mean())
        expected = head[i] * prior_mean * F
        k = pulls[i]
        if np.isnan(k) or F <= 0:
            w = np.ones_like(m)
        else:
            w = pulls_pmf(int(k), int(head[i]), m * F, params.get("pen_clustering_icc", 0.0))
            if w.sum() <= 0:          # pulls far outside the prior: fall back to the observed rate
                m = np.full_like(m, min((k / head[i]) / max(F, 1e-6), 0.99))
                w = np.ones_like(m)
        w = w / w.sum()
        post = float(np.sum(w * m))
        lo, hi = _wquantile(m, w, 0.05), _wquantile(m, w, 0.95)
        p_high = float(np.sum(w * (m >= thr_high)))
        if np.isnan(k):
            status = "No pulls recorded"
        elif post >= thr_high and p_high >= 0.5 and prior_mean < thr_high:
            # same rule as day 0: only call it high when the whole 90% range is above the threshold
            status = NEW_HIGH if lo >= thr_high else BORDERLINE
        elif k > max(escalate_ratio * expected, expected + escalate_excess):
            status = "Above expected"
        elif k < 0.5 * expected and expected - k >= escalate_excess:   # display label only; no action attached
            status = "Below expected"
        else:
            status = "On track"
        remaining = head[i] * post * (1 - F)
        rows.append({"lot_id": lots["lot_id"].iloc[i], "yard": lots["yard"].iloc[i], "head_count": int(head[i]),
                     "option": opt, "expected_pulls_to_date": expected, "pulls_to_date": k,
                     "prior_risk": prior_mean, "updated_risk": post, "updated_lo": lo, "updated_hi": hi,
                     "p_high": p_high, "remaining_cases": remaining, "status": status})
    return pd.DataFrame(rows)


def hospital_outlook(upd: pd.DataFrame, params: dict, caps: dict, day: int) -> pd.DataFrame:
    """Projected hospital-pen load per yard for the coming week vs free capacity."""
    spread = params["case_onset_spread_days"]
    days_left = max(spread - day, 0)
    out = []
    for y, g in upd.groupby("yard"):
        rate = g["remaining_cases"].sum() / days_left if days_left > 0 else 0.0
        load = rate * params["hospital_days_per_case"]
        cap = caps[y]["hosp_free"]
        out.append({"yard": y, "projected_new_cases_next_7d": rate * min(7, days_left),
                    "projected_hospital_load": load, "free_hospital_head": cap,
                    "status": "SHORTFALL - escalate" if load > cap + 1e-9 else
                              ("tight (>85%)" if load > 0.85 * cap else "ok")})
    return pd.DataFrame(out)


def suggested_actions(lots: pd.DataFrame, upd: pd.DataFrame, params: dict, caps: dict,
                      rider_used: dict) -> list[str]:
    """Monitoring/capacity suggestions only - never drug or dose advice."""
    acts = []
    slack = {y: caps[y]["rider_h_day"] - rider_used.get(y, 0.0) for y in caps}
    for _, r in upd.sort_values("updated_risk", ascending=False).iterrows():
        if r["status"] in ESCALATE:
            enh = OPTIONS[r["option"]][2]
            need = float(pens(np.array([r["head_count"]]), params)[0] * params["enhanced_monitoring_hours_per_pen_day"])
            msg = (f"**{r['lot_id']}** (yard {r['yard']}): {r['pulls_to_date']:.0f} pulls vs "
                   f"{r['expected_pulls_to_date']:.1f} expected; updated 45-day risk {100 * r['updated_risk']:.0f}% "
                   f"({100 * r['updated_lo']:.0f}-{100 * r['updated_hi']:.0f}%, "
                   f"P(≥15%) {100 * r['p_high']:.0f}%) - {r['status'].lower()}.")
            if not enh:
                if slack[r["yard"]] >= need - 1e-9:
                    msg += (f" Add enhanced monitoring (needs {need:.1f} rider h/day; "
                            f"{slack[r['yard']]:.1f} free in yard {r['yard']}).")
                    slack[r["yard"]] -= need
                else:
                    msg += (f" Enhanced monitoring needs {need:.1f} rider h/day but only "
                            f"{max(slack[r['yard']], 0):.1f} are free - escalate staffing.")
            else:
                msg += " Already on enhanced monitoring."
            msg += " Veterinarian to review the treatment protocol for this lot."
            acts.append(msg)
    return acts
