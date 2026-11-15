"""Seeded SYNTHETIC data generator for the BRD decision aid.

No real dataset was provided by the challenge. Cohorts are drawn from yard profiles
and a documented data-generating process (DGP) whose coefficients live in
config/assumptions.yaml (section `dgp`). Directions follow the cited literature;
magnitudes are team assumptions. Outputs:

  data/historical_cohorts.csv  past lots with 45-day outcomes (for fitting/evaluating)
  data/current_arrivals.csv    lots arriving this week, no outcomes (the decision set)

Usage:  python -m src.data.generate --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT, load_assumptions

DATA_DIR = ROOT / "data"

ARRIVAL_COLUMNS = [
    "lot_id", "yard", "arrival_date", "arrival_quarter", "head_count",
    "arrival_weight_lb", "sex", "source_type", "n_sources", "transit_hours",
    "vacc_history", "shrink_pct", "pen_area_m2_per_head", "bunk_cm_per_head",
    "weather_stress",
]
OUTCOME_COLUMNS = ["metaphylaxis_given", "brd_cases_45d", "morbidity_45d", "high_risk"]
MISSABLE = {  # column: (missing prob if auction, missing prob otherwise)
    "transit_hours": (0.30, 0.08),
    "n_sources": (0.35, 0.05),
    "shrink_pct": (0.25, 0.20),
    "pen_area_m2_per_head": (0.10, 0.10),
}


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def untreated_logit(df: pd.DataFrame, dgp: dict, yard_eff: dict[str, float]) -> np.ndarray:
    """True logit of 45-day morbidity WITHOUT metaphylaxis, using complete (unmasked) values."""
    z = np.full(len(df), dgp["intercept"], dtype=float)
    z += dgp["weight_per_100lb_below_700"] * np.clip((700 - df["arrival_weight_lb"]) / 100, -2, None)
    z += np.where(df["source_type"] == "auction", dgp["source_auction"], 0.0)
    z += np.where(df["source_type"] == "backgrounded", dgp["source_backgrounded"], 0.0)
    z += dgp["per_extra_source"] * np.clip(df["n_sources"] - 1, 0, 15)
    z += dgp["transit_per_hour"] * df["transit_hours"]
    z += np.where(df["vacc_history"] == "unknown", dgp["vacc_unknown"], dgp["vacc_known"])
    z += np.where(df["sex"] == "heifer", dgp["sex_heifer"], 0.0)
    z += np.where(df["sex"] == "mixed", dgp["sex_mixed"], 0.0)
    z += df["arrival_quarter"].map(dgp["quarter"]).to_numpy(dtype=float)
    z += dgp["shrink_per_pct_above_3"] * np.clip(df["shrink_pct"] - 3, 0, None)
    z += dgp["crowding_per_m2_below_15"] * np.clip(15 - df["pen_area_m2_per_head"], 0, None)
    z += df["weather_stress"].map(dgp["weather"]).to_numpy(dtype=float)
    z += df["yard"].map(yard_eff).to_numpy(dtype=float)
    return z


def draw_lots(n: int, rng: np.random.Generator, cfg: dict, dates: pd.DatetimeIndex,
              yard_weights: dict[str, float], id_prefix: str) -> pd.DataFrame:
    yards = cfg["yards"]
    keys = list(yard_weights)
    p = np.array([yard_weights[k] for k in keys], dtype=float)
    yard = rng.choice(keys, size=n, p=p / p.sum())
    rows = []
    for i, y in enumerate(yard):
        Y = yards[y]
        mix = Y["source_mix"]
        src = rng.choice(list(mix), p=np.array(list(mix.values())) / sum(mix.values()))
        date = dates[rng.integers(len(dates))]
        q = f"Q{(date.month - 1) // 3 + 1}"
        # fall calves are lighter
        w = rng.normal(Y["weight_mean"] - (40 if q == "Q4" else 0), Y["weight_sd"])
        w = float(np.clip(w, 380, 1000))
        head = int(np.clip(rng.lognormal(np.log(130), 0.35), 40, 320))
        if src == "ranch_direct":
            n_src = 1
        elif src == "backgrounded":
            n_src = int(rng.integers(1, 4))
        else:
            n_src = int(2 + rng.poisson(5))
        sex = rng.choice(["steer", "heifer", "mixed"],
                         p=[0.5, 0.3, 0.2] if src != "auction" else [0.4, 0.25, 0.35])
        transit = float(np.clip(rng.gamma(4, Y["transit_mean_h"] / 4), 1, 40))
        vacc = "known" if rng.random() < {"ranch_direct": 0.6, "backgrounded": 0.85,
                                          "auction": 0.15}[src] else "unknown"
        shrink = float(np.clip(rng.normal(2.5 + transit * 0.12, 1.2), 0.5, 12))
        pen_area = float(np.clip(rng.normal(18, 4), 8, 35))
        bunk = float(np.clip(rng.normal(30, 5), 15, 50))
        weather = rng.choice(["low", "moderate", "high"],
                             p=[0.35, 0.40, 0.25] if q in ("Q4", "Q1") else [0.6, 0.3, 0.1])
        if y == "D":  # harsher swings at the small western yard
            weather = rng.choice(["low", "moderate", "high"], p=[0.25, 0.40, 0.35])
        rows.append(dict(
            lot_id=f"{id_prefix}{i + 1:04d}", yard=y, arrival_date=date.date().isoformat(),
            arrival_quarter=q, head_count=head, arrival_weight_lb=round(w), sex=sex,
            source_type=src, n_sources=n_src, transit_hours=round(transit, 1),
            vacc_history=vacc, shrink_pct=round(shrink, 1),
            pen_area_m2_per_head=round(pen_area, 1), bunk_cm_per_head=round(bunk, 1),
            weather_stress=weather,
        ))
    return pd.DataFrame(rows, columns=ARRIVAL_COLUMNS)


def simulate_outcomes(df: pd.DataFrame, cfg: dict, rng: np.random.Generator,
                      yard_eff: dict[str, float]) -> pd.DataFrame:
    dgp = cfg["dgp"]
    rrr = dgp["true_metaphylaxis_rrr"]   # the simulator's truth, NOT the planner's assumption (D14)
    z = untreated_logit(df, dgp, yard_eff) + rng.normal(0, dgp["lot_unobserved_sd"], len(df))
    p0 = _logistic(z)
    light = df["arrival_weight_lb"] < cfg["decision"]["baseline_meta_weight_lb"]
    meta = rng.random(len(df)) < np.where(light, dgp["hist_meta_prob_light"], dgp["hist_meta_prob_heavy"])
    p_obs = p0 * np.where(meta, 1 - rrr, 1.0)
    cases = rng.binomial(df["head_count"].to_numpy(), p_obs)
    out = df.copy()
    out["metaphylaxis_given"] = meta.astype(int)
    out["brd_cases_45d"] = cases
    out["morbidity_45d"] = np.round(cases / df["head_count"], 4)
    out["high_risk"] = (out["morbidity_45d"] >= cfg["decision"]["high_risk_morbidity"]).astype(int)
    out["_true_p0"] = p0  # kept in memory for tests only; not written to CSV
    return out


def mask_missing(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Realistic, source-dependent missingness (sale-barn lots have the worst records)."""
    out = df.copy()
    auction = out["source_type"] == "auction"
    for col, (p_auc, p_oth) in MISSABLE.items():
        m = rng.random(len(out)) < np.where(auction, p_auc, p_oth)
        out[col] = out[col].astype(float)
        out.loc[m, col] = np.nan
    return out


# ---------------- truth-known simulation for validation (never written to CSV) ----------------
# A "world" = the seed-42 yard effects + the documented DGP, optionally changed by a scenario
# (src/model/stress.py). Fresh lots drawn from it carry the truth, so the pipeline can be checked
# against known answers. Columns added in memory only:
#   _true_lp  covariate-only logit (no lot effect)
#   _true_p0  untreated risk of THIS lot (includes its unobserved lot effect)
#   _marg_p0  E[_true_p0 | recorded covariates] = the estimand p0(lot) of docs/DESIGN.md s1
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(40)
_GH_W = _GH_W / _GH_W.sum()
HIST_DATES = pd.date_range("2024-01-01", "2026-10-31", freq="D")
FUTURE_DATES = pd.date_range("2026-11-01", "2027-10-31", freq="D")
HIST_WEIGHTS = {"A": 0.32, "B": 0.26, "C": 0.24, "D": 0.18}
CUR_WEIGHTS = {"A": 0.25, "B": 0.30, "C": 0.25, "D": 0.20}
WORLD_DEFAULTS = {"interaction_auction_highwx": 0.0, "quad_transit": 0.0, "lot_sd": None,
                  "mnar_long_transit": 0.0, "drift_per_year": 0.0}


def world_yard_effects(cfg: dict, world_seed: int = 42) -> dict[str, float]:
    """The yard effects generate(seed) draws first, rebuilt so fresh lots share the CSV's world."""
    rng = np.random.default_rng(world_seed)
    return {y: float(rng.normal(0, cfg["dgp"]["yard_effect_sd"])) for y in cfg["yards"]}


def marginal_p(lp: np.ndarray, sd: float) -> np.ndarray:
    """E[logistic(lp + u)], u ~ N(0, sd^2) (Gauss-Hermite)."""
    z = np.asarray(lp, float)[:, None] + sd * _GH_X[None, :]
    return (1 / (1 + np.exp(-z))) @ _GH_W


def world_logit(df: pd.DataFrame, cfg: dict, yard_eff: dict[str, float], world: dict | None = None) -> np.ndarray:
    w = {**WORLD_DEFAULTS, **(world or {})}
    z = untreated_logit(df, cfg["dgp"], yard_eff)
    z = z + w["interaction_auction_highwx"] * ((df["source_type"] == "auction") &
                                               (df["weather_stress"] == "high")).to_numpy(float)
    z = z + w["quad_transit"] * (df["transit_hours"].to_numpy(float) - 12) ** 2
    if w["drift_per_year"]:
        yrs = (pd.to_datetime(df["arrival_date"]) - pd.Timestamp("2024-01-01")).dt.days.to_numpy() / 365.25
        z = z + w["drift_per_year"] * yrs
    return z


def simulate_world(n: int, seed: int, cfg: dict, *, world: dict | None = None, dates=FUTURE_DATES,
                   weights=CUR_WEIGHTS, prefix: str = "F", meta_probs: tuple[float, float] | None = None,
                   true_rrr: float | None = None, world_seed: int = 42, mask: bool = True) -> pd.DataFrame:
    """n fresh lots with outcomes and truth columns. meta_probs=(light, heavy) overrides historical treatment."""
    w = {**WORLD_DEFAULTS, **(world or {})}
    dgp = cfg["dgp"]
    sd = dgp["lot_unobserved_sd"] if w["lot_sd"] is None else w["lot_sd"]
    ye = world_yard_effects(cfg, world_seed)
    rng = np.random.default_rng(seed)
    df = draw_lots(n, rng, cfg, dates, weights, prefix)
    lp = world_logit(df, cfg, ye, w)
    p0 = _logistic(lp + rng.normal(0, sd, len(df)))
    pl, ph = (dgp["hist_meta_prob_light"], dgp["hist_meta_prob_heavy"]) if meta_probs is None else meta_probs
    light = df["arrival_weight_lb"] < cfg["decision"]["baseline_meta_weight_lb"]
    meta = rng.random(len(df)) < np.where(light, pl, ph)
    rrr = dgp["true_metaphylaxis_rrr"] if true_rrr is None else true_rrr
    cases = rng.binomial(df["head_count"].to_numpy(), p0 * np.where(meta, 1 - rrr, 1.0))
    df["metaphylaxis_given"] = meta.astype(int)
    df["brd_cases_45d"] = cases
    df["morbidity_45d"] = np.round(cases / df["head_count"], 4)
    df["high_risk"] = (df["morbidity_45d"] >= cfg["decision"]["high_risk_morbidity"]).astype(int)
    df["_true_lp"] = lp
    df["_true_p0"] = p0
    df["_marg_p0"] = marginal_p(lp, sd)
    if not mask:
        return df
    out = mask_missing(df, rng)
    if w["mnar_long_transit"]:                       # extra missingness for long hauls (not at random)
        long = df["transit_hours"].to_numpy(float) > 20
        extra = rng.random(len(df)) < np.where(long, w["mnar_long_transit"], 0.0)
        out.loc[extra, "transit_hours"] = np.nan
    return out


GENERATOR_KEYS = ("dgp", "yards")


def config_hash(cfg: dict | None = None) -> str:
    """SHA-256 (first 16 hex) of everything that defines the synthetic world (D10: generator freeze)."""
    import hashlib
    import json
    cfg = cfg or load_assumptions()
    blob = {k: cfg[k] for k in GENERATOR_KEYS}
    blob["decision"] = {k: cfg["decision"][k] for k in ("high_risk_morbidity", "baseline_meta_weight_lb")}
    return hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()[:16]


def generate(seed: int = 42, n_hist: int = 1400, n_current: int = 24):
    cfg = load_assumptions()
    rng = np.random.default_rng(seed)
    yard_eff = {y: float(rng.normal(0, cfg["dgp"]["yard_effect_sd"])) for y in cfg["yards"]}
    hist_dates = pd.date_range("2024-01-01", "2026-10-31", freq="D")
    hist = draw_lots(n_hist, rng, cfg, hist_dates,
                     {"A": 0.32, "B": 0.26, "C": 0.24, "D": 0.18}, "H")
    hist = simulate_outcomes(hist, cfg, rng, yard_eff)
    hist = mask_missing(hist, rng)
    cur_dates = pd.date_range("2026-11-16", "2026-11-20", freq="D")
    cur = draw_lots(n_current, rng, cfg, cur_dates,
                    {"A": 0.25, "B": 0.30, "C": 0.25, "D": 0.20}, "C")
    cur = mask_missing(cur, rng)
    return hist, cur, cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-hist", type=int, default=1400)
    ap.add_argument("--n-current", type=int, default=24)
    a = ap.parse_args()
    hist, cur, _ = generate(a.seed, a.n_hist, a.n_current)
    DATA_DIR.mkdir(exist_ok=True)
    hist[ARRIVAL_COLUMNS + OUTCOME_COLUMNS].to_csv(DATA_DIR / "historical_cohorts.csv", index=False)
    cur[ARRIVAL_COLUMNS].to_csv(DATA_DIR / "current_arrivals.csv", index=False)
    print(f"historical: {len(hist)} lots, {hist['head_count'].sum():,} head; "
          f"mean morbidity {hist['morbidity_45d'].mean():.3f}; high-risk prevalence {hist['high_risk'].mean():.3f}")
    print(hist.groupby("yard")[["arrival_weight_lb", "morbidity_45d", "high_risk", "metaphylaxis_given"]].mean().round(3))
    print(f"current arrivals: {len(cur)} lots; missing cells: {int(cur.isna().sum().sum())}")
    write_manifest(a.seed)


# Seeds used while the generator was tuned (intercept and lot SD were adjusted while watching hold-out AUC on the
# seed-42 data) versus seeds reserved for fresh evaluation (critic tests, oracle recovery, regret and stress studies).
TUNING_SEEDS = [42]
FRESH_EVAL_SEEDS = [7, 18, 31337, 4242, 777, 778, 2024, 99, 101, 5, 6] + list(range(50_000, 50_006)) +     list(range(60_000, 60_012)) + list(range(9000, 9010))


def write_manifest(seed: int = 42) -> None:
    """Freeze record for the synthetic world (critic D10)."""
    import json
    man = {"seed": seed, "config_hash": config_hash(), "tuning_seeds": TUNING_SEEDS,
           "fresh_eval_seeds": FRESH_EVAL_SEEDS,
           "note": "config_hash covers the dgp and yards sections plus the high-risk threshold and baseline weight. "
                   "Change any of them and the manifest test fails until the data are regenerated."}
    (DATA_DIR / "generator_manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
