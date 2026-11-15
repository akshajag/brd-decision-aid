"""Monte Carlo comparison of plans under parameter and risk uncertainty.

Each draw samples every economics/health/intervention parameter from a triangular
(low, value, high) distribution and one column of the risk model's draws, then evaluates
all plans on the SAME draw so differences are paired.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.decision.consequences import plan_totals

FIXED = {"head_per_pen", "amu_stewardship_weight",  # policy/structure, not uncertain
         "pen_clustering_icc"}  # used only by the days 1-14 check-in, not by day-0 outcomes


def sample_params(specs: dict, base: dict, rng: np.random.Generator) -> dict:
    out = dict(base)
    for k, s in specs.items():
        if k in FIXED:
            continue
        lo, hi, mode = float(s["low"]), float(s["high"]), float(base[k])
        lo, hi = min(lo, mode), max(hi, mode)
        out[k] = mode if hi <= lo else float(rng.triangular(lo, mode, hi))
    return out


def compare(lots: pd.DataFrame, risk_draws: np.ndarray, plans: dict[str, list[str]],
            specs: dict, base_params: dict, n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lots = lots.reset_index(drop=True)
    rows = []
    for d in range(n):
        prm = sample_params(specs, base_params, rng)
        p0 = risk_draws[:, rng.integers(risk_draws.shape[1])]
        for name, ch in plans.items():
            t = plan_totals(lots, p0, ch, prm)
            rows.append({"draw": d, "plan": name, "cash_cost": t["cash_cost"], "deaths": t["deaths"],
                         "cases": t["cases"], "regimens": t["regimens"]})
    return pd.DataFrame(rows)


def summarize(mc: pd.DataFrame, plan: str, ref: str) -> dict:
    """Paired comparison of `plan` vs `ref`."""
    a = mc[mc.plan == plan].set_index("draw")
    b = mc[mc.plan == ref].set_index("draw")
    diff = a[["cash_cost", "deaths", "cases", "regimens"]] - b[["cash_cost", "deaths", "cases", "regimens"]]
    return {
        "p_cheaper": float((diff["cash_cost"] < 0).mean()),
        "p_fewer_deaths": float((diff["deaths"] <= 0).mean()),
        "p_fewer_regimens": float((diff["regimens"] <= 0).mean()),
        "cost_diff_q": diff["cash_cost"].quantile([0.05, 0.5, 0.95]).tolist(),
        "deaths_diff_q": diff["deaths"].quantile([0.05, 0.5, 0.95]).tolist(),
        "regimens_diff_q": diff["regimens"].quantile([0.05, 0.5, 0.95]).tolist(),
    }


def thresholds(cfg: dict | None = None) -> tuple[float, float]:
    """Pre-registered verdict cut-offs from config/assumptions.yaml (decision section)."""
    if cfg is None:
        from src.config import load_assumptions
        cfg = load_assumptions()
    d = cfg["decision"]
    return float(d["verdict_min_p_cheaper"]), float(d["verdict_min_p_no_more_deaths"])


def verdict(s: dict, cfg: dict | None = None) -> tuple[bool, list[str]]:
    """Is the recommended plan robust vs the current approach? Returns (confident, failed criteria)."""
    min_cheaper, min_deaths = thresholds(cfg)
    failed = []
    if s["p_cheaper"] < min_cheaper:
        failed.append(f"cheaper in only {100 * s['p_cheaper']:.0f}% of draws (need {100 * min_cheaper:.0f}%)")
    if s["p_fewer_deaths"] < min_deaths:
        failed.append(f"no more deaths in only {100 * s['p_fewer_deaths']:.0f}% of draws "
                      f"(need {100 * min_deaths:.0f}%)")
    return not failed, failed


PLAIN = {"metaphylaxis_morbidity_rrr": "metaphylaxis cuts BRD cases by {:.0f}%",
         "enhanced_monitoring_cfr_rrr": "extra riding cuts BRD deaths among sick cattle by {:.0f}%",
         "receiving_protocol_morbidity_rrr": "low-stress receiving cuts BRD cases by {:.0f}%"}
ROBUST = "robust"
FAVOURABLE = "favourable if assumed effects hold"
LOW = "low confidence"


def corner_check(lots: pd.DataFrame, p0: np.ndarray, plan: list[str], ref: list[str], specs: dict,
                 base_params: dict, cfg: dict | None = None, keys: list[str] | None = None) -> dict:
    """Does `plan` still beat `ref` at every END of the unvalidated intervention-effect ranges? (critic R2-1)

    The Monte Carlo averages over the ranges, so it can say "cheaper in 95% of draws" even when the plan loses
    at a plausible corner (for example, metaphylaxis works better than assumed AND extra riding does nothing).
    Here every combination of low/high ends is evaluated, other parameters at their base values.
    """
    import itertools
    if keys is None:
        if cfg is None:
            from src.config import load_assumptions
            cfg = load_assumptions()
        keys = list(cfg["decision"].get("robust_effect_keys", []))
    keys = [k for k in keys if k in specs]
    worst_cost, worst_deaths, worst_corner, rows = -np.inf, -np.inf, {}, []
    for ends in itertools.product(*[(float(specs[k]["low"]), float(specs[k]["high"])) for k in keys]):
        prm = dict(base_params)
        prm.update(dict(zip(keys, ends)))
        a, b = plan_totals(lots, p0, plan, prm), plan_totals(lots, p0, ref, prm)
        dc, dd = a["cash_cost"] - b["cash_cost"], a["deaths"] - b["deaths"]
        rows.append({**dict(zip(keys, ends)), "cost_diff": dc, "deaths_diff": dd})
        if dc > worst_cost:
            worst_cost, worst_corner = dc, dict(zip(keys, ends))
        worst_deaths = max(worst_deaths, dd)
    if not keys:
        worst_cost = worst_deaths = -np.inf
    return {"worst_cost_diff": float(worst_cost), "worst_deaths_diff": float(worst_deaths),
            "worst_cost_corner": worst_corner, "corners": pd.DataFrame(rows),
            "robust": bool(worst_cost <= 0 and worst_deaths <= 1e-9)}


def break_even(lots: pd.DataFrame, p0: np.ndarray, plan: list[str], ref: list[str], specs: dict,
               base_params: dict, cfg: dict | None = None, keys: list[str] | None = None, n_grid: int = 41) -> pd.DataFrame:
    """One-way threshold analysis: for each unvalidated effect, the value at which the plan stops beating `ref`
    on deaths and on cost, other parameters at their base values. Answers "how wrong can this assumption be
    before the recommendation flips?". A threshold of None means no crossing inside the declared range."""
    if keys is None:
        if cfg is None:
            from src.config import load_assumptions
            cfg = load_assumptions()
        keys = list(cfg["decision"].get("robust_effect_keys", []))
    rows = []
    for k in [k for k in keys if k in specs]:
        lo, hi = float(specs[k]["low"]), float(specs[k]["high"])
        grid = np.linspace(lo, hi, n_grid)
        dd, dc = [], []
        for v in grid:
            prm = dict(base_params); prm[k] = float(v)
            a, b = plan_totals(lots, p0, plan, prm), plan_totals(lots, p0, ref, prm)
            dd.append(a["deaths"] - b["deaths"]); dc.append(a["cash_cost"] - b["cash_cost"])
        out = {"key": k, "default": float(base_params.get(k, specs[k].get("value", np.nan))), "low": lo, "high": hi}
        for name, d in (("deaths", np.array(dd)), ("cost", np.array(dc))):
            thr, side = None, None
            s = np.sign(np.where(np.abs(d) < 1e-9, 0.0, d))
            idx = np.where(s[:-1] * s[1:] < 0)[0]
            if len(idx):
                i = int(idx[0])
                thr = float(grid[i] + (grid[i + 1] - grid[i]) * (0 - d[i]) / (d[i + 1] - d[i]))
                side = "above" if d[i] > 0 else "below"   # plan wins when the effect is above / below thr
            out[f"{name}_threshold"] = thr
            out[f"{name}_plan_wins_when"] = side
            out[f"{name}_wins_everywhere"] = bool(np.all(d <= 1e-9))
            out[f"{name}_loses_everywhere"] = bool(np.all(d > 1e-9))
        rows.append(out)
    return pd.DataFrame(rows)


def narrowing_needed(lots: pd.DataFrame, p0: np.ndarray, plan: list[str], ref: list[str], specs: dict,
                     base_params: dict, cfg: dict | None = None, keys: list[str] | None = None,
                     steps: int = 20) -> dict:
    """How far must ALL unvalidated effect ranges shrink toward their defaults before the plan wins at every
    corner? Returns the smallest shrink fraction f (0 = declared ranges, 1 = defaults only) that makes the
    corner check robust, and the narrowed ranges. f = None means not even the defaults alone are robust."""
    if keys is None:
        if cfg is None:
            from src.config import load_assumptions
            cfg = load_assumptions()
        keys = list(cfg["decision"].get("robust_effect_keys", []))
    keys = [k for k in keys if k in specs]
    for f in np.linspace(0.0, 1.0, steps + 1):
        sp = {}
        for k in keys:
            d = float(base_params.get(k, specs[k].get("value")))
            lo, hi = float(specs[k]["low"]), float(specs[k]["high"])
            sp[k] = {**specs[k], "low": d - (1 - f) * (d - lo), "high": d + (1 - f) * (hi - d)}
        c = corner_check(lots, p0, plan, ref, {**specs, **sp}, base_params, keys=keys)
        if c["robust"]:
            return {"shrink": float(f), "ranges": {k: (sp[k]["low"], sp[k]["high"]) for k in keys}}
    return {"shrink": None, "ranges": {}}


def robust_verdict(s: dict, corner: dict | None, cfg: dict | None = None) -> tuple[str, list[str]]:
    """Three-state verdict (critic R2-1).

    robust: the Monte Carlo criteria pass AND the plan is no costlier and has no more deaths at every end of the
            unvalidated effect ranges. Only this state is shown as "confident".
    favourable if assumed effects hold: the Monte Carlo criteria pass but some corner of the effect ranges loses.
    low confidence: the Monte Carlo criteria fail.
    """
    ok, failed = verdict(s, cfg)
    if not ok:
        return LOW, failed
    if corner is not None and not corner["robust"]:
        c = corner
        msg = []
        if c["worst_cost_diff"] > 0:
            msg.append(f"costs ${c['worst_cost_diff']:,.0f} more if " + ", ".join(
                PLAIN.get(k, k.replace('_', ' ') + ' = {:.2f}').format(100 * v if k in PLAIN else v)
                for k, v in c["worst_cost_corner"].items()))
        if c["worst_deaths_diff"] > 0:
            msg.append(f"up to {c['worst_deaths_diff']:.1f} more deaths at an end of the effect ranges")
        return FAVOURABLE, msg
    return ROBUST, []


def interval_table(mc: pd.DataFrame) -> pd.DataFrame:
    q = mc.groupby("plan")[["cash_cost", "deaths", "cases", "regimens"]].quantile([0.05, 0.5, 0.95])
    return q.unstack()
