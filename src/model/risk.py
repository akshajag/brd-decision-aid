"""Cohort BRD risk model: expected 45-day morbidity WITHOUT arrival metaphylaxis.

- Binomial logistic regression fit on lot-level case counts (cases / non-cases as weights).
- `metaphylaxis_given` is a covariate so history can be adjusted; predictions set it to 0
  ("untreated-equivalent risk"). The app does NOT use this fitted coefficient as the
  intervention effect: that comes from config/assumptions.yaml (a vet-reviewed assumption).
- Uncertainty: bootstrap refits x multiple imputation of missing fields from similar lots
  (same yard and source). Missing records therefore widen intervals instead of being hidden.
- Confidence status per lot: 'confident' | 'borderline' | 'insufficient data'.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

NUMERIC = ["arrival_weight_lb", "head_count", "n_sources", "transit_hours", "shrink_pct",
           "pen_area_m2_per_head", "bunk_cm_per_head"]
CATEGORICAL = {
    "sex": ["heifer", "mixed"],
    "source_type": ["auction", "backgrounded"],
    "vacc_history": ["unknown"],
    "arrival_quarter": ["Q1", "Q3", "Q4"],
    "weather_stress": ["moderate", "high"],
    "yard": ["B", "C", "D"],
}
IMPUTABLE = ["n_sources", "transit_hours", "shrink_pct", "pen_area_m2_per_head", "bunk_cm_per_head"]
CRITICAL = ["n_sources", "transit_hours", "shrink_pct"]
REQUIRED = ["lot_id", "yard", "head_count", "arrival_weight_lb", "sex", "source_type",
            "vacc_history", "arrival_quarter", "weather_stress"]
LABELS = {
    "arrival_weight_lb": "Light arrival weight", "head_count": "Lot size",
    "n_sources": "Commingling (no. of sources)", "transit_hours": "Long transit",
    "shrink_pct": "High shrink", "pen_area_m2_per_head": "Pen space",
    "bunk_cm_per_head": "Bunk space", "sex": "Sex mix", "source_type": "Source (sale barn)",
    "vacc_history": "Unknown vaccination history", "arrival_quarter": "Arrival season",
    "weather_stress": "Weather stress", "yard": "Yard", "missing": "Missing records",
}


def validate_lots(df: pd.DataFrame) -> list[str]:
    """Return human-readable problems; empty list means OK."""
    problems = [f"Missing required column: {c}" for c in REQUIRED if c not in df.columns]
    if problems:
        return problems
    if df["lot_id"].duplicated().any():
        problems.append("Duplicate lot_id values")
    for c in NUMERIC:
        if c in df.columns:
            raw = df[c]
            num = pd.to_numeric(raw, errors="coerce")
            bad = num.isna() & raw.notna() & (raw.astype(str).str.strip() != "")
            if bad.any():
                problems.append(f"{c}: non-numeric values {sorted(raw[bad].astype(str).unique())[:3]} "
                                "(enter numbers only, or leave blank if unknown)")
    w = pd.to_numeric(df["arrival_weight_lb"], errors="coerce")
    bad_w = ~w.between(250, 1200)
    if bad_w.any():
        problems.append(f"{int(bad_w.sum())} lot(s) with arrival weight missing or outside 250-1200 lb")
    bad_h = ~(pd.to_numeric(df["head_count"], errors="coerce") > 0)
    if bad_h.any():
        problems.append(f"{int(bad_h.sum())} lot(s) with missing or non-positive head count")
    allowed = {"sex": {"steer", "heifer", "mixed"},
               "source_type": {"ranch_direct", "backgrounded", "auction"},
               "vacc_history": {"known", "unknown"},
               "arrival_quarter": {"Q1", "Q2", "Q3", "Q4"},
               "weather_stress": {"low", "moderate", "high"},
               "yard": {"A", "B", "C", "D"}}
    for c, ok in allowed.items():
        bad = ~df[c].isin(ok)
        if bad.any():
            problems.append(f"{c}: unexpected values {sorted(df.loc[bad, c].astype(str).unique())[:5]}")
    return problems


PLAUSIBLE = {"transit_hours": (0, 48), "n_sources": (1, 30), "shrink_pct": (0, 12),
             "pen_area_m2_per_head": (5, 40), "bunk_cm_per_head": (10, 60), "head_count": (1, 600)}


def plausibility_warnings(df: pd.DataFrame) -> list[str]:
    """Non-blocking warnings for values outside the range the model was built on."""
    out = []
    for c, (lo, hi) in PLAUSIBLE.items():
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            bad = v.notna() & ~v.between(lo, hi)
            if bad.any():
                out.append(f"{c} outside {lo}-{hi} for lot(s) {', '.join(df.loc[bad, 'lot_id'].astype(str))}: "
                           "the model extrapolates there, so treat their risk with caution")
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types and add any optional columns as missing."""
    out = df.copy()
    for c in IMPUTABLE:
        if c not in out.columns:
            out[c] = np.nan
    for c in NUMERIC:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def design(df: pd.DataFrame, medians: dict[str, float], meta: np.ndarray | None = None) -> pd.DataFrame:
    cols = {}
    for c in NUMERIC:
        v = df[c].to_numpy(float)
        cols[c] = np.where(np.isnan(v), medians[c], v)
    cols["head_count"] = np.log(cols["head_count"])
    for c in IMPUTABLE:
        cols[f"miss_{c}"] = df[c].isna().to_numpy(float)
    for c, levels in CATEGORICAL.items():
        v = df[c].to_numpy()
        for lv in levels:
            cols[f"{c}={lv}"] = (v == lv).astype(float)
    cols["metaphylaxis_given"] = np.zeros(len(df)) if meta is None else np.broadcast_to(np.asarray(meta, float), (len(df),)).copy()
    return pd.DataFrame(cols, index=df.index)


def _group(col: str) -> str:
    if col.startswith("miss_"):
        return "missing"
    return col.split("=")[0]


_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(40)
_GH_W = _GH_W / _GH_W.sum()
# dense grid for integrals that include a sharp binomial likelihood (Gauss-Hermite under-resolves those)
_ZG = np.linspace(-6, 6, 241)
_ZW = norm.pdf(_ZG) / norm.pdf(_ZG).sum()


def _sig(x):
    return 1 / (1 + np.exp(-x))


def marginal_p(lp: np.ndarray, sd: float) -> np.ndarray:
    """E[logistic(lp + u)], u ~ N(0, sd^2)."""
    return _sig(np.asarray(lp, float)[..., None] + sd * _GH_X) @ _GH_W


def lot_centre(p: np.ndarray, sd: float) -> np.ndarray:
    """Inverse of marginal_p: the lot-effect-free logit whose logit-normal MEAN equals p (vectorised bisection)."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    lo, hi = np.full(p.shape, -15.0), np.full(p.shape, 15.0)
    for _ in range(45):
        mid = (lo + hi) / 2
        below = marginal_p(mid, sd) < p
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return (lo + hi) / 2


def prob_high_from_draws(draws: np.ndarray, sd: float, threshold: float) -> np.ndarray:
    """P(a lot's own untreated morbidity >= threshold).

    Integrates over BOTH kinds of uncertainty: the epistemic draws of the lot's expected risk (columns) and the
    unrecorded lot-to-lot spread (logit-normal with SD `sd`, centred so its mean equals each draw).
    One shared definition for the arrival table and the days 1-14 prior (critic D1/D12).
    """
    D = np.atleast_2d(np.asarray(draws, float))
    if D.shape[0] == 1 and np.ndim(draws) == 1:
        D = D.T
    lt = np.log(threshold / (1 - threshold))
    return norm.sf((lt - lot_centre(D, sd)) / sd).mean(axis=1)


def prob_obs_high(p_obs: np.ndarray, head: np.ndarray, sd: float, threshold: float) -> np.ndarray:
    """P(OBSERVED 45-day morbidity >= threshold): lot spread plus binomial sampling (for scoring observed labels)."""
    from scipy.stats import binom
    lc = lot_centre(p_obs, sd)
    q = _sig(lc[:, None] + sd * _ZG[None, :])
    n = np.asarray(head, float)[:, None]
    k = np.ceil(threshold * n - 1e-9)
    return binom.sf(k - 1, n, q) @ _ZW


_POOL_CACHE: dict[int, dict] = {}


def _pools(donors: pd.DataFrame) -> dict:
    """{(column, yard, source): observed values} for donor imputation (cached per donor table)."""
    key = id(donors)
    if key not in _POOL_CACHE:
        pools = {}
        for c in IMPUTABLE:
            allv = donors[c].dropna().to_numpy(float)
            for (y, s), g in donors.groupby(["yard", "source_type"]):
                v = g[c].dropna().to_numpy(float)
                pools[(c, y, s)] = v if len(v) else allv
            pools[(c, None, None)] = allv
        _POOL_CACHE[key] = (donors, pools)   # keep a reference so the id is not reused
    return _POOL_CACHE[key][1]


def _donor_fill(df: pd.DataFrame, donors: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Replace missing IMPUTABLE values by random draws from donors of the same yard and source type."""
    pools = _pools(donors)
    out = df.copy()
    yards, srcs = out["yard"].to_numpy(), out["source_type"].to_numpy()
    for c in IMPUTABLE:
        col = out[c].to_numpy(float).copy()
        miss = np.flatnonzero(np.isnan(col))
        if not len(miss):
            continue
        for y, s in sorted(set(zip(yards[miss], srcs[miss]))):   # sorted: set order varies with hash seed
            idx = miss[(yards[miss] == y) & (srcs[miss] == s)]
            pool = pools.get((c, y, s), pools[(c, None, None)])
            col[idx] = pool[rng.integers(len(pool), size=len(idx))]
        out[c] = col
    return out


# Fields that identify a lot for its imputation random stream: stable under row order and lot renaming, and
# unaffected by the numeric what-if inputs (so changing transit never re-draws a lot's imputed shrink).
_KEY_COLS = ["yard", "source_type", "sex", "vacc_history", "arrival_quarter", "weather_stress", "head_count"]


def _lot_keys(df: pd.DataFrame) -> np.ndarray:
    """One 64-bit key per lot built only from that lot's own fields: categorical fields, head count, arrival weight
    and which fields are missing. Independent of row position, lot ID and the other lots in the upload (critic
    R2-4, R3-4, R4-2, R5-2). Changing transit or another numeric what-if input does not re-draw a lot's imputations;
    editing its weight or filling in a missing field does."""
    import hashlib
    cols = [c for c in _KEY_COLS + ["arrival_weight_lb"] if c in df.columns]
    miss = df[[c for c in IMPUTABLE if c in df.columns]].isna().to_numpy()
    keys = []
    for i, r in enumerate(df[cols].astype(str).itertuples(index=False)):
        s = "|".join(r) + "|" + "".join("1" if m else "0" for m in miss[i])
        keys.append(int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little"))
    return np.array(keys, dtype=np.uint64)


def _lot_imputations(df: pd.DataFrame, donors: pd.DataFrame, n: int, seed: int) -> dict:
    """{column: (n_lots, n) donor draws} for missing cells, from one random stream per lot (critic R2-4)."""
    pools = _pools(donors)
    keys = _lot_keys(df)
    out = {}
    for ci, c in enumerate(IMPUTABLE):
        miss = np.flatnonzero(df[c].isna().to_numpy())
        if not len(miss):
            continue
        arr = np.full((len(df), n), np.nan)
        for i in miss:
            pool = pools.get((c, df["yard"].iat[i], df["source_type"].iat[i]), pools[(c, None, None)])
            rng = np.random.default_rng([int(keys[i]), seed, ci])
            arr[i] = pool[rng.integers(len(pool), size=n)]
        out[c] = arr
    return out


def _fill_from(df: pd.DataFrame, imp: dict, j: int) -> pd.DataFrame:
    out = df.copy()
    for c, arr in imp.items():
        col = out[c].to_numpy(float).copy()
        m = np.isnan(col)
        col[m] = arr[m, j]
        out[c] = col
    return out


# Directions fixed by the literature (and by the simulator): these factors may never LOWER risk when they get
# worse (critic R2-3). Coefficients that come out with the wrong sign are set to zero and the member is refit
# (an active-set sign constraint). Bunk-space direction is a team assumption (ASM).
SIGNS = {"arrival_weight_lb": -1, "transit_hours": 1, "n_sources": 1, "shrink_pct": 1,
         "pen_area_m2_per_head": -1, "bunk_cm_per_head": -1}


def _design_mi(df: pd.DataFrame, filled: pd.DataFrame, medians: dict, meta=None) -> pd.DataFrame:
    """Design matrix from donor-imputed values, keeping the ORIGINAL missingness indicators (same in fit and predict)."""
    X = design(filled, medians, meta)
    for c in IMPUTABLE:
        X[f"miss_{c}"] = df[c].isna().to_numpy(float)
    return X


@dataclass
class RiskModel:
    """Multiple-imputation x bootstrap ensemble. Members 0..n_point-1 are full-data fits on different imputations
    of the training data (their average is the point estimate); the rest are bootstrap members (uncertainty).
    Every member has its own preprocessing (medians, scaler)."""
    pre: list = field(default_factory=list)          # [(medians, mu, sd)] per member
    coefs: list[np.ndarray] = field(default_factory=list)
    intercepts: list[float] = field(default_factory=list)
    resid_sd: float = 1.0
    donors: pd.DataFrame | None = None
    n_impute: int = 10
    n_point: int = 1
    drift: dict = field(default_factory=dict)

    # backwards-compatible views of member 0
    @property
    def medians(self):
        return self.pre[0][0]

    @property
    def mu(self):
        return self.pre[0][1]

    @property
    def sd(self):
        return self.pre[0][2]

    def _logit(self, X: pd.DataFrame, k: int = 0) -> np.ndarray:
        _, mu, sd = self.pre[k]
        Z = ((X[mu.index] - mu) / sd).to_numpy()
        return Z @ self.coefs[k] + self.intercepts[k]

    def predict_mean(self, df: pd.DataFrame, meta: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
        """Point estimate of expected morbidity: average over the full-data members and over donor imputations of
        any missing fields (Rubin-style pooling of predictions). No bootstrap noise, so it is seed-stable.

        meta=None predicts the untreated-equivalent risk; any metaphylaxis column in `df` is ignored.
        """
        df = prepare(df).reset_index(drop=True)
        has_miss = df[IMPUTABLE].isna().any().any()
        m = max(self.n_impute, self.n_point) if has_miss else self.n_point
        imp = _lot_imputations(df, self.donors, m, seed) if has_miss else {}
        acc = np.zeros(len(df))
        for j in range(m):
            k = j % self.n_point
            X = _design_mi(df, _fill_from(df, imp, j), self.pre[k][0], meta)
            acc += _sig(self._logit(X, k))
        return acc / m

    def prob_high(self, p: np.ndarray, threshold: float) -> np.ndarray:
        """P(lot morbidity >= threshold). `p` = a vector of expected risks or an (n_lots, n_draws) draw matrix."""
        return prob_high_from_draws(p, self.resid_sd, threshold)

    @property
    def n_boot(self) -> int:
        return len(self.coefs) - self.n_point

    def draws(self, df: pd.DataFrame, n_draws: int | None = None, seed: int = 1) -> np.ndarray:
        """(n_lots, n_draws) untreated expected-risk draws: one DISTINCT bootstrap member per draw (critic R2-6),
        each with its own donor imputation of any missing fields from the lot's own random stream."""
        df = prepare(df).reset_index(drop=True)
        nb = self.n_boot
        n_draws = nb if (n_draws is None or nb == 0) else min(n_draws, nb) if nb else 1
        if nb == 0:
            n_draws = max(n_draws, 1)
        imp = _lot_imputations(df, self.donors, n_draws, seed)
        out = np.empty((len(df), n_draws))
        for d in range(n_draws):
            k = self.n_point + d if nb else d % self.n_point
            X = _design_mi(df, _fill_from(df, imp, d), self.pre[k][0])
            out[:, d] = _sig(self._logit(X, k))
        return out

    def drivers(self, df: pd.DataFrame, top: int = 3) -> list[list[tuple[str, float]]]:
        """Top risk-increasing factors per lot, as (label, logit contribution vs average lot)."""
        df = prepare(df)
        X = design(df, self.medians)
        Z = (X[self.mu.index] - self.mu) / self.sd
        contrib = Z * self.coefs[0]
        contrib = contrib.drop(columns=["metaphylaxis_given"])
        grouped = contrib.T.groupby(_group).sum().T
        res = []
        for _, row in grouped.iterrows():
            r = row.sort_values(ascending=False)
            res.append([(LABELS.get(k, k), float(v)) for k, v in r.items() if v > 0.05][:top])
        return res


def _fit_member(df: pd.DataFrame, idx: np.ndarray, donors: pd.DataFrame, rng: np.random.Generator, C: float):
    """One ensemble member: resample rows `idx`, refit preprocessing, impute from donors, fit."""
    d = df.iloc[idx].reset_index(drop=True)
    medians = {c: float(d[c].median()) for c in NUMERIC}
    X = _design_mi(d, _donor_fill(d, donors, rng), medians, d["metaphylaxis_given"].to_numpy(float))
    mu, sd = X.mean(), X.std().replace(0, 1.0)
    Z = ((X - mu) / sd).to_numpy()
    cases = d["brd_cases_45d"].to_numpy(float)
    non = d["head_count"].to_numpy(float) - cases
    y = np.r_[np.ones(len(d)), np.zeros(len(d))]
    w = np.r_[cases, non]
    cols = list(X.columns)
    fixed: set[str] = set()
    for _ in range(len(SIGNS) + 1):
        Zf = Z.copy()
        for c in fixed:
            Zf[:, cols.index(c)] = 0.0
        m = LogisticRegression(C=C, max_iter=3000)
        m.fit(np.vstack([Zf, Zf]), y, sample_weight=w / w.mean())
        coef = m.coef_[0].copy()
        bad = [c for c, s in SIGNS.items() if c not in fixed and coef[cols.index(c)] * s < 0]
        if not bad:
            break
        fixed |= set(bad)
    for c in fixed:
        coef[cols.index(c)] = 0.0
    return (medians, mu, sd), coef, float(m.intercept_[0])


def estimate_lot_sd(cases: np.ndarray, head: np.ndarray, p0: np.ndarray, mult: np.ndarray | None = None) -> float:
    """MLE of the lot-level logit SD given (out-of-fold) expected observed morbidity p_obs.

    Model: k ~ Binomial(n, mult * logistic(c + sd*z)), z ~ N(0,1), with c chosen so the mean of the
    UNTREATED risk equals p0; mult = (expected observed / expected untreated) carries metaphylaxis
    (a random-intercept logistic model profiled on a fixed marginal mean). Replaces the old in-sample
    moment estimate that understated the spread (critic D13).
    """
    from scipy.optimize import minimize_scalar
    k = np.asarray(cases, float)[:, None]
    n = np.asarray(head, float)[:, None]
    mult = np.ones(len(k)) if mult is None else np.asarray(mult, float)

    from scipy.special import gammaln
    const = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    logw = np.log(_ZW)[None, :]

    def nll(sd):
        q = np.clip(_sig(lot_centre(p0, sd)[:, None] + sd * _ZG[None, :]) * mult[:, None], 1e-12, 1 - 1e-12)
        ll = const + k * np.log(q) + (n - k) * np.log1p(-q) + logw
        mx = ll.max(axis=1, keepdims=True)
        return -np.sum(mx[:, 0] + np.log(np.exp(ll - mx).sum(axis=1)))
    r = minimize_scalar(nll, bounds=(0.2, 3.0), method="bounded", options={"xatol": 1e-3})
    return float(r.x)


def fit(hist: pd.DataFrame, n_boot: int = 100, C: float = 1.0, seed: int = 0, oof_folds: int = 5,
        n_point: int = 10, drift_check: bool = True) -> RiskModel:
    """Fit the ensemble: n_point full-data members (different imputations of the training data; their average is
    the point estimate) and n_boot bootstrap members (the uncertainty). Every member refits its own medians/scaler
    and uses its own donor imputation of its training rows (critic D2/D21)."""
    hist = prepare(hist).reset_index(drop=True)
    donors = hist[["yard", "source_type"] + IMPUTABLE].copy()
    rng = np.random.default_rng(seed)
    pre, coefs, ints = [], [], []
    for b in range(n_point + n_boot):
        idx = np.arange(len(hist)) if b < n_point else rng.integers(0, len(hist), len(hist))
        # point members use fixed streams: the point estimate is a function of the data only, and `seed` controls
        # only the bootstrap (critic R2-7). Their imputation Monte Carlo error falls with n_point.
        mrng = np.random.default_rng([20261115, b]) if b < n_point else rng
        pp, c, i = _fit_member(hist, idx, donors, mrng, C)
        pre.append(pp)
        coefs.append(c)
        ints.append(i)
    model = RiskModel(pre, coefs, ints, donors=donors, n_point=n_point)
    # Lot-level SD, estimated OUT OF FOLD so in-sample overfit cannot shrink it.
    meta = hist["metaphylaxis_given"].to_numpy(float)
    p_oof = np.empty(len(hist))
    p0_oof = np.empty(len(hist))
    folds = np.random.default_rng(seed + 1).permutation(len(hist)) % max(oof_folds, 2)
    for f in range(max(oof_folds, 2)):
        tr, te = folds != f, folds == f
        sub_d = hist[tr].reset_index(drop=True)
        pp, c, i = _fit_member(sub_d, np.arange(len(sub_d)), sub_d[["yard", "source_type"] + IMPUTABLE], rng, C)
        sub = RiskModel([pp], [c], [i], donors=sub_d[["yard", "source_type"] + IMPUTABLE])
        p_oof[te] = sub.predict_mean(hist[te], meta=meta[te])
        p0_oof[te] = sub.predict_mean(hist[te])
    model.resid_sd = estimate_lot_sd(hist["brd_cases_45d"].to_numpy(), hist["head_count"].to_numpy(),
                                     p0_oof, p_oof / p0_oof)
    if drift_check and "arrival_date" in hist.columns:
        model.drift = check_drift(model, hist, rng, C)
        if model.drift.get("alarm"):
            model.intercepts = [i + model.drift["offset"] for i in model.intercepts]
    return model


def check_drift(model: RiskModel, hist: pd.DataFrame, rng: np.random.Generator, C: float = 1.0,
                window_days: int | None = None, z_crit: float | None = None) -> dict:
    """Has risk shifted recently? (critic R2-8)

    A model fitted on lots BEFORE the most recent window predicts the window's lots (out of sample). We compare
    observed with expected BRD cases: z = sum(k - n p) / sqrt(sum (k - n p)^2), with lot-level residuals so the
    lot-to-lot spread is included (a logit-scale residual would be biased by that spread). If |z| > z_crit we raise
    an alarm and shift every member's intercept so the full model reproduces the window's total cases (a
    recalibration-in-the-large; conservative, because the full model has already partly absorbed the shift).
    Cut-offs are in config.
    """
    if window_days is None or z_crit is None:
        from src.config import load_assumptions
        dec = load_assumptions()["decision"]
        window_days = dec.get("drift_window_days", 180) if window_days is None else window_days
        z_crit = dec.get("drift_alarm_z", 2.5) if z_crit is None else z_crit
    from scipy.optimize import brentq
    d = pd.to_datetime(hist["arrival_date"])
    recent = (d > d.max() - pd.Timedelta(days=window_days)).to_numpy()
    if recent.sum() < 50 or (~recent).sum() < 200:
        return {"alarm": False, "ran": False}
    meta = hist["metaphylaxis_given"].to_numpy(float)
    win = hist[recent].reset_index(drop=True)
    k = win["brd_cases_45d"].to_numpy(float)
    n = win["head_count"].to_numpy(float)
    pre = hist[~recent].reset_index(drop=True)
    dn = pre[["yard", "source_type"] + IMPUTABLE]
    pp, c, i = _fit_member(pre, np.arange(len(pre)), dn, rng, C)
    p_out = RiskModel([pp], [c], [i], donors=dn).predict_mean(win, meta=meta[recent])
    r = k - n * p_out
    z = float(r.sum() / np.sqrt((r ** 2).sum()))
    p_in = np.clip(model.predict_mean(win, meta=meta[recent]), 1e-6, 1 - 1e-6)
    lg = np.log(p_in / (1 - p_in))
    offset = float(brentq(lambda o: (n / (1 + np.exp(-(lg + o)))).sum() - k.sum(), -3, 3))
    return {"ran": True, "window_days": window_days, "n_window": int(recent.sum()), "z": z,
            "observed_cases": float(k.sum()), "expected_cases_out_of_sample": float((n * p_out).sum()),
            "offset": offset, "alarm": bool(abs(z) > z_crit)}


def recentre(D: np.ndarray, p: np.ndarray, iters: int = 4) -> np.ndarray:
    """Shift each lot's draws on the logit scale so their mean equals the point estimate p (critic D11 + R2-7).

    The bootstrap supplies the SPREAD; the multiply-imputed full-data fit supplies the seed-stable CENTRE.
    Newton steps on a per-lot logit shift solve mean(sigmoid(logit D + delta)) = p.
    """
    L = np.log(np.clip(D, 1e-9, 1 - 1e-9) / (1 - np.clip(D, 1e-9, 1 - 1e-9)))
    delta = np.zeros((D.shape[0], 1))
    for _ in range(iters):
        s = _sig(L + delta)
        g = s.mean(axis=1, keepdims=True) - p[:, None]
        h = (s * (1 - s)).mean(axis=1, keepdims=True)
        delta -= g / np.maximum(h, 1e-9)
    return _sig(L + delta)


def assess(model: RiskModel, lots: pd.DataFrame, cfg: dict, n_draws: int | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-lot risk table used by the dashboard and the optimizer, plus the (n_lots, n_draws) risk draws."""
    lots = prepare(lots).reset_index(drop=True)
    dec = cfg["decision"]
    thr_hi, thr_mod = dec["high_risk_morbidity"], dec["moderate_risk_morbidity"]
    tail = 100 * (1 - dec.get("interval_level", 0.90)) / 2
    D = model.draws(lots, n_draws=n_draws)
    # Point = multiply-imputed full-data estimate (seed-stable, critic R2-7). It is an estimate of the mean of the
    # draws (outcomes are linear in p, so a mean is the right plug-in, critic D11) without bootstrap noise.
    p = model.predict_mean(lots)
    D = recentre(D, p)
    # Weibull plotting positions p(n+1) (Hyndman-Fan type 6): the expected coverage of the range is exactly the
    # nominal level for exchangeable draws; numpy's default gives only ~86% with 40 draws (critic D2)
    lo = np.percentile(D, tail, axis=1, method="weibull")
    hi = np.percentile(D, 100 - tail, axis=1, method="weibull")
    n_missing = lots[CRITICAL].isna().sum(axis=1).to_numpy()
    tier = np.where(p >= thr_hi, "High", np.where(p >= thr_mod, "Moderate", "Low"))
    # borderline: the uncertainty interval crosses the high-risk threshold that drives the decision
    straddle = (lo < thr_hi) & (hi >= thr_hi)
    status = np.where(n_missing >= dec.get("insufficient_data_min_missing", 2), "insufficient data",
                      np.where(straddle, "borderline", "confident"))
    drv = model.drivers(lots)
    out = lots[["lot_id", "yard", "head_count", "arrival_weight_lb", "source_type"]].copy()
    out["risk_mean"] = p
    out["risk_lo"] = lo
    out["risk_hi"] = hi
    out["p_high"] = model.prob_high(D, thr_hi)
    out["tier"] = tier
    out["confidence"] = status
    out["missing_critical"] = n_missing
    out["drivers"] = ["; ".join(f"{k}" for k, _ in d) or "No strong drivers" for d in drv]
    return out, D


def adequacy(obs: np.ndarray, pred: np.ndarray) -> dict:
    """Observed-vs-predicted adequacy statistics for a continuous outcome.

    * CCC: Lin's concordance correlation coefficient = precision (Pearson r) x accuracy (bias factor Cb).
    * MSEP decomposition: mean squared error of prediction split into mean bias, slope bias
      (systematic) and random error (unsystematic).
    * Regression of observed on predicted: ideal intercept 0, slope 1.
    """
    o, p = np.asarray(obs, float), np.asarray(pred, float)
    mo, mp = o.mean(), p.mean()
    so, sp = o.std(), p.std()
    r = float(np.corrcoef(o, p)[0, 1])
    ccc = 2 * r * so * sp / (so ** 2 + sp ** 2 + (mo - mp) ** 2)
    msep = float(np.mean((o - p) ** 2))
    mean_bias = (mp - mo) ** 2
    slope_bias = (sp - r * so) ** 2
    random = (1 - r ** 2) * so ** 2
    slope, intercept = np.polyfit(p, o, 1)
    return {"ccc": float(ccc), "pearson_r": r, "accuracy_cb": float(ccc / r) if r else float("nan"),
            "rmsep": float(np.sqrt(msep)), "msep": msep,
            "share_mean_bias": mean_bias / msep if msep > 0 else 0.0,
            "share_slope_bias": slope_bias / msep if msep > 0 else 0.0,
            "share_random": random / msep if msep > 0 else 0.0, "slope": float(slope), "intercept": float(intercept)}


def _cal_slope(cases, head, p) -> tuple[float, float]:
    from scipy.optimize import minimize
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    k, n = np.asarray(cases, float), np.asarray(head, float)

    def nll(w):
        q = np.clip(_sig(w[0] + w[1] * lp), 1e-9, 1 - 1e-9)
        return -np.sum(k * np.log(q) + (n - k) * np.log(1 - q))
    r = minimize(nll, np.array([0.0, 1.0]), method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-6})
    return float(r.x[1]), float(r.x[0])


def ccc_ceiling(p_obs: np.ndarray, head: np.ndarray, sd: float, n_sim: int = 200, seed: int = 3,
                p0: np.ndarray | None = None) -> float:
    """CCC that a PERFECT expected-risk model would score against observed morbidity on these lots
    (lot spread + binomial noise make 1.0 unreachable; critic D9). Simulated under the fitted lot SD."""
    rng = np.random.default_rng(seed)
    p0 = p_obs if p0 is None else p0
    mult = np.asarray(p_obs, float) / np.asarray(p0, float)     # metaphylaxis acts on untreated risk
    lc = lot_centre(p0, sd)
    vals = []
    for _ in range(n_sim):
        q = mult * _sig(lc + rng.normal(0, sd, len(lc)))
        obs = rng.binomial(np.asarray(head, int), q) / np.asarray(head, float)
        vals.append(adequacy(obs, p_obs)["ccc"])
    return float(np.mean(vals))


def evaluate(hist: pd.DataFrame, cfg: dict, cutoff: str = "2026-01-01", n_boot: int = 0) -> dict:
    """TEMPORAL hold-out evaluation (train before `cutoff`, test on or after) on SYNTHETIC data.

    Circular by construction: it shows the pipeline recovers the process that generated the data, not
    real-world accuracy. The observed label is post-treatment morbidity >= 15%, so predictions are scored
    with each lot's actual metaphylaxis status (the decision itself uses untreated risk; see DESIGN s5).
    """
    hist = prepare(hist).reset_index(drop=True)
    d = pd.to_datetime(hist["arrival_date"])
    test = (d >= pd.Timestamp(cutoff)).to_numpy()
    m = fit(hist[~test], n_boot=n_boot)
    te = hist[test].reset_index(drop=True)
    meta = te["metaphylaxis_given"].to_numpy(float)
    p = m.predict_mean(te, meta=meta)
    thr = cfg["decision"]["high_risk_morbidity"]
    head = te["head_count"].to_numpy(float)
    cases = te["brd_cases_45d"].to_numpy(float)
    ph = prob_obs_high(p, head, m.resid_sd, thr)
    y = te["high_risk"].to_numpy()
    pred = ph >= 0.5
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    obs = te["morbidity_45d"].to_numpy()
    bins = pd.qcut(p, 10, duplicates="drop")
    calib = (pd.DataFrame({"pred": p, "obs": obs, "bin": bins})
             .groupby("bin", observed=True)[["pred", "obs"]].mean().reset_index(drop=True))
    slope, icpt = _cal_slope(cases, head, p)
    ad = adequacy(obs, p)
    ad["ccc_ceiling"] = ccc_ceiling(p, head, m.resid_sd, p0=m.predict_mean(te))
    return {
        "split": "temporal", "cutoff": cutoff,
        "n_train": int((~test).sum()), "n_test": int(test.sum()), "prevalence": float(y.mean()),
        "auc": float(roc_auc_score(y, ph)), "brier": float(brier_score_loss(y, ph)),
        "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
        "ppv": tp / max(tp + fp, 1), "npv": tn / max(tn + fn, 1),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}, "calibration": calib,
        "cal_in_the_large": float(cases.sum() / head.sum() - np.sum(p * head) / head.sum()),
        "cal_slope": slope, "cal_intercept": icpt, "resid_sd": m.resid_sd,
        "adequacy": ad,
        "obs_pred": pd.DataFrame({"pred": p, "obs": obs}),
    }


def fit_cached(hist_path, cache_dir, **kw) -> RiskModel:
    """fit() with a local disk cache, so the dashboard starts instantly after a restart.

    The cache key hashes the training CSV, config/assumptions.yaml, this module's source and the fit arguments,
    so any change to data, parameters or model code forces a refit. Only files this function wrote are loaded."""
    import hashlib
    import pickle
    from pathlib import Path
    hist_path, cache_dir = Path(hist_path), Path(cache_dir)
    root = Path(__file__).resolve().parents[2]
    h = hashlib.sha256()
    for f in (hist_path, root / "config" / "assumptions.yaml", Path(__file__).resolve()):
        h.update(f.read_bytes())
    h.update(repr(sorted(kw.items())).encode())
    path = cache_dir / f"risk_model_{h.hexdigest()[:16]}.pkl"
    if path.exists():
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass  # corrupt or incompatible cache: refit below
    model = fit(pd.read_csv(hist_path), **kw)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for old in cache_dir.glob("risk_model_*.pkl"):
            old.unlink(missing_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(model, fh)
        tmp.replace(path)
    except OSError:
        pass  # read-only checkout: still works, just without the cache
    return model
