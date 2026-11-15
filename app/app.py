"""BRD Arrival Decision Aid - Streamlit dashboard.

Run:  streamlit run app/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.templates["ranch"] = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FBF6EC", font=dict(color="#2B2118"),
    xaxis=dict(gridcolor="#E3D3B6", zerolinecolor="#C9B28C"), yaxis=dict(gridcolor="#E3D3B6", zerolinecolor="#C9B28C")))
pio.templates.default = "plotly+ranch"
import streamlit as st

from src.config import load_assumptions, param_specs, param_values
from src.data.generate import ARRIVAL_COLUMNS, config_hash
import base64
import json

from src.decision import followup as fu
from src.decision import uncertainty as unc
from src.decision.consequences import COST_PARTS, OPTIONS, cost_breakdown, plan_totals
from src.decision.optimize import BASELINE_RULES, capacities, optimize
from src.decision.summary import build_summary
from src.model import risk as riskmod
from src.model.animals import animal_view

st.set_page_config(page_title="BRD Arrival Decision Aid", layout="wide")

DATA = ROOT / "data"
CURRENT = "Current approach"
RECOMMENDED = "Recommended plan"
TIER_COLORS = {"High": "#c0392b", "Moderate": "#e67e22", "Low": "#2e86c1"}
SLIDER_KEYS = ["metaphylaxis_morbidity_rrr", "enhanced_monitoring_cfr_rrr", "receiving_protocol_morbidity_rrr",
               "case_fatality_rate", "production_loss_per_case", "treatment_cost_per_case",
               "metaphylaxis_cost_per_head", "feeder_price_per_lb", "labor_cost_per_hour",
               "amu_stewardship_weight", "hospital_days_per_case", "case_onset_spread_days", "pen_clustering_icc"]
FRIENDLY = {
    "metaphylaxis_morbidity_rrr": "Metaphylaxis: share of BRD cases prevented",
    "enhanced_monitoring_cfr_rrr": "Extra monitoring: share of BRD deaths prevented",
    "receiving_protocol_morbidity_rrr": "Low-stress receiving: share of BRD cases prevented",
    "case_fatality_rate": "Share of sick cattle that die",
    "production_loss_per_case": "Lost performance per sick animal ($)",
    "treatment_cost_per_case": "Treatment cost per sick animal ($)",
    "metaphylaxis_cost_per_head": "Metaphylaxis cost per head ($)",
    "feeder_price_per_lb": "Feeder cattle price ($/lb)",
    "labor_cost_per_hour": "Rider labour ($/hour)",
    "amu_stewardship_weight": "Antimicrobial stewardship penalty ($ per course)",
    "hospital_days_per_case": "Days in hospital pen per sick animal",
    "case_onset_spread_days": "Days over which new BRD cases appear",
    "pen_clustering_icc": "Check-in sensitivity: extra clustering of early pulls (0 = off)",
    "death_loss_extra_cost": "Extra cost per death ($)",
    "retreat_fraction": "Re-treatments per sick animal",
    "receiving_protocol_cost_per_head": "Low-stress receiving cost per head ($)",
    "enhanced_monitoring_days": "Days of extra monitoring",
    "enhanced_monitoring_hours_per_pen_day": "Extra rider hours per pen per day",
    "head_per_pen": "Head per pen",
}
OPTION_SHORT = {"STD": "Standard care", "META": "Metaphylaxis", "ENH": "Extra monitoring",
                "ENH_META": "Metaphylaxis + extra monitoring"}


def friendly(k):
    return FRIENDLY.get(k, k.replace("_", " "))


# ---------- cached computation ----------
@st.cache_resource(show_spinner="Fitting risk model on historical (synthetic) lots...")
def get_model():
    hist = pd.read_csv(DATA / "historical_cohorts.csv")
    return riskmod.fit_cached(DATA / "historical_cohorts.csv", ROOT / ".cache"), hist


@st.cache_data(show_spinner="Evaluating model on held-out lots...")
def get_eval():
    cfg = load_assumptions()
    hist = pd.read_csv(DATA / "historical_cohorts.csv")
    return riskmod.evaluate(hist, cfg)


@st.cache_data(show_spinner="Scoring lots (bootstrap x imputation)...")
def get_risk(lots: pd.DataFrame):
    model, _ = get_model()
    return riskmod.assess(model, lots, load_assumptions())


@st.cache_data(show_spinner=False)
def get_animals(lot_dict: dict, lot_risk: float):
    model, _ = get_model()
    return animal_view(model, pd.Series(lot_dict), lot_risk, load_assumptions())


@st.cache_data(show_spinner="Running Monte Carlo comparison...")
def get_mc(lots, draws, plans_items, params_items, n=200):
    specs = param_specs(load_assumptions())
    return unc.compare(lots, draws, dict(plans_items), specs, dict(params_items), n=n)


def fmt_money(x):
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


# ---------- sidebar: data, assumptions, constraints ----------
cfg = load_assumptions()
specs = param_specs(cfg)
defaults = param_values(cfg)

SCENARIO_PREFIXES = ("base_rule", "base_cut", "stepdown", "p_", "r_", "h_", "wx_", "use_cap", "amu_cap")
WX_CHOICES = ["as recorded", "low", "moderate", "high"]


def _scenario_value(k, v):
    """Validated value for a saved-scenario key, or None if it would break a widget."""
    num = isinstance(v, (int, float)) and not isinstance(v, bool)
    if k.startswith("p_") and k[2:] in SLIDER_KEYS and num:
        s = specs[k[2:]]
        return float(v) if float(s["low"]) <= v <= float(s["high"]) else None
    if k == "base_rule":
        return v if v in BASELINE_RULES else None
    if k == "base_cut":
        return int(v) if num and 450 <= v <= 900 else None
    if k in ("stepdown", "use_cap"):
        return v if isinstance(v, bool) else None
    if k == "amu_cap":
        return int(v) if num and 0 <= v <= 20000 else None
    if k[:2] in ("r_", "h_") and k[2:] in cfg["yards"] and num:
        return float(v) if 0 <= v <= (50.0 if k[0] == "r" else 200.0) else None
    if k.startswith("wx_") and k[3:] in cfg["yards"]:
        return v if v in WX_CHOICES else None
    return None


if "_pending_scenario" in st.session_state:
    _applied, _skipped = 0, []
    for _k, _v in st.session_state.pop("_pending_scenario").items():
        _val = _scenario_value(_k, _v) if isinstance(_k, str) else None
        if _val is None:
            _skipped.append(str(_k))
        else:
            st.session_state[_k] = _val
            _applied += 1
    _msg = f"Scenario loaded: {_applied} settings applied."
    if _skipped:
        _msg += " Skipped (unknown or out of range): " + ", ".join(_skipped[:8]) + ("..." if len(_skipped) > 8 else "")
    if st.session_state.get("stepdown"):
        _msg += " NOTE: the veterinarian high-risk step-down override is ON in this scenario."
    st.session_state["_scenario_msg"] = _msg

st.sidebar.title("BRD Decision Aid")
st.sidebar.caption("Arrival decisions for the first 45 days on feed")

src = st.sidebar.radio("Arrival data", ["Demo week (synthetic)", "Upload CSV"],
                       help="Demo data is synthetic, generated by src/data/generate.py. Upload uses the same columns.")
data_label = "Synthetic demonstration lots (not real yard records)"
if src == "Upload CSV":
    up = st.sidebar.file_uploader("Arrival lots CSV", type="csv")
    st.sidebar.download_button("Download CSV template",
                               pd.read_csv(DATA / "current_arrivals.csv").head(3).to_csv(index=False),
                               "arrival_template.csv", "text/csv")
    if up is None:
        st.info("Upload a CSV with the template columns, or switch to the demo week.")
        st.stop()
    raw = pd.read_csv(up)
    data_label = f"Uploaded file: {up.name}"
else:
    raw = pd.read_csv(DATA / "current_arrivals.csv")

with st.sidebar.expander("Current operating approach", expanded=False):
    st.caption("The comparator every option is measured against (ASM - set it to what your yard actually does). "
               "For context: NAHMS 2019 reports 15.6% of US feedlot cattle were group-treated with an "
               "injectable antimicrobial in 2016.")
    base_rule = st.selectbox("Rule", list(BASELINE_RULES), format_func=BASELINE_RULES.get, key="base_rule")
    base_cut = st.slider("Cut-off weight (lb)", 450, 900, int(cfg["decision"]["baseline_meta_weight_lb"]), 25,
                         key="base_cut")
baseline_desc = f"{BASELINE_RULES[base_rule]} ({base_cut} lb); standard pen riding otherwise"
allow_stepdown = st.sidebar.checkbox(
    "Veterinarian allows step-down of high-risk lots", value=False, key="stepdown",
    help="Off (default): lots whose whole 90% risk range is above 15% always keep metaphylaxis. "
         "On: the optimiser may substitute enhanced monitoring; each such lot is flagged for sign-off.")

with st.sidebar.expander("Assumptions (what-if)", expanded=False):
    st.caption("EST = published estimate from another population · ASM = our assumption (no direct evidence). "
               "0.45 means 45%. Hover the ? for the source and range.")
    params = dict(defaults)
    for k in SLIDER_KEYS:
        s = specs[k]
        lo, hi, v = float(s["low"]), float(s["high"]), float(s["value"])
        step = 0.01 if hi <= 1 else (0.05 if hi <= 10 else 1.0)
        params[k] = st.slider(f"{friendly(k)} [{s['tag']}]", lo, hi, v, step,
                              help=f"`{k}` - {s['unit']}. Source: {s['source']}. {s.get('note', '')}", key=f"p_{k}")
changed = [k for k in SLIDER_KEYS if abs(params[k] - defaults[k]) > 1e-9]

with st.sidebar.expander("Operating constraints", expanded=False):
    st.caption("Scenario capacities per yard (ASM). Hospital = free head places after current occupancy.")
    base_caps = capacities(cfg)
    caps = {}
    for y, c in base_caps.items():
        c1, c2 = st.columns(2)
        caps[y] = {
            "rider_h_day": c1.number_input(f"{y}: rider h/day", 0.0, 50.0, c["rider_h_day"], 0.5, key=f"r_{y}"),
            "hosp_free": c2.number_input(f"{y}: hospital head", 0.0, 200.0, round(c["hosp_free"], 1), 1.0, key=f"h_{y}"),
        }
    use_cap = st.checkbox("Cap antimicrobial regimens (stewardship target)", key="use_cap")
    amu_cap = st.number_input("Max regimens this week", 0, 20000, 2500, 50, key="amu_cap") if use_cap else None

with st.sidebar.expander("Weather forecast (per yard)", expanded=False):
    st.caption("Manager's forecast of weather stress over the next 2 weeks (manual input; no external weather feed "
               "is connected). Overrides each lot's recorded weather stress for that yard.")
    wx = {y: st.selectbox(f"Yard {y}", WX_CHOICES, key=f"wx_{y}") for y in cfg["yards"]}

with st.sidebar.expander("Save / load scenario", expanded=False):
    st.caption("Saves the sidebar settings (baseline, assumptions, capacities, weather) - not the lot data.")
    scen = {k: v for k, v in st.session_state.items()
            if isinstance(k, str) and k.startswith(SCENARIO_PREFIXES) and isinstance(v, (int, float, str, bool))}
    st.download_button("Save scenario (JSON)", json.dumps(scen, indent=1, sort_keys=True), "brd_scenario.json",
                       "application/json")
    if "_scenario_msg" in st.session_state:
        st.info(st.session_state["_scenario_msg"])
    scen_up = st.file_uploader("Load scenario", type="json", key="scenario_file")
    if scen_up is not None and st.session_state.get("_scenario_applied") != scen_up.file_id:
        try:
            loaded = json.loads(scen_up.getvalue().decode("utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("not a JSON object")
            st.session_state["_scenario_applied"] = scen_up.file_id
            st.session_state["_pending_scenario"] = loaded
            st.rerun()
        except (ValueError, UnicodeDecodeError) as e:
            st.error(f"Could not read scenario: {e}")

if st.sidebar.button("Reset all inputs"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# ---------- header ----------
_hero_img = ROOT / "app" / "assets" / "cattle_herd.jpg"
_hero_bg = (f"url(data:image/jpeg;base64,{base64.b64encode(_hero_img.read_bytes()).decode()})"
            if _hero_img.exists() else "none")
st.markdown(f"""
<style>
.ranch-hero {{position:relative; min-height:250px; border-radius:14px; overflow:hidden; margin-bottom:8px;
  background-color:#3A2A1E;
  background-image:linear-gradient(90deg, rgba(30,20,12,0.88) 0%, rgba(30,20,12,0.62) 45%, rgba(30,20,12,0.10) 100%), {_hero_bg};
  background-size:cover; background-position:center 38%; border-bottom:5px solid #8B3A1E;}}
.ranch-hero .txt {{position:relative; padding:34px 34px 26px 34px; max-width:640px;}}
.ranch-hero h1 {{color:#F7F0E2; margin:0; padding:0; font-size:2.3rem; line-height:1.12; letter-spacing:-0.01em;}}
.ranch-hero p {{color:#EADBC0; margin:8px 0 0 0; font-size:1.02rem; line-height:1.45;}}
.ranch-hero .tag {{display:inline-block; background:#E9B44C; color:#2B2118; font-weight:600; font-size:0.8rem;
  padding:3px 12px; border-radius:20px; margin-top:14px;}}
.ranch-hero .credit {{position:absolute; right:12px; bottom:8px; font-size:0.68rem; color:rgba(247,240,226,0.85);}}
div[data-baseweb="tab-list"] button p {{font-weight:600;}}
section[data-testid="stSidebar"] {{border-right:4px solid #C9B28C;}}
</style>
<div class="ranch-hero"><div class="txt">
  <h1>BRD Arrival Decision Aid</h1>
  <p>Receiving decisions for the first 45 days on feed across four yards in the Texas Panhandle and western Kansas:
  which lots need attention, what to do, what it costs, and whether the crew can deliver it.</p>
  <span class="tag">Synthetic demo data &middot; medication changes require veterinary review</span></div>
  <div class="credit">Photo: Keith Weller, USDA Agricultural Research Service (public domain)</div>
</div>
""", unsafe_allow_html=True)
st.warning("**Demonstration build.** Lots and outcomes are **synthetic**; intervention effects are **assumptions** "
           "you can change, not validated results. **Medication-related recommendations require veterinary review.**")

tabs = st.tabs(["Start here", "1 · Lots & data", "2 · Risk priorities", "3 · Plan vs current approach",
                "4 · What-if & sensitivity", "5 · Decision summary", "6 · Days 1-14 check-in",
                "7 · Model & evidence"])

# ---------- tab 1: data ----------
with tabs[1]:
    st.subheader("Arriving lots")
    st.caption("Edit cells directly (e.g. fill in a missing transit time) - every tab recomputes. "
               "Empty cells are missing records; they widen uncertainty instead of being hidden.")
    first = ["lot_id", "yard", "head_count", "arrival_weight_lb", "source_type"] + riskmod.CRITICAL
    cols = [c for c in first if c in raw.columns] + [c for c in ARRIVAL_COLUMNS if c in raw.columns and c not in first]
    edited = st.data_editor(raw[cols], num_rows="dynamic", use_container_width=True, key="editor",
                            column_config={"yard": st.column_config.SelectboxColumn(options=["A", "B", "C", "D"])})
    edited = edited[edited["lot_id"].notna() & (edited["lot_id"].astype(str).str.strip() != "")] \
        if "lot_id" in edited.columns else edited
    problems = riskmod.validate_lots(edited)
    if problems:
        st.error("Data problems - fix before analysis:\n\n- " + "\n- ".join(problems))
        st.stop()
    lots = riskmod.prepare(edited).reset_index(drop=True)
    wx_set = {y: v for y, v in wx.items() if v != "as recorded"}
    for y, v in wx_set.items():
        lots.loc[lots["yard"] == y, "weather_stress"] = v
    if wx_set:
        st.info("Weather forecast override applied: " + ", ".join(f"yard {y} = {v}" for y, v in wx_set.items()))
    for wmsg in riskmod.plausibility_warnings(lots):
        st.warning(wmsg)
    yards_sel = st.multiselect("Yards in this decision", sorted(lots["yard"].unique()),
                               default=sorted(lots["yard"].unique()))
    lots = lots[lots["yard"].isin(yards_sel)].reset_index(drop=True)
    if lots.empty:
        st.stop()
    miss = lots[riskmod.CRITICAL].isna().sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lots", len(lots))
    c2.metric("Head", f"{int(lots['head_count'].sum()):,}")
    c3.metric("Lots with missing key records", int((lots[riskmod.CRITICAL].isna().sum(axis=1) > 0).sum()))
    c4.metric("Unknown vaccination history", int((lots["vacc_history"] == "unknown").sum()))
    st.caption("Missing key fields: " + ", ".join(f"{k} {v}" for k, v in miss.items()))
    with st.expander("Yard profiles (scenario assumptions)"):
        st.dataframe(pd.DataFrame([{"yard": y, "name": Y["name"], "profile": Y["profile"],
                                    "rider h/day": caps[y]["rider_h_day"], "free hospital head": caps[y]["hosp_free"]}
                                   for y, Y in cfg["yards"].items()]), hide_index=True, use_container_width=True)

# ---------- shared computation ----------
risk, draws = get_risk(lots)
opt_kw = dict(caps=caps, amu_cap=amu_cap, baseline_rule=base_rule, baseline_cutoff=base_cut,
              allow_high_stepdown=allow_stepdown)
res = optimize(lots, risk, params, cfg, **opt_kw)
plans = {"base": res.baseline, "rec": res.choice}
mc = get_mc(lots, draws, tuple((k, tuple(v)) for k, v in plans.items()), tuple(sorted(params.items())))
mcs = unc.summarize(mc, "rec", "base")
corner = unc.corner_check(lots.reset_index(drop=True), risk["risk_mean"].to_numpy(), res.choice, res.baseline,
                          param_specs(cfg), params, cfg)
v_state, failed = unc.robust_verdict(mcs, corner, cfg)
confident = v_state == unc.ROBUST


@st.cache_data(show_spinner=False)
def get_thresholds(lots, p0, plan_t, ref_t, params_items):
    sp, prm = param_specs(load_assumptions()), dict(params_items)
    be = unc.break_even(lots, p0, list(plan_t), list(ref_t), sp, prm, load_assumptions())
    nar = unc.narrowing_needed(lots, p0, list(plan_t), list(ref_t), sp, prm, load_assumptions())
    return be, nar


def robust_panel():
    """'What would make it robust?' - one-way break-even values and the joint narrowing the corner check needs."""
    be, nar = get_thresholds(lots.reset_index(drop=True), risk["risk_mean"].to_numpy(), tuple(res.choice),
                             tuple(res.baseline), tuple(sorted(params.items())))
    with st.expander("What would make this recommendation robust?", expanded=not confident):
        st.caption("How wrong can each unvalidated effect be before the plan stops beating the current approach? "
                   "One effect is moved at a time across its declared range; the others stay at their defaults.")
        lines = []
        for _, r in be.iterrows():
            nm = friendly(r["key"])
            if r["deaths_wins_everywhere"] and r["cost_wins_everywhere"]:
                lines.append(f"- **{nm}**: the plan is no worse on cost or deaths anywhere in "
                             f"{r['low']:.2f}–{r['high']:.2f}.")
                continue
            for metric, word in (("deaths", "deaths"), ("cost", "cost")):
                if r[f"{metric}_wins_everywhere"]:
                    lines.append(f"- **{nm}**: the plan is no worse on {word} anywhere in "
                                 f"{r['low']:.2f}–{r['high']:.2f}.")
                elif r[f"{metric}_loses_everywhere"]:
                    lines.append(f"- **{nm}**: the plan is worse on {word} across the whole range.")
                elif r[f"{metric}_threshold"] is not None and r[f"{metric}_threshold"] == r[f"{metric}_threshold"]:
                    side = "at least" if r[f"{metric}_plan_wins_when"] == "above" else "at most"
                    lines.append(f"- **{nm}**: the plan wins on {word} only if this effect is **{side} "
                                 f"{r[f'{metric}_threshold']:.2f}** (default {r['default']:.2f}, range "
                                 f"{r['low']:.2f}–{r['high']:.2f}). **Verify this one first.**")
        st.markdown("\n".join(lines))
        if confident:
            st.success("The plan already wins at every corner of the declared ranges.")
        elif nar["shrink"] is None:
            st.warning("Even at the default effect values the plan does not win on both cost and deaths, so no "
                       "narrowing of the effect ranges makes it robust.")
        else:
            rng = "; ".join(f"{friendly(k)} {lo:.2f}–{hi:.2f}" for k, (lo, hi) in nar["ranges"].items())
            st.info(f"**Jointly:** the plan wins at every corner once all effect ranges are narrowed "
                    f"{100 * nar['shrink']:.0f}% toward their defaults, i.e. to: {rng}. A pen-level trial or "
                    "veterinary judgement that supports these ranges would turn this into a robust call. All ranges "
                    "shrink by the same fraction, so this is an upper bound on the narrowing needed; the one-way "
                    "values above show which effect matters most.")

# ---------- tab 0: start here (plain-language overview, computed live) ----------
GLOSSARY = {
    "BRD (bovine respiratory disease)": "Pneumonia complex in feedlot cattle; the most common and costly disease in the "
                                        "first weeks after arrival.",
    "Lot / cohort": "A group of cattle that arrived together and are managed as one pen group. Decisions here are per lot.",
    "45-day morbidity": "Share of the lot treated for BRD at least once in the first 45 days on feed (DOF).",
    "High-risk lot": "Expected 45-day morbidity of 15% or more if the lot gets no metaphylaxis (15% cut-off "
                     "from Rojas et al. 2022, who applied it to observed morbidity).",
    "Metaphylaxis": "Giving the whole lot an antimicrobial at arrival, before cattle look sick. Needs a veterinarian's "
                    "order; this tool never names drugs or doses.",
    "Extra monitoring + low-stress receiving": "Twice-daily pen riding for 14 days to find sick cattle earlier, plus "
                                               "rest, hay and calmer processing at arrival.",
    "Antimicrobial course (regimen)": "One metaphylaxis dose or one treatment course. Fewer courses = better stewardship.",
    "Case fatality": "Share of sick cattle that die.",
    "Current approach (baseline)": "What the yard does today. Editable in the sidebar; every result is compared with it.",
    "Care floor": "Hard rule: high-risk lots, and borderline lots whose range reaches 15%, never get standard care only, and lots that are clearly high risk keep "
                  "metaphylaxis unless a veterinarian overrides.",
    "Verdict (robust / favourable / low confidence)": "Robust: the plan wins in most what-if runs AND at every end of the "
                                                       "unproven effect ranges. Favourable if assumed effects hold: it wins "
                                                       "on average but loses if an effect is at the bad end. Low confidence: "
                                                       "it does not reliably win.",
    "Binding constraint / shortfall": "A resource used to its limit (binding), or needed beyond its limit (shortfall: "
                                      "escalate to management; the tool never quietly rations care).",
    "90% range (of risk)": "Range for a lot's expected (average) risk given its records - wider when records "
                           "are missing. The lot's actual outcome can vary much more than this, because lots "
                           "differ for reasons yards do not record; P(≥15%) includes that extra spread.",
    "Low confidence": "The evidence is not strong enough to recommend a switch with confidence; the tool says so "
                      "instead of guessing.",
    "EST / ASM": "EST = estimate published for other cattle or years. ASM = our assumption. Both can be changed.",
    "AUC": "How well the model ranks risky lots above safe ones (0.5 = coin flip, 1 = perfect).",
    "PPV": "Of lots flagged high risk, the share that really were.",
    "Calibration / concordance": "Whether predicted percentages match observed percentages, not just the ranking.",
    "Synthetic data": "Realistic but made-up records generated by our code, because no real yard data was provided.",
}

with tabs[0]:
    st.subheader("This week in one minute")
    _b, _r = res.totals[CURRENT], res.totals[RECOMMENDED]
    n_change = int(res.plan["changed"].sum())
    st.markdown(f"**{len(lots)} lots ({int(lots['head_count'].sum()):,} head) arrived at "
                f"{lots['yard'].nunique()} yards.** The tool looked at each lot's chance of BRD, what each option "
                f"would do, and whether the crew and hospital pens can cope. It suggests changing the plan for "
                f"**{n_change} lots**. Compared with what the yard does now:")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Sick cattle (BRD cases)", f"{_r['cases']:.0f}", f"{_r['cases'] - _b['cases']:+.0f} vs now",
              delta_color="inverse")
    s2.metric("Deaths from BRD", f"{_r['deaths']:.1f}", f"{_r['deaths'] - _b['deaths']:+.1f} vs now",
              delta_color="inverse")
    s3.metric("Antimicrobial courses", f"{_r['regimens']:,.0f}", f"{_r['regimens'] - _b['regimens']:+,.0f} vs now",
              delta_color="inverse")
    s4.metric("Expected BRD cost", fmt_money(_r["cash_cost"]),
              f"{fmt_money(_r['cash_cost'] - _b['cash_cost'])} vs now", delta_color="inverse")
    if confident:
        st.success(f"**How sure are we? Robust.** Across {mc['draw'].nunique()} what-if runs that vary every uncertain number, the plan "
                   f"was cheaper in {100 * mcs['p_cheaper']:.0f}% and had no more deaths in "
                   f"{100 * mcs['p_fewer_deaths']:.0f}%, and it still wins at every end of the unvalidated "
                   "intervention-effect ranges. Confident enough to discuss with the veterinarian.")
    elif v_state == unc.FAVOURABLE:
        st.info(f"**How sure are we? Favourable if the assumed effects hold.** In {mc['draw'].nunique()} what-if runs "
                f"the plan was cheaper in {100 * mcs['p_cheaper']:.0f}% and had no more deaths in "
                f"{100 * mcs['p_fewer_deaths']:.0f}%. But at one end of the effect ranges it loses: "
                f"{'; '.join(failed)}. These effects are assumptions the data cannot validate, so the veterinarian "
                "should confirm them before switching.")
    else:
        st.warning(f"**How sure are we? Not sure enough.** {'; '.join(failed).capitalize()}. The tool recommends "
                   "keeping the current approach until the flagged assumptions are checked.")
    vet_lots = res.plan[res.plan["vet_flag"] != ""]["lot_id"].tolist()
    todo = []
    if vet_lots:
        todo.append(f"Veterinarian to sign off medication changes for {len(vet_lots)} lots: {', '.join(vet_lots)}.")
    for s in res.shortfalls:
        todo.append("Capacity problem to escalate: " + s)
    low = risk[risk["confidence"] == "insufficient data"]["lot_id"].tolist()
    if low:
        todo.append(f"Missing records make {len(low)} lots uncertain ({', '.join(low)}); filling in transit time or "
                    "number of sources would sharpen the advice.")
    if todo:
        st.markdown("**What needs a person:**\n" + "\n".join(f"- {x}" for x in todo))

    st.markdown("### How the tool thinks: four questions, in order")
    f1, f2, f3, f4 = st.columns(4)
    with f1.container(border=True):
        st.markdown("**1. Risk**  \n*How likely is each lot to get sick?*")
        st.metric("High-risk lots", int((risk["tier"] == "High").sum()))
        st.caption("From arrival records: weight, source, transit, mixing, vaccination, weather. Tab 2.")
    with f2.container(border=True):
        st.markdown("**2. Action**  \n*What should we do with each lot?*")
        st.metric("Lots given metaphylaxis", f"{sum(OPTIONS[c][1] for c in res.choice)} "
                                            f"(now {sum(OPTIONS[c][1] for c in res.baseline)})")
        st.caption("Four options: standard care, metaphylaxis, extra monitoring, or both. Tab 3.")
    with f3.container(border=True):
        st.markdown("**3. Consequences**  \n*What happens to health, drugs and money?*")
        st.metric("Cost per head", fmt_money(_r["cost_per_head"]), f"{fmt_money(_r['cost_per_head'] - _b['cost_per_head'])}",
                  delta_color="inverse")
        st.caption("Cases, deaths, antimicrobial courses and cost, always shown together. Tabs 3-4.")
    with f4.container(border=True):
        st.markdown("**4. Feasibility**  \n*Can the crew and hospital pens handle it?*")
        st.metric("Resources at their limit", int(res.constraints["status"].isin(["binding", "SHORTFALL"]).sum()))
        st.caption("Rider hours and hospital-pen places per yard; shortfalls are flagged, never hidden. Tab 3.")

    t1, t2 = st.columns(2)
    with t1.container(border=True):
        st.markdown("**What you can rely on**")
        st.markdown("- The logic is transparent: every number has a source tag and a range.\n"
                    "- Vulnerable cattle are protected by a hard care floor.\n"
                    "- Uncertainty is shown, and the tool says when it cannot decide.\n"
                    "- Change any input and every result recomputes in seconds.")
    with t2.container(border=True):
        st.markdown("**What is not proven**")
        st.markdown("- The lots and outcomes are **synthetic** (no real yard data was provided).\n"
                    "- How much each option helps is an **assumption**, not a measured result.\n"
                    "- Model accuracy here is measured on our own synthetic data, so it is optimistic.\n"
                    "- **Any medication decision requires veterinary review.**")

    st.markdown("### Where to look next")
    st.dataframe(pd.DataFrame([
        ("Which lots should I worry about?", "2 · Risk priorities"),
        ("What should I do differently, and what does it cost?", "3 · Plan vs current approach"),
        ("What if our numbers are wrong?", "4 · What-if & sensitivity (and the sidebar)"),
        ("What do I give the vet and the boss?", "5 · Decision summary"),
        ("It's day 7 - are lots on track?", "6 · Days 1-14 check-in"),
        ("How good is the model, honestly?", "7 · Model & evidence"),
    ], columns=["Question", "Tab"]), hide_index=True, use_container_width=True)
    with st.expander("Glossary - plain meanings of the terms used"):
        st.dataframe(pd.DataFrame(list(GLOSSARY.items()), columns=["Term", "Meaning"]), hide_index=True,
                     use_container_width=True)

# ---------- tab 2: risk ----------
with tabs[2]:
    st.subheader("Which lots warrant attention, and why?")
    _top = risk.sort_values("risk_mean", ascending=False).iloc[0]
    st.info(f"**In plain words:** {int((risk['tier'] == 'High').sum())} of {len(risk)} lots are expected to have "
            f"15% or more of their cattle treated for BRD in the first 45 days if they get no metaphylaxis. The "
            f"riskiest is {_top['lot_id']} (yard {_top['yard']}, about {100 * _top['risk_mean']:.0f}%), mainly because "
            f"of: {_top['drivers'].lower()}. Wide bars mean we are less sure, usually because records are missing.")
    st.caption("Risk = expected 45-day BRD morbidity **if no arrival metaphylaxis** is given. "
               "High tier ≥15% (Rojas et al. 2022 definition). Bars show 90% range from model refits "
               "and imputation of missing records.")
    k1, k2, k3, k4 = st.columns(4)
    for col, t in zip((k1, k2, k3), ("High", "Moderate", "Low")):
        col.metric(f"{t} risk lots", int((risk["tier"] == t).sum()))
    k4.metric("Low-confidence lots", int((risk["confidence"] != "confident").sum()))
    r = risk.sort_values("risk_mean")
    fig = go.Figure()
    for t, color in TIER_COLORS.items():
        s = r[r["tier"] == t]
        fig.add_trace(go.Scatter(
            x=s["risk_mean"] * 100, y=s["lot_id"] + " (" + s["yard"] + ")", mode="markers", name=t,
            marker=dict(color=color, size=11, symbol=["x" if c == "insufficient data" else
                                                     ("diamond" if c == "borderline" else "circle")
                                                     for c in s["confidence"]]),
            error_x=dict(type="data", symmetric=False, array=(s["risk_hi"] - s["risk_mean"]) * 100,
                         arrayminus=(s["risk_mean"] - s["risk_lo"]) * 100, color=color),
            customdata=np.stack([s["confidence"], s["drivers"]], axis=1),
            hovertemplate="%{y}: %{x:.1f}%<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>"))
    fig.add_vline(x=cfg["decision"]["high_risk_morbidity"] * 100, line_dash="dash",
                  annotation_text="High-risk threshold 15%")
    fig.update_layout(height=max(380, 24 * len(r)), xaxis_title="Expected 45-day morbidity without metaphylaxis (%)",
                      margin=dict(l=10, r=10, t=30, b=10), legend_title="Tier")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("● confident · ◆ borderline (range crosses 15%) · ✕ insufficient data (≥2 of transit, "
               "commingling, shrink missing)")
    show = risk.sort_values("risk_mean", ascending=False)[
        ["lot_id", "yard", "head_count", "arrival_weight_lb", "source_type", "risk_mean", "risk_lo", "risk_hi",
         "p_high", "tier", "confidence", "drivers"]].copy()
    for c in ("risk_mean", "risk_lo", "risk_hi", "p_high"):
        show[c] = (show[c] * 100).round(1)
    st.dataframe(show.rename(columns={"risk_mean": "risk %", "risk_lo": "low %", "risk_hi": "high %",
                                      "p_high": "P(≥15%) %", "drivers": "main drivers"}),
                 hide_index=True, use_container_width=True)

    with st.expander("Individual animals in one lot (secondary view: which head to look at first)"):
        st.caption("A **ranking aid for pen riders**, not a diagnosis. No animal-level records exist, so each head gets "
                   "a synthetic arrival weight around the lot mean (within-lot spread is an assumption), and its risk "
                   "is the lot model at that weight. The animal risks average to the lot's risk, so the lot's expected "
                   "cases do not change. Caveat: the model was fitted to lot averages, so this assumes the weight "
                   "effect within a lot matches the effect between lots. Real use needs per-head records.")
        _order = risk.sort_values("risk_mean", ascending=False)
        _lid = st.selectbox("Lot", _order["lot_id"].tolist(), key="animal_lot",
                            format_func=lambda x: f"{x} (yard {risk.set_index('lot_id').loc[x, 'yard']}, "
                                                  f"{100 * risk.set_index('lot_id').loc[x, 'risk_mean']:.0f}% risk)")
        _i = int(np.flatnonzero(lots["lot_id"].to_numpy() == _lid)[0])
        _av = get_animals(lots.iloc[_i].to_dict(), float(risk.iloc[_i]["risk_mean"]))
        if _av is None:
            st.warning("This lot has no head count or arrival weight, so there is nothing to rank individuals by.")
        else:
            a1, a2, a3 = st.columns(3)
            a1.metric("Head in lot", len(_av))
            a2.metric("Expected BRD cases if untreated", f"{_av['risk'].sum():.1f}")
            a3.metric("Look at first", int(_av["watch_first"].sum()),
                      help="Top share of head by risk (config individual_view.watch_fraction, an assumption).")
            _h = go.Figure(go.Histogram(x=_av["risk"] * 100, nbinsx=25, marker_color="#2F6690"))
            _h.add_vline(x=float(risk.iloc[_i]["risk_mean"]) * 100, line_dash="dash", annotation_text="lot average")
            _h.update_layout(height=260, xaxis_title="Animal risk (%)", yaxis_title="Head",
                             margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(_h, use_container_width=True)
            _w = _av[_av["watch_first"]].copy()
            _w["risk"] = (_w["risk"] * 100).round(1)
            st.dataframe(_w.rename(columns={"animal_id": "animal", "arrival_weight_lb": "arrival weight (lb)",
                                            "risk": "risk %", "watch_first": "look at first"}),
                         hide_index=True, use_container_width=True)
            if risk.iloc[_i]["confidence"] != "confident":
                st.info(f"This lot is labelled *{risk.iloc[_i]['confidence']}*, so its animal ranking inherits "
                        "the lot's uncertainty.")

# ---------- tab 3: plan ----------
with tabs[3]:
    st.subheader("Recommended plan vs the current operating approach")
    _bt, _rt = res.totals[CURRENT], res.totals[RECOMMENDED]
    st.info(f"**In plain words:** the tool changes the plan for {int(res.plan['changed'].sum())} of {len(lots)} lots. "
            f"Expected result vs today: {abs(_rt['cases'] - _bt['cases']):.0f} "
            f"{'more' if _rt['cases'] > _bt['cases'] else 'fewer'} sick cattle, "
            f"{abs(_rt['deaths'] - _bt['deaths']):.1f} {'more' if _rt['deaths'] > _bt['deaths'] else 'fewer'} deaths, "
            f"{abs(_rt['regimens'] - _bt['regimens']):,.0f} "
            f"{'more' if _rt['regimens'] > _bt['regimens'] else 'fewer'} antimicrobial courses and "
            f"{fmt_money(abs(_rt['cash_cost'] - _bt['cash_cost']))} {'more' if _rt['cash_cost'] > _bt['cash_cost'] else 'less'} "
            "spent on BRD. Red banners below are problems a person must deal with.")
    st.caption(f"Current approach (assumed, editable in sidebar): {baseline_desc}. "
               "The recommended plan minimises expected BRD cost + stewardship penalty within capacity, "
               "under a hard minimum-care floor.")
    for s in res.shortfalls:
        st.error("**Capacity shortfall** - " + s)
    stepdowns = res.plan[res.plan["vet_flag"] == "HIGH-RISK STEP-DOWN"]
    if len(stepdowns):
        st.error("**High-risk step-down - veterinarian sign-off required:** " + ", ".join(stepdowns["lot_id"]) +
                 ". These high-risk lots would get enhanced monitoring instead of metaphylaxis; see the lot reasons "
                 "for the case/death/antimicrobial trade-off.")
    stats = (f"cheaper in **{100 * mcs['p_cheaper']:.0f}%** of uncertainty draws, no more deaths in "
             f"**{100 * mcs['p_fewer_deaths']:.0f}%**, fewer antimicrobial regimens in "
             f"**{100 * mcs['p_fewer_regimens']:.0f}%**")
    if confident:
        st.success("**Robust** - recommended plan vs current approach: " + stats + ", and it still wins at every "
                   "end of the unvalidated intervention-effect ranges.")
    elif v_state == unc.FAVOURABLE:
        st.info("**Favourable if assumed effects hold** - recommended plan vs current approach: " + stats + ". At "
                "some ends of the effect ranges the plan loses: " + "; ".join(failed) + ". Confirm the assumed "
                "effects with the veterinarian (the table below shows every corner).")
    else:
        st.warning("**Low confidence** - recommended plan vs current approach: " + "; ".join(failed) + ". The available information cannot support a "
                   "confident switch from the current approach. Review assumptions and flagged lots with the "
                   "veterinarian. (Overall: " + stats + ".)")
    with st.expander("Every corner of the unvalidated intervention-effect ranges (plan minus current approach)"):
        _ct = corner["corners"].rename(columns={"cost_diff": "Cost difference ($)",
                                                "deaths_diff": "Deaths difference"}).round(2)
        st.caption("Rows where the plan costs more or has more deaths than the current approach are highlighted.")
        st.dataframe(_ct.style.apply(lambda r: ["background-color: rgba(220, 60, 60, 0.18)"] * len(r)
                                     if (r["Cost difference ($)"] > 0 or r["Deaths difference"] > 0) else [""] * len(r),
                                     axis=1).format(precision=2),
                     hide_index=True, use_container_width=True)
    robust_panel()
    rows = []
    for name, t in res.totals.items():
        rows.append({"Policy": name, "BRD cases": t["cases"], "BRD deaths": t["deaths"],
                     "Antimicrobial regimens": t["regimens"], "Expected cost": t["cash_cost"],
                     "Cost / head": t["cost_per_head"], "Capacity shortfalls": t["capacity_shortfalls"],
                     "Meets care floor": "yes" if t["meets_care_floor"] else "NO"})
    comp = pd.DataFrame(rows)
    st.dataframe(comp.style.format({"BRD cases": "{:.0f}", "BRD deaths": "{:.1f}",
                                    "Antimicrobial regimens": "{:.0f}", "Expected cost": "${:,.0f}",
                                    "Cost / head": "${:.2f}"}), hide_index=True, use_container_width=True)
    base_t, rec_t = res.totals[CURRENT], res.totals[RECOMMENDED]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Δ BRD cases", f"{rec_t['cases'] - base_t['cases']:+.0f}", delta_color="inverse",
              delta=f"{rec_t['cases'] - base_t['cases']:+.0f}")
    m2.metric("Δ BRD deaths", f"{rec_t['deaths'] - base_t['deaths']:+.1f}", delta_color="inverse",
              delta=f"{rec_t['deaths'] - base_t['deaths']:+.1f}")
    m3.metric("Δ antimicrobial regimens", f"{rec_t['regimens'] - base_t['regimens']:+.0f}", delta_color="inverse",
              delta=f"{100 * (rec_t['regimens'] / base_t['regimens'] - 1):+.0f}%")
    m4.metric("Δ expected cost", fmt_money(rec_t["cash_cost"] - base_t["cash_cost"]), delta_color="inverse",
              delta=f"90% range {fmt_money(mcs['cost_diff_q'][0])} to {fmt_money(mcs['cost_diff_q'][2])}")

    # trade-off chart: each policy relative to current approach
    fig2 = go.Figure()
    metrics = [("cases", "Cases"), ("deaths", "Deaths"), ("regimens", "AMU regimens"), ("cash_cost", "Cost")]
    for name, t in res.totals.items():
        if name == CURRENT:
            continue
        fig2.add_trace(go.Bar(name=name, x=[m[1] for m in metrics],
                              y=[100 * (t[k] / base_t[k] - 1) for k, _ in metrics]))
    fig2.add_hline(y=0, line_color="gray")
    fig2.update_layout(barmode="group", height=340, yaxis_title="% change vs current approach",
                       title="Trade-offs: every option vs the current approach (lower is better)",
                       margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Where the money moves (partial budget)")
    p_pt = risk["risk_mean"].to_numpy()
    cb_base = cost_breakdown(lots, p_pt, res.baseline, params)
    cb_rec = cost_breakdown(lots, p_pt, res.choice, params)
    deltas = {COST_PARTS[k]: cb_rec[k] - cb_base[k] for k in COST_PARTS}
    figw = go.Figure(go.Waterfall(
        orientation="v", measure=["absolute"] + ["relative"] * len(deltas) + ["total"],
        x=["Current approach"] + list(deltas) + ["Recommended plan"],
        base=0,
        y=[sum(cb_base.values())] + list(deltas.values()) + [0],
        text=[fmt_money(sum(cb_base.values()))] + [("+" if v > 0 else "") + fmt_money(v) for v in deltas.values()]
             + [fmt_money(sum(cb_rec.values()))],
        textposition="outside", connector=dict(line=dict(color="#bbb")),
        increasing=dict(marker_color="#c0392b"), decreasing=dict(marker_color="#27ae60"),
        totals=dict(marker_color="#2e86c1")))
    figw.update_layout(height=380, yaxis_title="Expected BRD-related cost this week ($)", showlegend=False,
                       margin=dict(l=10, r=10, t=30, b=10),
                       yaxis=dict(range=[0, max(sum(cb_base.values()), sum(cb_rec.values())) * 1.12]))
    st.plotly_chart(figw, use_container_width=True)
    _big = max(deltas, key=lambda k: abs(deltas[k]))
    st.caption(f"Green = cost goes down, red = cost goes up. The largest change is **{_big.lower()}** "
               f"({fmt_money(deltas[_big])}). Per head: {fmt_money(sum(cb_base.values()) / lots['head_count'].sum())} "
               f"now vs {fmt_money(sum(cb_rec.values()) / lots['head_count'].sum())} under the plan. Prices mix years "
               "(see Model & evidence) - change them in the sidebar.".replace("$", r"\$"))

    st.markdown("#### Lot-by-lot plan")
    plan_show = res.plan[["lot_id", "yard", "head_count", "tier", "confidence", "baseline_option",
                          "recommended_option", "vet_flag", "expected_cases", "expected_deaths", "regimens", "reason"]]

    def _row_style(row):
        flag = str(res.plan.at[row.name, "vet_flag"])
        color = ("#f8d7da" if flag == "HIGH-RISK STEP-DOWN" else
                 ("#fff3cd" if bool(res.plan.at[row.name, "changed"]) else ""))
        return [f"background-color: {color}" if color else ""] * len(row)

    st.dataframe(plan_show.style.format({"expected_cases": "{:.1f}", "expected_deaths": "{:.2f}",
                                         "regimens": "{:.0f}"}).apply(_row_style, axis=1),
                 hide_index=True, use_container_width=True)
    st.caption("Yellow = differs from current approach · red = high-risk step-down needing vet sign-off. "
               "A metaphylaxis dose and a treatment course each count as one regimen.")
    st.markdown("#### Resource feasibility")
    cons = res.constraints.copy()
    st.dataframe(cons.style.format({"used": "{:.1f}", "capacity": "{:.1f}", "utilisation": "{:.0%}",
                                    "value_of_one_more_unit": lambda v: "" if pd.isna(v) else f"${v:,.0f}"})
                 .apply(lambda row: [{"SHORTFALL": "background-color: #f8d7da",
                                      "binding": "background-color: #fff3cd"}.get(str(cons.at[row.name, "status"]), "")]
                        * len(row), axis=1), hide_index=True, use_container_width=True)
    st.caption("'Value of one more unit' = reduction in expected BRD cost + stewardship penalty if one more rider "
               "hour/day or hospital head place were available (shown only when no shortfall remains; otherwise the "
               "note says whether it reduces the shortfall). Rider hours are a hard limit. Solver: " + res.solver)

# ---------- tab 4: what-if ----------
with tabs[4]:
    st.subheader("What changes the answer?")
    st.info("**In plain words:** many numbers in this tool are educated guesses. This tab shows which guesses "
            "matter. A long bar = if that guess is wrong, the answer could change. Bars crossing the black line mean "
            "the recommended plan could lose to the current approach.")
    st.caption("Change any assumption or capacity in the sidebar and every tab recomputes live. "
               "Below: how the plan's advantage over the current approach moves as each uncertain input "
               "goes from its low to high value (others held at current settings).")
    if changed:
        st.info("Inputs changed from defaults: " + ", ".join(friendly(k) for k in changed))
    p0 = risk["risk_mean"].to_numpy()

    def _diff(p, key):
        return plan_totals(lots, p0, res.choice, p)[key] - plan_totals(lots, p0, res.baseline, p)[key]

    def _tornado(key, xlabel):
        base_d = _diff(params, key)
        rows = []
        for k, s in specs.items():
            if k == "head_per_pen":
                continue
            lo_v, hi_v = (_diff(dict(params, **{k: float(v)}), key) for v in (s["low"], s["high"]))
            rows.append({"param": k, "tag": s["tag"], "low": lo_v, "high": hi_v, "swing": abs(hi_v - lo_v)})
        tor = pd.DataFrame(rows).sort_values("swing")
        tor = tor[tor["swing"] > 1e-9].tail(8)
        fig = go.Figure()
        lab = tor["param"].map(friendly) + " [" + tor["tag"] + "]"
        fig.add_trace(go.Bar(y=lab, x=tor["low"] - base_d, base=base_d, orientation="h", name="input LOW",
                             marker_color="#5dade2"))
        fig.add_trace(go.Bar(y=lab, x=tor["high"] - base_d, base=base_d, orientation="h", name="input HIGH",
                             marker_color="#af7ac5"))
        fig.add_vline(x=0, line_color="black")
        fig.update_layout(barmode="overlay", height=380, xaxis_title=xlabel, margin=dict(l=10, r=10, t=30, b=10),
                          legend=dict(orientation="h", y=-0.25))
        flips = [friendly(k) for k in tor[(tor["low"] > 1e-9) | (tor["high"] > 1e-9)]["param"]]
        return fig, flips

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Cost** (recommended − current, $; <0 saves)")
        f_cost, flips_cost = _tornado("cash_cost", "Δ expected BRD cost ($)")
        st.plotly_chart(f_cost, use_container_width=True)
    with t2:
        st.markdown("**Deaths** (recommended − current; <0 fewer)")
        f_d, flips_deaths = _tornado("deaths", "Δ expected BRD deaths")
        st.plotly_chart(f_d, use_container_width=True)
    if flips_cost or flips_deaths:
        msg = []
        if flips_cost:
            msg.append("be **more expensive** if " + ", ".join(flips_cost) + " moves to the edge of its range")
        if flips_deaths:
            msg.append("have **more deaths** if " + ", ".join(flips_deaths) + " moves to the edge of its range")
        st.warning("Compared with the current approach, the recommended plan could " + "; ".join(msg) +
                   " (holding the plan fixed). These are the "
                   "questions to settle with the veterinarian and your records before acting.")
    else:
        st.success("No single input, moved across its plausible range, makes the recommended plan more expensive or "
                   "deadlier than the current approach.")
    st.markdown("#### Re-optimising across metaphylaxis effect sizes")
    st.caption("The most uncertain assumption. For each value, the plan is re-optimised.")
    sweep = []
    for v in np.linspace(specs["metaphylaxis_morbidity_rrr"]["low"], specs["metaphylaxis_morbidity_rrr"]["high"], 5):
        rs = optimize(lots, risk, dict(params, metaphylaxis_morbidity_rrr=float(v)), cfg, **opt_kw)
        t = rs.totals[RECOMMENDED]
        sweep.append({"metaphylaxis prevents (share of cases)": f"{100 * float(v):.0f}%",
                      "lots given metaphylaxis": sum(OPTIONS[c][1] for c in rs.choice),
                      "regimens": round(t["regimens"]), "deaths": round(t["deaths"], 1),
                      "expected cost": round(t["cash_cost"])})
    st.dataframe(pd.DataFrame(sweep), hide_index=True, use_container_width=True)

# ---------- tab 5: summary ----------
with tabs[5]:
    st.subheader("Decision summary")
    c1, c2 = st.columns(2)
    reviewer = c1.text_input("Veterinarian / reviewer name (optional)")
    notes = c2.text_input("Notes (optional)")
    md = build_summary(res, mcs, params, specs, changed, data_label, reviewer, notes, baseline_desc, corner=corner)
    st.download_button("⬇️ Download summary (Markdown)", md, "brd_decision_summary.md", "text/markdown")
    st.download_button("⬇️ Download lot plan (CSV)", res.plan.to_csv(index=False), "brd_lot_plan.csv", "text/csv")
    st.markdown(md)

# ---------- tab 6: days 1-14 check-in ----------
with tabs[6]:
    st.subheader("Days 1-14 check-in: are lots tracking the plan?")
    st.caption("Enter BRD pulls (first treatments) recorded so far for each lot. The tool updates each lot's 45-day "
               "risk from its arrival estimate, projects hospital load for the coming week and suggests monitoring "
               "or capacity actions. Assumes cases start evenly over the first "
               f"{params['case_onset_spread_days']:.0f} days (ASM; real pulls often cluster earlier - change "
               "'case onset spread days' in the sidebar to see the effect). Lot-to-lot spread, including within-pen "
               "spread of a contagious disease, is already in the arrival prior. A sensitivity input in the sidebar "
               "('extra clustering of early pulls', default off) treats pulls to date as noisier; raising it pulls "
               "updated risks back toward the arrival estimate and escalates fewer lots (less cautious). Effect sizes "
               "are held at their current values, so ranges are somewhat too narrow. **Any change to medication requires veterinary review.**")
    model_, _ = get_model()
    cday, cpre = st.columns([2, 3])
    day = cday.slider("Days on feed so far", 1, 14, 7, key="checkin_day")
    prefill = cpre.checkbox("Pre-fill SYNTHETIC pulls for the demo", value=True, key="checkin_prefill",
                            help="Simulated from the risk model - not real records. Untick and type your own counts.")
    sim = (fu.simulate_pulls(lots, draws, model_.resid_sd, res.choice, params, day) if prefill
           else np.full(len(lots), np.nan))
    pull_in = pd.DataFrame({"lot_id": lots["lot_id"], "yard": lots["yard"], "head": lots["head_count"],
                            "plan": [OPTION_SHORT[c] for c in res.choice], "pulls_to_date": sim})
    pull_ed = st.data_editor(pull_in, hide_index=True, use_container_width=True,
                             key=f"pulls_demo_{day}" if prefill else "pulls_manual",
                             disabled=["lot_id", "yard", "head", "plan"],
                             column_config={
                                 "lot_id": st.column_config.TextColumn("lot_id", width="small"),
                                 "yard": st.column_config.TextColumn("yard", width="small"),
                                 "head": st.column_config.NumberColumn("head", width="small"),
                                 "plan": st.column_config.TextColumn("plan", width="large"),
                                 "pulls_to_date": st.column_config.NumberColumn(
                                     "pulls to date", min_value=0, step=1, width="medium",
                                     help="Blank = not recorded")})
    pulls = pd.to_numeric(pull_ed["pulls_to_date"], errors="coerce").to_numpy(float)
    pulls = np.minimum(pulls, lots["head_count"].to_numpy(float))
    upd = fu.update(lots, draws, model_.resid_sd, res.choice, params, day, pulls,
                    thr_high=cfg["decision"]["high_risk_morbidity"])
    rec_mask = ~np.isnan(pulls)
    n_esc = int(upd["status"].isin(fu.ESCALATE).sum())
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Pulls to date", f"{np.nansum(pulls):.0f}",
              delta=f"{np.nansum(pulls) - upd.loc[rec_mask, 'expected_pulls_to_date'].sum():+.0f} vs expected",
              delta_color="inverse")
    q2.metric("Lots to review", n_esc)
    q3.metric("Below expected", int((upd["status"] == "Below expected").sum()))
    q4.metric("Not recorded", int((upd["status"] == "No pulls recorded").sum()))
    if prefill:
        st.info("Pull counts shown are **synthetic** (simulated for the demo), not real records.")
    st.markdown("#### Updated 45-day risk")
    u = upd.sort_values("updated_risk")
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=u["prior_risk"] * 100, y=u["lot_id"], mode="markers", name="at arrival (plan)",
                              marker=dict(color="#95a5a6", size=9)))
    colors = {fu.NEW_HIGH: "#c0392b", fu.BORDERLINE: "#d35400", "Above expected": "#e67e22", "On track": "#2e86c1",
              "Below expected": "#27ae60", "No pulls recorded": "#7f8c8d"}
    fig6.add_trace(go.Scatter(
        x=u["updated_risk"] * 100, y=u["lot_id"], mode="markers", name="updated (colour = status)",
        marker=dict(color=[colors[s] for s in u["status"]], size=11),
        error_x=dict(type="data", symmetric=False, array=(u["updated_hi"] - u["updated_risk"]) * 100,
                     arrayminus=(u["updated_risk"] - u["updated_lo"]) * 100, color="#bbb"),
        customdata=u["status"], hovertemplate="%{y}: %{x:.1f}%<br>%{customdata}<extra></extra>"))
    fig6.add_vline(x=cfg["decision"]["high_risk_morbidity"] * 100, line_dash="dash")
    fig6.update_layout(height=max(380, 24 * len(u)), xaxis_title="Expected 45-day morbidity under the plan (%)",
                       margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.08),
                       yaxis=dict(type="category", dtick=1, categoryorder="array", categoryarray=list(u["lot_id"])))
    st.plotly_chart(fig6, use_container_width=True)
    show6 = upd.copy()
    for c in ("prior_risk", "updated_risk", "updated_lo", "updated_hi", "p_high"):
        show6[c] = (show6[c] * 100).round(1)
    show6 = show6.rename(columns={"prior_risk": "arrival risk %", "updated_risk": "updated risk %",
                                  "updated_lo": "low %", "updated_hi": "high %", "p_high": "P(≥15%) %",
                                  "expected_pulls_to_date": "expected pulls",
                                  "remaining_cases": "projected remaining cases"})
    pct_fmt = {c: "{:.1f}" for c in ("arrival risk %", "updated risk %", "low %", "high %", "P(≥15%) %")}
    st.dataframe(show6.style.format({"expected pulls": "{:.1f}", "projected remaining cases": "{:.1f}",
                                     "pulls_to_date": "{:.0f}", **pct_fmt})
                 .apply(lambda row: [("background-color: #f8d7da" if upd.at[row.name, "status"] in fu.ESCALATE
                                      else "")] * len(row), axis=1),
                 hide_index=True, use_container_width=True)
    st.markdown("#### Hospital pen outlook (next 7 days)")
    st.dataframe(fu.hospital_outlook(upd, params, caps, day).style.format(
        {"projected_new_cases_next_7d": "{:.1f}", "projected_hospital_load": "{:.1f}", "free_hospital_head": "{:.1f}"}),
        hide_index=True, use_container_width=True)
    st.markdown("#### Suggested actions")
    rider_used = res.constraints[res.constraints["resource"].str.contains("rider")].set_index("yard")["used"].to_dict()
    acts = fu.suggested_actions(lots, upd, params, caps, rider_used)
    if acts:
        for a in acts:
            st.markdown("- " + a)
    else:
        st.success("No lot is tracking above its plan. Keep the current protocol.")
    st.caption("Suggestions cover monitoring, staffing and escalation only - never drug choice or dosing.")
    src_tag = "SYNTHETIC demo pulls" if prefill else "entered by user"
    st.download_button("⬇️ Download check-in (CSV)", upd.assign(pulls_source=src_tag).to_csv(index=False),
                       f"brd_checkin_day{day}{'_SYNTHETIC' if prefill else ''}.csv", "text/csv")

# ---------- tab 7: model & evidence ----------
with tabs[7]:
    st.subheader("How good is the risk model - honestly?")
    ev = get_eval()
    st.error("These metrics come from **synthetic** lots generated by our own assumptions. They show that "
             "the pipeline recovers the data-generating process - **not** real-world accuracy. Published benchmark on "
             "real feedlot data (Rojas et al. 2022, 1,733 cohorts): **AUC 0.68-0.79, PPV 11-25%**.")
    a, b, c, d = st.columns(4)
    st.caption(f"Temporal hold-out: trained on {ev['n_train']} lots that arrived before {ev['cutoff']}, tested on the "
               f"{ev['n_test']} lots that arrived after (the model never saw them). The label is OBSERVED 45-day "
               "morbidity ≥15% (after any metaphylaxis), so predictions are scored with each lot's actual treatment.")
    a.metric("AUC (synthetic, temporal)", f"{ev['auc']:.2f}", help="Published real-data range 0.68-0.79")
    b.metric("Brier score", f"{ev['brier']:.3f}")
    c.metric("Sensitivity / specificity", f"{ev['sensitivity']:.0%} / {ev['specificity']:.0%}")
    d.metric("PPV / NPV", f"{ev['ppv']:.0%} / {ev['npv']:.0%}", help=f"at prevalence {ev['prevalence']:.0%}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Calibration slope", f"{ev['cal_slope']:.2f}", help="1.00 = predictions spread exactly as much as "
              "outcomes; below 1 = overconfident.")
    c2.metric("Calibration-in-the-large", f"{100 * ev['cal_in_the_large']:+.1f} pts", help="Observed minus predicted "
              "average morbidity (head-weighted).")
    c3.metric("Lot-to-lot spread (logit SD)", f"{ev['resid_sd']:.2f}", help="Estimated out of fold by maximum "
              "likelihood; the simulator used 1.3. Drives P(≥15%).")
    cal = ev["calibration"]
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=cal["pred"] * 100, y=cal["obs"] * 100, mode="markers+lines", name="deciles"))
    mx = float(max(cal["pred"].max(), cal["obs"].max()) * 100) + 2
    fig4.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(dash="dash", color="gray"), name="perfect"))
    fig4.update_layout(height=340, xaxis_title="Predicted morbidity (%)", yaxis_title="Observed morbidity (%)",
                       title=f"Calibration on {ev['n_test']} held-out synthetic lots", margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("#### Does the model predict *how much* disease, not just *which* lots? (adequacy)")
    adq = ev["adequacy"]
    st.info(f"**In plain words:** comparing predicted with observed 45-day morbidity on {ev['n_test']} held-out lots, "
            f"concordance is {adq['ccc']:.2f}. That sounds low, but a **perfect** model of expected risk would only score "
            f"about {adq['ccc_ceiling']:.2f} on these lots, because lots vary for reasons yards do not record and small "
            f"lots are noisy. {100 * adq['share_random']:.1f}% of the error is random and "
            f"{100 * (adq['share_mean_bias'] + adq['share_slope_bias']):.1f}% systematic. So this check mainly shows "
            "the model is not consistently too high or too low; it cannot show much more. Still synthetic, so still circular.")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Concordance (CCC)", f"{adq['ccc']:.2f}", delta=f"ceiling {adq['ccc_ceiling']:.2f}", delta_color="off",
              help="Lin's concordance correlation coefficient = precision (Pearson r) × accuracy (bias correction "
              "factor Cb). Ceiling = what a perfect expected-risk model would score here (simulated).")
    e2.metric("Precision r / accuracy Cb", f"{adq['pearson_r']:.2f} / {adq['accuracy_cb']:.2f}")
    e3.metric("Root mean square error", f"{100 * adq['rmsep']:.1f} pts", help="Typical miss, in percentage points "
              "of morbidity.")
    e4.metric("Observed vs predicted slope", f"{adq['slope']:.2f}", help="1.00 = no slope bias; intercept "
              f"{100 * adq['intercept']:+.1f} pts.")
    a1, a2 = st.columns([3, 2])
    op = ev["obs_pred"]
    figa = go.Figure()
    figa.add_trace(go.Scatter(x=op["pred"] * 100, y=op["obs"] * 100, mode="markers", name="held-out lot",
                              marker=dict(size=6, color="#2e86c1", opacity=0.55)))
    _mx = float(max(op["pred"].max(), op["obs"].max()) * 100) + 2
    figa.add_trace(go.Scatter(x=[0, _mx], y=[0, _mx], mode="lines", name="perfect (y = x)",
                              line=dict(dash="dash", color="gray")))
    figa.update_layout(height=340, xaxis_title="Predicted morbidity (%)", yaxis_title="Observed morbidity (%)",
                       margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.1))
    a1.plotly_chart(figa, use_container_width=True)
    figm = go.Figure(go.Bar(x=[100 * adq["share_mean_bias"], 100 * adq["share_slope_bias"], 100 * adq["share_random"]],
                            y=["Mean bias", "Slope bias", "Random error"], orientation="h",
                            marker_color=["#c0392b", "#e67e22", "#95a5a6"],
                            text=[f"{100 * adq[k]:.1f}%" for k in ("share_mean_bias", "share_slope_bias", "share_random")],
                            textposition="auto"))
    figm.update_layout(height=340, xaxis_title="Share of mean squared error of prediction (%)",
                       margin=dict(l=10, r=10, t=30, b=10), title="Where the error comes from")
    a2.plotly_chart(figm, use_container_width=True)
    st.caption("Methods: concordance correlation coefficient (Lin 1989) and decomposition of the mean squared error "
               "of prediction into mean bias, slope bias and random error, as recommended for evaluating "
               "animal-science models (Tedeschi 2006).")
    _drift = get_model()[0].drift
    if _drift.get("ran"):
        _msg = (f"**Drift check** (last {_drift['window_days']} days, {_drift['n_window']} lots): observed "
                f"{_drift['observed_cases']:,.0f} BRD cases vs {_drift['expected_cases_out_of_sample']:,.0f} expected "
                f"by a model fitted on earlier lots (z = {_drift['z']:+.2f}; alarm above "
                f"±{cfg['decision'].get('drift_alarm_z', 2.5)}).")
        if _drift["alarm"]:
            st.warning(_msg + f" **Alarm:** risk has shifted, so every prediction was recalibrated by "
                       f"{_drift['offset']:+.2f} on the logit scale. Check what changed (buyers, season, weather).")
        else:
            st.success(_msg + " No recalibration needed.")
    st.markdown("#### Does it make good decisions? Validation against a known truth")
    _vr = ROOT / "reports" / "validation_results.json"
    if _vr.exists():
        import json as _json
        V = _json.loads(_vr.read_text(encoding="utf-8"))
        _stale = V.get("config_hash") != config_hash()
        st.caption(f"Pre-computed by `python -m scripts.run_validation` on {V['generated']} (about 15 minutes; too slow "
                   "to run live). Full write-up: `docs/VALIDATION_REPORT.md`."
                   + (" Warning: the synthetic world has changed since; re-run the script." if _stale else ""))
        rb = V["regret_base"]["headline"]
        sm = {r["policy"]: r for r in V["regret_base"]["summary"]}
        worst = min(V["stress"], key=lambda r: r["realised_saving"])
        wdeath = max(V["stress"], key=lambda r: r["d_deaths"])   # worst on health, shown too (critic R3-1)
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Weekly regret vs oracle", fmt_money(sm["recommended"]["regret"]),
                  delta=f"current approach {fmt_money(sm['current']['regret'])}", delta_color="off",
                  help="Extra expected cost per week compared with a planner that knows each lot's true risk and the "
                       "true effects (lower is better; 0 = perfect).")
        v2.metric("Saving vs current (simulated truth)", fmt_money(rb["realised_saving_vs_current"]) + "/wk",
                  delta=f"tool estimated {fmt_money(rb['estimated_saving_vs_current'])}", delta_color="off",
                  help="Expected saving when each week is scored with the TRUE risk of each lot. In this base study "
                       "the true intervention effects equal the planner's assumptions.")
        v3.metric("Truly-high lots left on standard care", f"{sm['recommended']['floor_violations']:.2f}/wk")
        v4.metric("Most adverse scenario tested", fmt_money(worst["realised_saving"]) + "/wk",
                  delta=f"{worst['d_deaths']:+.2f} deaths/wk", delta_color="inverse",
                  help=worst["label"])
        _stv = {r["scenario"]: r for r in V["stress"]}
        st.caption("**Caveat:** in the base study the simulated truth equals the planner's assumptions, so these "
                   "numbers show the risk model and optimizer work, not that the assumed effects are right. Regret "
                   "when the effects are wrong: " + ", ".join(
                       f"{_stv[k]['label'].split(':')[0]} {fmt_money(_stv[k]['mean_regret'])}/wk"
                       for k in ("S6_no_enh_benefit", "S7_adverse", "S8_combined") if k in _stv)
                   + ". The oracle ignores the care floor and has unlimited hospital space.")
        st.warning(f"**Headline risk:** if the assumptions are wrong in the unfavourable direction ({worst['label']}), "
                   f"the plan saves only {fmt_money(worst['realised_saving'])} per week, loses money in "
                   f"{100 * worst['p_loss_week']:.0f}% of weeks and changes deaths by {worst['d_deaths']:+.2f}/week. "
                   + ("This is also the worst scenario for cattle health. " if wdeath is worst else
                      f"**Worst for cattle health** ({wdeath['label']}): deaths change by {wdeath['d_deaths']:+.2f}/week "
                      f"and the plan costs more in {100 * wdeath['p_loss_week']:.0f}% of weeks. ") +
                   "The benefit of extra riding cannot be validated on these data; confirm effects with the veterinarian.")
        with st.expander("Stress tests, leave-one-yard-out and baselines"):
            stt = pd.DataFrame(V["stress"])[["label", "auc", "cal_slope", "estimated_saving", "realised_saving",
                                              "p_loss_week", "d_deaths", "floor_violations_per_week"]]
            st.dataframe(stt.rename(columns={"label": "world", "auc": "AUC", "cal_slope": "cal. slope",
                                             "estimated_saving": "est. saving $/wk", "realised_saving": "true saving $/wk",
                                             "p_loss_week": "P(loss week)", "d_deaths": "Δ deaths/wk",
                                             "floor_violations_per_week": "truly-high on std care/wk"}).round(3),
                         hide_index=True, use_container_width=True)
            l1, l2 = st.columns(2)
            l1.markdown("**Leave one yard out** (a yard the model never saw)")
            l1.dataframe(pd.DataFrame(V["loyo"]).round(3), hide_index=True, use_container_width=True)
            l2.markdown("**Baselines** (temporal split; lower Brier is better)")
            l2.dataframe(pd.DataFrame(V["baselines"]).T.round(3), use_container_width=True)
            st.markdown("**Positivity** - how many lots in each weight band were left untreated. The untreated-risk "
                        "estimate for light lots rests on few untreated lots, so it leans on the metaphylaxis "
                        "adjustment (confounding by indication).")
            st.dataframe(pd.DataFrame(V["positivity"]).sort_values("band").rename(
                columns={"band": "weight band (lb)", "treated_share": "share given metaphylaxis",
                         "untreated_lots": "untreated lots"}).round(2), hide_index=True, use_container_width=True)
    else:
        st.info("Run `python -m scripts.run_validation` to add the decision-validation results here.")
    st.markdown("#### Every number in the model")
    at = pd.DataFrame([{"parameter": friendly(k), "code name": k, "value": params[k], "low": s["low"], "high": s["high"], "unit": s["unit"],
                        "tag": s["tag"], "source": s["source"]} for k, s in specs.items()])
    st.dataframe(at, hide_index=True, use_container_width=True)
    st.markdown("**Tags:** EST = published estimate from another population/time; ASM = team assumption "
                "(no direct evidence; tested in sensitivity). No values are OBS (observed in real data for these yards). "
                "Full sources: `docs/SOURCES.md`; data definitions: `data/DATA_DICTIONARY.md`.")
