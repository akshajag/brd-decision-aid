"""Build the downloadable decision summary (Markdown)."""
from __future__ import annotations

from datetime import date

import pandas as pd

from src.decision.optimize import PlanResult
from src.decision.uncertainty import FAVOURABLE, ROBUST, robust_verdict, thresholds


def _money(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def build_summary(res: PlanResult, mc_summary: dict, params: dict, specs: dict,
                  changed_params: list[str], data_label: str, reviewer: str = "",
                  notes: str = "", baseline_desc: str = "", corner: dict | None = None) -> str:
    base = res.totals["Current approach"]
    rec = res.totals["Recommended plan"]
    d = {k: rec[k] - base[k] for k in ("cases", "deaths", "regimens", "cash_cost")}
    conf = mc_summary
    state, failed = robust_verdict(conf, corner)
    lines = [
        f"# BRD arrival decision summary - {date.today().isoformat()}",
        "",
        f"**Data:** {data_label}  ",
        f"**Lots:** {len(res.plan)} ({int(res.plan['head_count'].sum()):,} head) · **Window:** first 45 days on feed  ",
        f"**Current approach (assumed):** {baseline_desc or 'weight-rule metaphylaxis'}  ",
        f"**Solver:** {res.solver}",
        "",
        "> **Medication-related recommendations require veterinary review before use.** "
        "Intervention effects are assumptions, not validated results for this operation.",
        "",
        "## Recommendation vs current operating approach",
        "| Measure | Current approach | Recommended | Change |",
        "|---|---:|---:|---:|",
        f"| Expected BRD cases | {base['cases']:.0f} | {rec['cases']:.0f} | {d['cases']:+.0f} |",
        f"| Expected BRD deaths | {base['deaths']:.1f} | {rec['deaths']:.1f} | {d['deaths']:+.1f} |",
        f"| Antimicrobial regimens | {base['regimens']:.0f} | {rec['regimens']:.0f} | {d['regimens']:+.0f} |",
        f"| Expected BRD-related cost | {_money(base['cash_cost'])} | {_money(rec['cash_cost'])} | {_money(d['cash_cost'])} |",
        f"| Capacity shortfalls | {base['capacity_shortfalls']} | {rec['capacity_shortfalls']} | |",
        "",
        "**Under uncertainty** (Monte Carlo over assumption ranges and risk-model uncertainty): "
        f"recommended plan is cheaper in {100 * conf['p_cheaper']:.0f}% of draws, has no more deaths in "
        f"{100 * conf['p_fewer_deaths']:.0f}%, and uses fewer antimicrobial regimens in "
        f"{100 * conf['p_fewer_regimens']:.0f}%. Cost change 90% range: "
        f"{_money(conf['cost_diff_q'][0])} to {_money(conf['cost_diff_q'][2])}.",
        "",
        ("**Confidence: ROBUST.** The recommendation held up across the assumption ranges tested "
         f"(cheaper in at least {100 * thresholds()[0]:.0f}% and no more deaths in at least {100 * thresholds()[1]:.0f}% of draws) "
         "and still wins at every end of the unvalidated intervention-effect ranges."
         if state == ROBUST else
         ("**Confidence: FAVOURABLE IF ASSUMED EFFECTS HOLD.** The Monte Carlo criteria pass "
          f"(cheaper in at least {100 * thresholds()[0]:.0f}%, no more deaths in at least {100 * thresholds()[1]:.0f}% of draws), "
          "but at an end of the effect ranges the plan loses (" + "; ".join(failed) + "). The veterinarian should "
          "confirm the assumed intervention effects before switching.")
         if state == FAVOURABLE else
         "**Confidence: LOW** (" + "; ".join(failed) + "). The available information cannot support a "
         "confident switch from the current approach. Review the flagged assumptions and lots with the veterinarian."),
        "",
    ]
    flagged = res.plan[res.plan["vet_flag"] != ""]
    if len(flagged):
        lines += ["## Changes that need veterinary sign-off"]
        lines += [f"- **{r.lot_id}** [{r.vet_flag}]: {r.reason}" for r in flagged.itertuples()]
        lines.append("")
    if res.shortfalls:
        lines += ["## Capacity shortfalls (escalate)"] + [f"- {s}" for s in res.shortfalls] + [""]
    bind = res.constraints[res.constraints["status"] != "ok"]
    if len(bind):
        lines += ["## Binding constraints", "| Yard | Resource | Used | Capacity | Status |", "|---|---|---:|---:|---|"]
        lines += [f"| {r.yard} | {r.resource} | {r.used:.1f} | {r.capacity:.1f} | {r.status} |" for r in bind.itertuples()]
        lines.append("")
    lines += ["## Lot-level plan", "| Lot | Yard | Head | Risk (90% range) | Confidence | Current | Recommended |",
              "|---|---|---:|---|---|---|---|"]
    for r in res.plan.itertuples():
        mark = " **(changed)**" if r.changed else ""
        lines.append(f"| {r.lot_id} | {r.yard} | {r.head_count} | {100 * r.risk_mean:.0f}% "
                     f"({100 * r.risk_lo:.0f}-{100 * r.risk_hi:.0f}%) | {r.confidence} | "
                     f"{r.baseline_option} | {r.recommended_option}{mark} |")
    low = res.plan[res.plan["confidence"] != "confident"]
    if len(low):
        lines += ["", "## Lots needing attention before acting"]
        lines += [f"- **{r.lot_id}** ({r.confidence}): {r.reason}" for r in low.itertuples()]
    lines += ["", "## Key assumptions used"]
    for k in ["metaphylaxis_morbidity_rrr", "enhanced_monitoring_cfr_rrr", "receiving_protocol_morbidity_rrr",
              "case_fatality_rate", "production_loss_per_case", "feeder_price_per_lb", "amu_stewardship_weight"]:
        s = specs[k]
        flag = " *(changed by user)*" if k in changed_params else ""
        lines.append(f"- {k} = {params[k]:g} {s['unit']} [{s['tag']}; range {s['low']}-{s['high']}]{flag}")
    lines += ["", "## Veterinary review", f"- Reviewer: {reviewer or '________________'}",
              "- Decision: approve / modify / reject (circle)", f"- Notes: {notes or ''}", ""]
    return "\n".join(lines)
