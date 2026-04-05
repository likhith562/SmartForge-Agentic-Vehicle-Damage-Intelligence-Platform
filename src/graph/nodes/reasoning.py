"""
SmartForge — reasoning_node
=============================
Batch 4: Financial Intelligence Engine.

Converts verified damage detections into a professional line-item repair
estimate using the Mitchell/Audatex-style REPAIR_DATABASE matrix, then
computes the overall vehicle health score and a Total Loss determination.

Two score views
---------------
confirmed_score    — computed from NON-rejected detections only.
                     This is the official score used by decision_node.
conservative_score — computed from ALL detections including rejected ones.
                     Surfaced in the audit trace for transparency; never
                     used for the claim ruling.

Financial line-item logic (Batch 4)
------------------------------------
For each confirmed damage:
    1. Look up part data in REPAIR_DATABASE (fuzzy name matching).
    2. Determine action from SEVERITY_TO_ACTION:
           Minor / Moderate → REPAIR/PAINT  (paint + 2 h labour)
           Severe / Critical → REPLACE       (replace cost + 4 h labour)
    3. Compute cost_usd = action cost + (labour_per_hour × hours).
    4. Convert to INR for display using USD_TO_INR.

Total Loss check
----------------
    If total_repair_cost_usd > VEHICLE_VALUE × TOTAL_LOSS_THRESHOLD
    → total_loss_flag = True → disposition = "TOTALED"

Severity preference order (Gemini > CV)
-----------------------------------------
reasoning_node prefers severity_gemini (set by verification_v2 Golden Frame)
over the CV-derived severity (set by compute_severity in perception).  This
means the financial estimate reflects Gemini's forensic assessment, not just
the raw depth-variance heuristic.

State mutations returned
------------------------
    damages_output      list  — full detection records with severity + cost
    financial_estimate  dict  — line_items, totals, disposition
    total_loss_flag     bool
    pipeline_trace      dict  — "reasoning_agent" entry added
    messages            list  — one entry appended
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from src.config.settings import cfg
from src.cv.perception import compute_severity, estimate_cost, severity_to_score
from src.graph.state import SmartForgeState, log_msg


# ─────────────────────────────────────────────────────────────────────────────
# REPAIR_DATABASE fuzzy lookup
# ─────────────────────────────────────────────────────────────────────────────

def _get_repair_data(location: str) -> Dict[str, float]:
    """
    Look up repair/replace cost data from REPAIR_DATABASE.

    Priority
    --------
    1. Exact match on location string.
    2. Fuzzy: DB key is a substring of location, or vice-versa (case-insensitive).
    3. Fall back to "_default" entry.
    """
    db = cfg.REPAIR_DATABASE

    # Exact match
    if location in db:
        return db[location]

    # Fuzzy match
    loc_lower = location.lower()
    for key, val in db.items():
        if key == "_default":
            continue
        if key.lower() in loc_lower or loc_lower in key.lower():
            return val

    return db["_default"]


# ─────────────────────────────────────────────────────────────────────────────
# reasoning_node
# ─────────────────────────────────────────────────────────────────────────────

def reasoning_node(state: SmartForgeState) -> dict:
    """
    LangGraph node: severity classification + financial estimation.

    Reads from state
    ----------------
    verified_damages (Batch 3 preferred) or raw_detections (fallback)

    Returns partial state update
    ----------------------------
    damages_output, financial_estimate, total_loss_flag,
    pipeline_trace, messages
    """
    # Prefer verified_damages (Batch 3 Golden Frame output)
    detections = state.get("verified_damages") or state.get("raw_detections", [])
    print(f"\n🔵 [reasoning] processing {len(detections)} detection(s)…")

    # Skip damages explicitly rejected by verification_v2
    confirmed_only = [d for d in detections if d.get("is_verified") is not False]
    rejected_v2    = [d for d in detections if d.get("is_verified") is False]
    if rejected_v2:
        print(
            f"   ↳ Skipping {len(rejected_v2)} damage(s) rejected by "
            "Golden Frame verification"
        )
    detections = confirmed_only

    # ── Per-detection severity + legacy cost (INR range) ─────────────────────
    damages_output:    List[Dict[str, Any]] = []
    severity_all:      List[str]            = []   # conservative (all detections)
    severity_confirmed: List[str]           = []   # official (non-rejected)
    cost_low  = cost_hi  = 0                       # all detections aggregate
    conf_low  = conf_hi  = 0                       # confirmed-only aggregate
    reasoning_lines: List[str] = []

    for det in detections:
        t        = det["type"]
        rv       = det["relative_deformation_index"]
        ar       = det["area_ratio"]
        rejected = det.get("rejected", False)

        sev, cat          = compute_severity(t, rv, ar)
        cost_range, repair_type = estimate_cost(t, sev)

        # Accumulate aggregate cost ranges (INR, quick estimate)
        try:
            nums = [
                int(x.replace("₹", "").replace(",", "").strip())
                for x in cost_range.split("–")
            ]
            cost_low += nums[0];  cost_hi += nums[-1]
            if not rejected:
                conf_low += nums[0];  conf_hi += nums[-1]
        except Exception:
            pass

        severity_all.append(sev)
        if not rejected:
            severity_confirmed.append(sev)

        reasoning_lines.append(
            f"{det['detection_id']}: {t}@{det['location']}→{sev}({cat})"
            f"{'[REJECTED]' if rejected else ''}, rv={rv:.4f}"
        )

        record = {k: v for k, v in det.items()}
        record.update({
            "severity":              sev,
            "damage_category":       cat,
            "repair_type":           repair_type,
            "estimated_repair_cost": cost_range,
        })
        damages_output.append(record)

    # ── Official and conservative health scores ───────────────────────────────
    confirmed_score    = max(0, 100 - sum(severity_to_score(s) for s in severity_confirmed))
    conservative_score = max(0, 100 - sum(severity_to_score(s) for s in severity_all))
    score              = confirmed_score   # official score for downstream nodes

    rec             = "Repair Required" if score < 80 else "Minor Damage"
    confirmed_total = (
        f"₹{conf_low:,}–₹{conf_hi:,}" if severity_confirmed else "₹0"
    )
    total_all = f"₹{cost_low:,}–₹{cost_hi:,}"

    # ── Batch 4: Financial line-item estimate (REPAIR_DATABASE matrix) ─────────
    line_items:      List[Dict[str, Any]] = []
    grand_total_usd: float                = 0.0

    for det in damages_output:
        if det.get("rejected", False):
            continue   # only confirmed damages enter the financial estimate

        part     = det.get("location", "Unknown")
        # Prefer Gemini-refined severity from Golden Frame verification
        sev_raw  = det.get("severity_gemini", det.get("severity", "Moderate"))
        action   = cfg.SEVERITY_TO_ACTION.get(sev_raw, "REPAIR/PAINT")
        rep_data = _get_repair_data(part)

        if action == "REPLACE":
            cost_usd = rep_data["replace"] + (rep_data["labor_per_hour"] * 4)
        else:
            cost_usd = rep_data["paint"]   + (rep_data["labor_per_hour"] * 2)

        cost_inr = int(cost_usd * cfg.USD_TO_INR)
        grand_total_usd += cost_usd

        line_items.append({
            "part":             part,
            "action":           action,
            "severity":         sev_raw,
            "cost_usd":         round(cost_usd, 2),
            "cost_inr":         cost_inr,
            "cost_inr_fmt":     f"₹{cost_inr:,}",
            "gemini_reasoning": det.get("gemini_reasoning", "CV + severity rule"),
        })

    grand_total_inr = int(grand_total_usd * cfg.USD_TO_INR)
    total_loss      = grand_total_usd > cfg.VEHICLE_VALUE * cfg.TOTAL_LOSS_THRESHOLD
    disposition     = "TOTALED" if total_loss else "REPAIRABLE"

    financial_estimate = {
        "line_items":           line_items,
        "total_repair_usd":     round(grand_total_usd, 2),
        "total_repair_inr":     grand_total_inr,
        "total_repair_inr_fmt": f"₹{grand_total_inr:,}",
        "vehicle_value_usd":    cfg.VEHICLE_VALUE,
        "total_loss_threshold": f"{cfg.TOTAL_LOSS_THRESHOLD * 100:.0f}%",
        "total_loss_flag":      total_loss,
        "disposition":          disposition,
        "currency_note":        f"USD costs; INR display at ×{int(cfg.USD_TO_INR)}",
    }

    if total_loss:
        print(
            f"   🚨 TOTAL LOSS: repair ${grand_total_usd:,.0f} > "
            f"{cfg.TOTAL_LOSS_THRESHOLD * 100:.0f}% of vehicle value "
            f"${cfg.VEHICLE_VALUE:,} → TOTALED"
        )
    else:
        print(
            f"   💰 Financial estimate: ${grand_total_usd:,.0f} USD "
            f"(₹{grand_total_inr:,}) — {disposition}"
        )

    trace_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reasoning": (
            "Batch 4 Financial Intelligence: REPAIR_DATABASE matrix applied. "
            "Repair vs Replace logic per severity. Total Loss check. "
            f"Total Loss: ${grand_total_usd:,.0f} vs "
            f"${cfg.VEHICLE_VALUE:,} × {cfg.TOTAL_LOSS_THRESHOLD}. "
            + " | ".join(reasoning_lines)
        ),
        "decision": (
            f"Confirmed score={score}/100 | "
            f"Conservative score={conservative_score}/100. "
            f"Rec={rec}. "
            f"Financial: ${grand_total_usd:,.0f} USD / ₹{grand_total_inr:,} INR. "
            f"Disposition: {disposition}. "
            f"ConfirmedCost={confirmed_total}."
        ),
        "details": {
            "overall_score":      score,
            "conservative_score": conservative_score,
            "recommendation":     rec,
            "confirmed_total":    confirmed_total,
            "total_cost_all":     total_all,
            "financial_estimate": financial_estimate,
        },
    }

    print(
        f"✅ reasoning: confirmed_score={score}/100 | "
        f"conservative={conservative_score}/100 | "
        f"cost={confirmed_total} | "
        f"financial=${grand_total_usd:.0f}USD | {disposition}"
    )

    return {
        "damages_output":    damages_output,
        "financial_estimate": financial_estimate,
        "total_loss_flag":   total_loss,
        "pipeline_trace": {
            **state["pipeline_trace"],
            "reasoning_agent": trace_entry,
        },
        "messages": [
            log_msg(
                "reasoning_agent",
                f"score={score}/100 cost={confirmed_total} "
                f"financial=${grand_total_usd:.0f}USD "
                f"disposition={disposition} rec={rec}",
            )
        ],
    }
