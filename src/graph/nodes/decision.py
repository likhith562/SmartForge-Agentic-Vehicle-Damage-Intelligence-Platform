"""
SmartForge — decision_node
===========================
Issues the insurance claim ruling.  This is the Human-in-the-Loop (HITL)
interrupt point: ``workflow.py`` compiles the graph with
``interrupt_before=["decision"]`` so the graph pauses here for high-value
claims, allowing a human reviewer to inspect the state before the final
verdict is committed.

AI-never-auto-approves policy
------------------------------
Even for high-scoring, clean, fully-verified claims the AI issues
CLM_PENDING — never CLM_APPROVED.  Final approval ALWAYS requires the
Auditor Dashboard (human underwriter).  This design choice:
  - Removes the liability of the AI making financial commitments.
  - Ensures every claim has an auditor signature in the DB.
  - Makes the ``auto_approved`` field always False in final_output.

Ruling codes
------------
CLM_MANUAL    — fraud flags present OR unconfirmed detections remain.
                Requires immediate forensic review.
CLM_WORKSHOP  — High-severity damage OR health score < ESCALATION_THRESHOLD.
                Workshop inspection required before settlement.
CLM_PENDING   — Clean, fully-verified claim awaiting auditor sign-off.
                (Previously CLM_APPROVED in the notebook — renamed for safety.)

Score computation
-----------------
Mirrors reasoning_node: computed from CONFIRMED (non-rejected) detections
only.  Previously the notebook computed from all damages (including rejected
ones) creating an inconsistency between the two nodes.  Both now agree on the
same confirmed-only score.

State mutations returned
------------------------
    final_output     dict  — complete claim entity with ruling metadata
    pipeline_trace   dict  — "decision_agent" entry added
    messages         list  — one entry appended
"""

from datetime import datetime, timezone
from typing import Any, Dict, List

from src.config.settings import cfg
from src.cv.perception import severity_to_score
from src.graph.state import SmartForgeState, log_msg


def decision_node(state: SmartForgeState) -> dict:
    """
    LangGraph node: insurance claim ruling.

    Reads from state
    ----------------
    damages_output, financial_estimate, is_fraud, job_id,
    vehicle_id, vehicle_type, vehicle_make_estimate, gemini_agent_ran

    Returns partial state update
    ----------------------------
    final_output, pipeline_trace, messages
    """
    damages  = state["damages_output"]
    job_id   = state["job_id"]
    print("\n🔵 [decision] issuing claims ruling…")

    # ── Compute score from CONFIRMED damages only ─────────────────────────────
    confirmed_damages = [d for d in damages if not d.get("rejected", False)]
    severity_list     = [d["severity"] for d in confirmed_damages]
    score             = max(0, 100 - sum(severity_to_score(s) for s in severity_list))

    has_high   = any(d["severity"] == "High"       for d in confirmed_damages)
    has_unconf = any(
        d.get("verification_status") == "unconfirmed"
        for d in confirmed_damages
    )

    # ── Confirmed aggregate cost (non-rejected detections) ───────────────────
    _low, _hi = 0, 0
    for _d in damages:
        if not _d.get("rejected", False):
            try:
                nums = [
                    int(x.replace("₹", "").replace(",", "").strip())
                    for x in _d.get("estimated_repair_cost", "₹0–₹0").split("–")
                ]
                _low += nums[0];  _hi += nums[-1]
            except Exception:
                pass
    total_cost = (
        f"₹{_low:,}–₹{_hi:,}"
        if (_low or _hi)
        else (damages[0].get("estimated_repair_cost", "₹0") if damages else "₹0")
    )

    # ── Ruling logic ──────────────────────────────────────────────────────────
    is_fraud_flagged = state.get("is_fraud", False)

    if is_fraud_flagged:
        status, code, ruling = (
            "manual_review_required",
            "CLM_MANUAL",
            "🚨 HIGH RISK: Fraud flags detected. Immediate manual forensic audit required.",
        )
    elif has_unconf:
        status, code, ruling = (
            "manual_review_required",
            "CLM_MANUAL",
            "Unconfirmed detections present — human inspector required before settlement.",
        )
    elif has_high or score < cfg.ESCALATION_THRESHOLD:
        status, code, ruling = (
            "pending_workshop_inspection",
            "CLM_WORKSHOP",
            f"Score {score}/100 — workshop inspection required before settlement.",
        )
    else:
        # Clean, fully-verified claim — AI assessment complete, awaiting auditor.
        status, code, ruling = (
            "claim_submitted",
            "CLM_PENDING",
            f"AI assessment complete (score {score}/100). "
            "Awaiting auditor verification for final approval.",
        )

    # ── Assemble final output entity ──────────────────────────────────────────
    fin_est = state.get("financial_estimate") or {}

    entity: Dict[str, Any] = {
        # Identifiers
        "claim_id":   f"CLM-{job_id}",
        "job_id":     job_id,
        "vehicle_id": state.get("vehicle_id", ""),
        "policy_id":  state["policy_id"],

        # Vehicle context (from Gemini agent)
        "vehicle_type":          state.get("vehicle_type",           "unknown"),
        "vehicle_make_estimate": state.get("vehicle_make_estimate",  "unknown"),
        "gemini_agent_ran":      state.get("gemini_agent_ran",       False),

        # Damage summary
        "damage_detected":             len(damages) > 0,
        "damages":                     damages,
        "overall_assessment_score":    score,
        "confirmed_damage_count":      len(confirmed_damages),
        "total_estimated_repair_cost": total_cost,
        "inspection_recommendation":   "Repair Required" if score < 80 else "Minor Damage",

        # Financial (Batch 4)
        "financial_estimate": fin_est,
        "total_loss_flag":    state.get("total_loss_flag", False),

        # Ruling
        "processing_status":   status,
        "auto_approved":       False,   # AI never auto-approves — auditor must sign off
        "claim_ruling":        ruling,
        "claim_ruling_code":   code,
        "settlement_estimate": total_cost,
        "ruling_timestamp":    datetime.now(timezone.utc).isoformat() + "Z",

        # Pipeline provenance
        "pipeline_stability_flag": state.get("pipeline_stability_flag", "Stable"),
        "pipeline_trace":          state["pipeline_trace"],
    }

    trace_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reasoning": (
            f"score={score}, has_high={has_high}, has_unconf={has_unconf}, "
            f"is_fraud={is_fraud_flagged}, "
            f"auto_threshold={cfg.AUTO_APPROVE_THRESHOLD}, "
            f"esc_threshold={cfg.ESCALATION_THRESHOLD}."
        ),
        "decision": f"[{code}] {ruling}",
        "details":  entity,
    }

    icon = "🟢" if code == "CLM_PENDING" else ("🟡" if code == "CLM_WORKSHOP" else "🔴")
    print(f"   {icon} [{code}] {status} | score={score} | cost={total_cost}")

    return {
        "final_output": entity,
        "pipeline_trace": {
            **state["pipeline_trace"],
            "decision_agent": trace_entry,
        },
        "messages": [
            log_msg(
                "decision_agent",
                f"[{code}] {status} score={score}",
            )
        ],
    }
