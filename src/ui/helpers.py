"""
SmartForge — UI Helper Functions
==================================
Shared logic used by both the User Dashboard and the Auditor Dashboard.

Exports
-------
    pipeline_timeline(agents_run, retry)   → HTML  pipeline node diagram
    status_stepper(status)                 → HTML  claim lifecycle stepper
    build_stats_html()                     → HTML  auditor stat-card row
    run_pipeline(image_path, …)            → (final_output, partial_state)
    build_checkpoint_list(partial)         → list  for DB storage
    extract_phash(fraud_report)            → str
    chat_with_session(message, history, session_id) → str  Groq chat reply
"""

import json
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.config.settings import cfg
from src.db.mongo_client import db_count, db_get, db_upsert
from src.ui.theme import stat_card


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline timeline HTML
# ─────────────────────────────────────────────────────────────────────────────

_PIPELINE_NODES = [
    ("intake",              "📥"),
    ("fraud",               "🛡️"),
    ("perception",          "👁️"),
    ("gemini_agent",        "🤖"),
    ("false_positive_gate", "🚧"),
    ("health_monitor",      "💊"),
    ("verification_v2",     "🔍"),
    ("reasoning",           "🧠"),
    ("decision",            "⚖️"),
    ("report",              "📄"),
]


def pipeline_timeline(
    agents_run: Optional[List[str]],
    retry: int = 0,
) -> str:
    """
    Render a row of pipeline-node pills, coloured green (done) or grey (idle).

    Parameters
    ----------
    agents_run : list of agent name strings from job_summary["agents_run"]
    retry      : retry_count from job_summary — shown as a warning note if > 0
    """
    ran   = set(agents_run or [])
    parts = []

    for name, icon in _PIPELINE_NODES:
        ok  = name in ran
        bg  = "var(--sf-node-done-bg)"  if ok else "var(--sf-node-idle-bg)"
        brd = "var(--sf-node-done-brd)" if ok else "var(--sf-node-idle-brd)"
        col = "var(--sf-node-done-txt)" if ok else "var(--sf-node-idle-txt)"
        check = (
            "<span style='position:absolute;top:-4px;right:-4px;font-size:10px;'>✓</span>"
            if ok else ""
        )
        label = name.replace("_", "\n")
        parts.append(
            f"<div class='sf-node' style='background:{bg};border-color:{brd};"
            f"position:relative;'>{check}"
            f"<span class='sf-node-icon'>{icon}</span>"
            f"<span class='sf-node-label' style='color:{col};'>{label}</span>"
            f"</div>"
        )

    retry_note = ""
    if retry and int(retry) > 0:
        retry_note = (
            f"<div style='margin-top:6px;font-size:11px;padding:3px 10px;"
            f"border-radius:6px;background:var(--sf-warn-bg);"
            f"border:1px solid var(--sf-warn-brd);color:var(--sf-warn-txt);"
            f"display:inline-block;'>🔄 {retry} HealthMonitor retry(s)</div>"
        )
    return (
        f"<div style='padding:4px 0;display:flex;flex-wrap:wrap;'>"
        + "".join(parts)
        + f"</div>{retry_note}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Status stepper HTML
# ─────────────────────────────────────────────────────────────────────────────

_STATUS_STEPS = [
    ("uploaded",        "📤 Uploaded"),
    ("pref_saved",      "🛡️ Pref Saved"),
    ("analyzed",        "🔬 Analyzed"),
    ("claim_submitted", "📋 Claim Filed"),
    ("fraud_flagged",   "🚨 Fraud Flagged"),
    ("approved",        "✅ Approved"),
    ("rejected",        "❌ Rejected"),
]
_STATUS_ORDER = [s[0] for s in _STATUS_STEPS]


def status_stepper(status: str) -> str:
    """
    Render the claim lifecycle as a horizontal colour-coded stepper bar.

    Parameters
    ----------
    status : str — current status string from the DB (e.g. "analyzed")
    """
    try:
        cur = _STATUS_ORDER.index(status)
    except ValueError:
        cur = 0

    parts = []
    for i, (key, label) in enumerate(_STATUS_STEPS):
        done   = i < cur
        active = i == cur

        if   active and key == "approved":      col, bg = "var(--sf-ok-txt)",   "var(--sf-ok-bg)"
        elif active and key in ("rejected",
                                "fraud_flagged"): col, bg = "var(--sf-err-txt)", "var(--sf-err-bg)"
        elif active:                             col, bg = "var(--sf-brand)",    "var(--sf-brand-light)"
        elif done:                               col, bg = "var(--sf-ok-txt)",   "var(--sf-ok-bg)"
        else:                                    col, bg = "var(--sf-text-hint)","var(--sf-neu-bg)"

        fw  = "700" if (active or done) else "400"
        pfx = "✓ " if done else ""
        parts.append(
            f"<div class='sf-step {('done' if done else '')} {('active' if active else '')}'"
            f" style='background:{bg};border-color:{col};"
            f"color:{col};font-weight:{fw};'>{pfx}{label}</div>"
        )

    return (
        "<div style='display:flex;gap:2px;border-radius:8px;overflow:hidden;"
        "border:1px solid var(--sf-border);margin-bottom:12px;'>"
        + "".join(parts)
        + "</div>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Auditor stat cards
# ─────────────────────────────────────────────────────────────────────────────

def build_stats_html() -> str:
    """Return a flex row of six stat cards pulled live from the DB."""
    c = db_count()
    return (
        "<div style='display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap;'>"
        + stat_card("Total Cases",    c.get("total",    0), "var(--sf-brand)")
        + stat_card("Analyzed",       c.get("analyzed", 0), "var(--sf-info-txt)")
        + stat_card("Fraud Flagged",  c.get("fraud",    0), "var(--sf-err-txt)")
        + stat_card("Approved",       c.get("approved", 0), "var(--sf-ok-txt)")
        + stat_card("Rejected",       c.get("rejected", 0), "var(--sf-err-txt)")
        + stat_card("Pending Review", c.get("pending",  0), "var(--sf-warn-txt)")
        + "</div>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner (used by User Dashboard Tab 3)
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    image_path:  str,
    vehicle_id:  str,
    policy_id:   str,
    claim_date:  str,
    claim_lat:   float,
    claim_lon:   float,
    bypass_fraud: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Execute the full LangGraph pipeline for one image and return the outputs.

    Temporarily patches cfg to match the per-call parameters, runs the graph
    to the HITL interrupt before decision_node, then resumes to completion.

    Parameters
    ----------
    image_path   : str   — path to the primary vehicle image
    vehicle_id   : str   — claimant vehicle ID
    policy_id    : str   — insurance policy number
    claim_date   : str   — ISO date of the accident
    claim_lat    : float — claimed GPS latitude
    claim_lon    : float — claimed GPS longitude
    bypass_fraud : bool  — True → skip all fraud checks

    Returns
    -------
    (final_output, partial_state)
        final_output   : dict  — assembled by report_node (may be empty on fraud halt)
        partial_state  : dict  — last streamed state (contains fraud_report etc.)
    """
    # Lazy import to avoid circular deps at module load time
    from src.graph.workflow import graph
    from src.graph.state import make_initial_state

    # Patch cfg for this call
    _orig_bypass = cfg.BYPASS_FRAUD
    cfg.BYPASS_FRAUD = bypass_fraud

    try:
        state  = make_initial_state(
            image_path = image_path,
            vehicle_id = vehicle_id or cfg.VEHICLE_ID,
            policy_id  = policy_id  or "",
            claim_date = claim_date or "",
            claim_lat  = float(claim_lat or 0.0),
            claim_lon  = float(claim_lon or 0.0),
        )
        thread = {"configurable": {"thread_id": state["job_id"]}}

        # Phase 1 — run to HITL interrupt (before decision_node)
        partial: Dict[str, Any] = {}
        for event in graph.stream(state, thread, stream_mode="values"):
            partial = event

        # Phase 2 — auto-resume and complete report
        final: Dict[str, Any] = {}
        try:
            for event in graph.stream(None, thread, stream_mode="values"):
                final = event
        except Exception:
            pass   # fraud-halted graphs have no phase 2

        fo = (final or partial or {}).get("final_output") or {}
        return fo, partial

    finally:
        cfg.BYPASS_FRAUD = _orig_bypass


# ─────────────────────────────────────────────────────────────────────────────
# DB utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_checkpoint_list(partial: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build a minimal checkpoint record from the partial state for DB storage."""
    return [{
        "step":              -1,
        "node":              "gradio_partial",
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "retry_count":       partial.get("retry_count",       0),
        "health_score":      partial.get("health_score",      "N/A"),
        "validation_passed": partial.get("validation_passed", "N/A"),
        "n_detections":      len(partial.get("raw_detections", [])),
        "n_messages":        len(partial.get("messages",       [])),
    }]


def extract_phash(fraud_report: Dict[str, Any]) -> str:
    """Extract the pHash hex string from a fraud_report dict."""
    return (
        (fraud_report or {})
        .get("details", {})
        .get("phash_check", {})
        .get("phash", "")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Groq chat helper (scoped to one session)
# ─────────────────────────────────────────────────────────────────────────────

def chat_with_session(
    message:    str,
    history:    List,
    session_id: str,
) -> str:
    """
    Answer a user question using Groq, with the current session data injected
    as system context.  Strict rule: only answer about this vehicle/claim.

    Parameters
    ----------
    message    : str  — user's chat message
    history    : list — [(user_msg, bot_reply), …] conversation history
    session_id : str  — DB case_id for context retrieval

    Returns
    -------
    str — Groq reply, or an error message
    """
    if not cfg.GROQ_ENABLED:
        return "⚠️ Groq API key not set. Add GROQ_API_KEY to your .env file."

    rec       = db_get(session_id) if session_id else {}
    fo        = rec.get("final_output") or {}
    insurance = rec.get("insurance")    or {}
    ud        = rec.get("user_data")    or {}
    fraud_rep = rec.get("fraud_report") or {}

    ctx = json.dumps({
        "vehicle_id":    rec.get("vehicle_id",   "unknown"),
        "vehicle_type":  fo.get("vehicle_type",  "unknown"),
        "health_score":  fo.get("overall_assessment_score", "N/A"),
        "ruling":        fo.get("claim_ruling",        "N/A"),
        "ruling_code":   fo.get("claim_ruling_code",   "N/A"),
        "damages":       fo.get("damages", [])[:5],
        "financial":     fo.get("financial_estimate",  {}),
        "fraud_status":  fraud_rep.get("status",       "not_checked"),
        "trust_score":   fraud_rep.get("trust_score",  "N/A"),
        "fraud_attempts": rec.get("fraud_attempts",    0),
        "insurance":     insurance,
        "owner_name":    ud.get("owner_name", "User"),
        "case_status":   rec.get("status",    "unknown"),
    }, default=str)

    system_prompt = (
        f"You are SmartForge AI Assistant for vehicle {rec.get('vehicle_id', 'unknown')}.\n"
        "STRICT RULE: Answer ONLY about this vehicle and claim. "
        "Never reveal other users' data.\n"
        "Use ONLY the data below. If info is missing, say so.\n\n"
        f"--- SESSION DATA ---\n{ctx}\n--- END ---\n\n"
        "Tone: professional, empathetic, concise. No jargon."
    )

    try:
        from groq import Groq
        client = Groq(api_key=cfg.GROQ_API_KEY)
        msgs   = [{"role": "system", "content": system_prompt}]
        for turn in (history or [])[-6:]:
            if isinstance(turn, (list, tuple)) and len(turn) == 2:
                msgs.append({"role": "user",      "content": str(turn[0])})
                msgs.append({"role": "assistant",  "content": str(turn[1])})
        msgs.append({"role": "user", "content": message})

        reply = client.chat.completions.create(
            model       = cfg.GROQ_MODEL,
            messages    = msgs,
            max_tokens  = 400,
            temperature = 0.3,
        ).choices[0].message.content.strip()

        # Persist chat history to DB
        new_hist = (history or []) + [[message, reply]]
        if session_id:
            db_upsert(session_id, chat_history=new_hist)

        return reply

    except Exception as exc:
        return f"⚠️ Chat error: {exc}"
