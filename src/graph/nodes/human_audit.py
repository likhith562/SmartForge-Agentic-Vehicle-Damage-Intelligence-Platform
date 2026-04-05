"""
SmartForge — human_audit_node
==============================
Terminal node for claims that fail the fraud gate.

When fraud_router routes to "human_audit", the LangGraph pipeline halts here
and no GPU or paid API resources are consumed for this claim.  The node:

1. Prints a structured fraud alert to the console / Colab cell output.
2. Assembles a ``final_output`` dict so the Gradio dashboards can display
   the fraud result without receiving a None final_output.
3. Saves a detailed JSON report to disk for the human underwriter.
4. Sets ``is_fraud=True`` in state so the DB layer and Auditor Dashboard
   correctly classify the case.

After this node the graph reaches END — no further nodes are executed.

3-Strike tolerance (enforced by the Gradio UI layer, not here)
--------------------------------------------------------------
The UI layer reads ``fraud_attempts`` from the DB and blocks re-submission
once ``MAX_FRAUD_RETRIES`` is reached.  This node always increments the
counter via its state return so the DB layer has the latest value.

State mutations returned
------------------------
    final_output   dict  — HUMAN_AUDIT_REQUIRED payload
    is_fraud       bool  — True
    messages       list  — one entry appended
"""

import json
import os
from datetime import datetime, timezone

from src.graph.state import SmartForgeState, log_msg


# Path where the fraud audit JSON report is written
_FRAUD_REPORT_PATH = "/content/fraud_audit_report.json"


def human_audit_node(state: SmartForgeState) -> dict:
    """
    LangGraph node: terminal handler for fraud-flagged claims.

    Halts the pipeline, saves a fraud report to disk, and returns a
    ``final_output`` payload that the dashboards can display.

    Parameters
    ----------
    state : SmartForgeState
        Expects ``fraud_report`` to be populated by fraud_node.

    Returns
    -------
    dict — partial state update; graph reaches END after this node.
    """
    fr = state.get("fraud_report") or {}

    # ── Console output ────────────────────────────────────────────────────────
    print("\n" + "═" * 64)
    print("  🚨  FRAUD ALERT — CLAIM ROUTED TO HUMAN AUDIT")
    print("═" * 64)
    print(f"  Trust Score  : {fr.get('trust_score', 'N/A')}/100")
    print(f"  Status       : {fr.get('status', 'N/A')}")
    print(f"  Checks Run   : {fr.get('checks_run', 'N/A')}")
    print(f"  Image        : {state['image_path']}")
    print(f"  Checked At   : {(fr.get('checked_at', '') or '')[:19]}")

    flags = fr.get("flags", [])
    if flags:
        print("\n  🔴 Active Fraud Flags:")
        for fl in flags:
            print(f"     • {fl}")

    print("\n  ➡️  Claim halted. Zero GPU/API resources consumed for this claim.")
    print("     Human underwriter review required before re-submission.")

    # ── Assemble final output ─────────────────────────────────────────────────
    final_out = {
        "job_id":       state["job_id"],
        "vehicle_id":   state.get("vehicle_id", ""),
        "claim_id":     f"CLM-FRAUD-{state['job_id']}",
        "status":       "HUMAN_AUDIT_REQUIRED",
        "claim_ruling_code":  "CLM_MANUAL",
        "processing_status":  "manual_review_required",
        "auto_approved":      False,
        "fraud_report":       fr,
        "overall_assessment_score": 0,
        "confirmed_damage_count":   0,
        "executive_summary": (
            "This claim has been flagged by the automated fraud detection system "
            "and requires human underwriter review before it can be processed."
        ),
        "message": (
            "Claim flagged by Batch 1 Fraud Layer (5-check). "
            "Human review required."
        ),
        "ruling_timestamp": datetime.now(timezone.utc).isoformat() + "Z",
    }

    # ── Save report to disk ───────────────────────────────────────────────────
    os.makedirs(os.path.dirname(_FRAUD_REPORT_PATH), exist_ok=True)
    try:
        with open(_FRAUD_REPORT_PATH, "w") as fh:
            json.dump(final_out, fh, indent=2, default=str)
        print(f"  💾 Detailed fraud report saved → {_FRAUD_REPORT_PATH}")
    except Exception as exc:
        print(f"  ⚠️  Could not save fraud report: {exc}")

    print("═" * 64)

    return {
        "final_output": final_out,
        "is_fraud":     True,
        "messages": [
            log_msg(
                "human_audit_node",
                f"Claim halted — fraud suspicion. "
                f"trust_score={fr.get('trust_score', 'N/A')} "
                f"flags={len(flags)}. "
                f"Report saved → {_FRAUD_REPORT_PATH}",
            )
        ],
    }
