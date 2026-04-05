"""
SmartForge — health_monitor_node + health_monitor_router
==========================================================
Validates the perception pipeline output before it proceeds to verification
and reasoning.  The conditional edge (health_monitor_router) is the routing
brain that enables the self-correcting retry loop.

Two validation checks
---------------------
Check 1  Tool-grounded bounds
    Every detection's area_ratio must be in (0, 1).
    Every relative_deformation_index must be in [0, 10].
    Violations indicate a SAHI slice mis-alignment or SAM mask overflow —
    re-running perception at a lower confidence can recover.

Check 2  Epistemic consistency (CV detections only)
    Confidence variance across YOLO detections must be ≤ 0.08.
    High variance means the model is uncertain across the scene — often a sign
    of poor image quality or an unusual vehicle angle where re-analysing at a
    lower threshold helps.
    Gemini-discovered detections are excluded from this variance computation
    because they use Gemini's confidence scale (0.65–1.0) which is different
    from YOLO's (0.3+) and would inflate variance artificially.

Removed check: MAJORITY_LOW_CONF
    Previously a third check triggered a retry when >50 % of detections were
    low-confidence.  This created a death spiral: lowering SAHI confidence
    on retry produced even more low-confidence boxes.  Low-confidence
    detections are now correctly handled by Gemini batch verification (Call B
    in gemini_agent_node) rather than perception retry.

Health score
------------
    n_checks  = 2  (Check 1 + Check 2)
    n_failed  = number of distinct checks that raised errors
    health_score = max(0.0, 1.0 - n_failed / n_checks)

pipeline_stability_flag
-----------------------
    "Stable"        — all checks passed
    "Unstable"      — at least one check failed, retry will run
    "CircuitBreaker" — max retries exhausted, degrading to reasoning

Conditional routing (health_monitor_router)
-------------------------------------------
    PASS                           → "reasoning"  (via verification_v2)
    FAIL + retries remaining       → "perception_retry"
    FAIL + retries exhausted       → "reasoning"  (circuit breaker)

State mutations returned
------------------------
    raw_detections            list  — verification_status updated per detection
    health_score              float
    validation_passed         bool
    validation_errors         list[str]
    pipeline_stability_flag   str
    pipeline_trace            dict  — "health_monitor" entry added
    messages                  list  — one entry appended
"""

from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np

from src.config.settings import cfg
from src.graph.state import SmartForgeState, log_msg


def health_monitor_node(state: SmartForgeState) -> dict:
    """
    LangGraph node: validate PerceptionAgent output.

    Reads from state
    ----------------
    raw_detections, retry_count

    Returns partial state update
    ----------------------------
    raw_detections (with verification_status updated),
    health_score, validation_passed, validation_errors,
    pipeline_stability_flag, pipeline_trace, messages
    """
    detections = state["raw_detections"]
    retry      = state["retry_count"]
    errors:    List[str] = []

    print(f"\n🔬 [health_monitor] validating {len(detections)} detections…")

    # ── Check 1: Tool-grounded bounds ─────────────────────────────────────────
    for d in detections:
        ar = d.get("area_ratio", 0)
        if not (0 < ar < 1):
            errors.append(
                f"INVALID_AREA_RATIO: {d['detection_id']} area={ar}"
            )
        rv = d.get("relative_deformation_index", 0)
        if rv < 0 or rv > 10:
            errors.append(
                f"INVALID_DEFORMATION: {d['detection_id']} rv={rv}"
            )

    # ── Check 2: Epistemic confidence variance (CV detections only) ───────────
    cv_dets = [
        d for d in detections
        if d.get("source", "cv_model") == "cv_model"
    ]
    if len(cv_dets) > 1:
        confs    = [d["confidence"] for d in cv_dets]
        variance = float(np.var(confs))
        if variance > 0.08:
            errors.append(
                f"HIGH_CONF_VARIANCE: {variance:.4f} — "
                "outputs epistemically uncertain, re-analyse"
            )

    # ── Health score ──────────────────────────────────────────────────────────
    n_checks     = 2
    n_failed     = min(len(errors), n_checks)
    health_score = max(0.0, 1.0 - (n_failed / n_checks))
    passed       = len(errors) == 0

    # ── Update verification_status per detection ──────────────────────────────
    updated: List[Dict[str, Any]] = []
    for d in detections:
        d = dict(d)
        # Gemini veto is final — never overwrite gemini_verified=False
        if d.get("gemini_verified") is False:
            d["verification_status"] = "unconfirmed"
        elif d["low_confidence_flag"]:
            d["verification_status"] = "unconfirmed" if not passed else "confirmed"
        else:
            d["verification_status"] = "confirmed"
        updated.append(d)

    # ── Pipeline stability flag ───────────────────────────────────────────────
    cb_will_fire = (not passed) and (retry >= cfg.MAX_RETRIES)
    if cb_will_fire:
        stab_flag = "CircuitBreaker"
        print(
            f"   ⚠️  pipeline_stability_flag → CircuitBreaker "
            f"(retries={retry}/{cfg.MAX_RETRIES})"
        )
    elif not passed:
        stab_flag = "Unstable"
    else:
        stab_flag = "Stable"

    status_str = "PASS" if passed else f"FAIL ({len(errors)} error(s))"
    print(f"   Health score: {health_score:.2f} | Status: {status_str}")
    if errors:
        for e in errors:
            print(f"   ❌ {e}")

    trace_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reasoning": (
            "Ran 2 validation checks: "
            "(1) area_ratio + deformation bounds, "
            "(2) epistemic confidence variance (CV detections only). "
            "MAJORITY_LOW_CONF check removed — low-conf detections are "
            "handled by Gemini batch verification, not retry. "
            f"Found {len(errors)} issue(s)."
        ),
        "decision": (
            f"health_score={health_score:.2f}. "
            + (
                "Routing to verification_v2."
                if passed or cb_will_fire
                else "Routing back to perception for re-analysis."
            )
        ),
        "details": {
            "health_score":  health_score,
            "errors":        errors,
            "retry":         retry,
            "stability":     stab_flag,
        },
    }

    return {
        "raw_detections":          updated,
        "health_score":            health_score,
        "validation_passed":       passed,
        "validation_errors":       errors,
        "pipeline_stability_flag": stab_flag,
        "pipeline_trace": {
            **state["pipeline_trace"],
            "health_monitor": trace_entry,
        },
        "messages": [
            log_msg(
                "health_monitor",
                f"health={health_score:.2f} passed={passed} "
                f"stability={stab_flag} errors={errors}",
            )
        ],
    }


def health_monitor_router(state: SmartForgeState) -> str:
    """
    Conditional edge function — the routing brain of the graph.

    Returns the name of the next node:
        "reasoning"        — validation passed OR circuit breaker fired
        "perception_retry" — validation failed, retries remaining
    """
    if state["validation_passed"]:
        print("   ✅ HealthMonitor → routing to verification_v2")
        return "reasoning"   # workflow.py maps "reasoning" → verification_v2

    retry = state["retry_count"]
    if retry < cfg.MAX_RETRIES:
        print(
            f"   🔄 HealthMonitor → re-routing to perception "
            f"(retry {retry + 1}/{cfg.MAX_RETRIES})"
        )
        return "perception_retry"

    # Circuit breaker: max retries exhausted — degrade gracefully
    print(
        f"   ⚡ CIRCUIT BREAKER — max retries ({cfg.MAX_RETRIES}) exhausted "
        "→ degrading to verification_v2"
    )
    return "reasoning"
