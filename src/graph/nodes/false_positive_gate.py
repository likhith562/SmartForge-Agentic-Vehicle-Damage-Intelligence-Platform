"""
SmartForge — false_positive_gate_node
=======================================
Positioned between gemini_agent_node and health_monitor_node.
Solves the domain-shift problem: the damage model was trained on cars and
fires false positives on motorcycles, auto-rickshaws, and background objects.

Four-layer rejection stack
--------------------------
Gate 0  Gemini-discovery bypass
    Detections added by Gemini's Task 4 full-image scan are treated separately:
    they have no YOLO confidence and bypass YOLO-specific gates 1 + 2.
    They are still subject to a minimum Gemini confidence floor (0.65) and
    the depth-flatness gate (Gate 3).

Gate 1  Vehicle-type confidence floor  (non-car detections only)
    YOLO was trained on cars.  On a motorcycle or 3-wheeler, the same damage
    class requires higher YOLO confidence to be trustworthy.
    Threshold: NON_CAR_CONF_THRESHOLD (0.60).

Gate 2  Minimum area  (non-car detections only)
    Sub-pixel or near-invisible detections on non-car vehicles are noise.
    Threshold: MIN_AREA_NON_CAR (0.003 of image area).

Gate 3  Depth flatness  (all detections)
    A detection with near-zero MiDaS deformation AND small area is almost
    certainly a normal vehicle surface — paint glare, a seam, or a shadow.
    2D surface damage types are exempt: Scratch, Paint chip, Flaking, Corrosion
    inherently show zero MiDaS deformation because they don't physically
    displace the panel.

Gate 4  Gemini explicit veto
    gemini_verified == False → always rejected, no override possible.

Gemini positive override (Senior Adjuster Rule)
-----------------------------------------------
If Gemini explicitly confirmed a detection (gemini_verified == True) but
Gate 3 would reject it because MiDaS shows a flat surface, the Gate 3
rejection is cleared.  This handles the critical case where a real scratch
is 2D and MiDaS naturally shows zero deformation.
Gates 1 and 2 are NOT overridable — they are hard signal limits, not
depth-model limitations.

Rejected detections are NOT deleted.  They are marked:
    rejected          = True
    rejection_reason  = "<gate label>: <explanation>"
    verification_status = "unconfirmed"

The decision_node sees unconfirmed detections and issues CLM_MANUAL.
The full audit trail is preserved in raw_detections for the Auditor Dashboard.

State mutations returned
------------------------
    raw_detections   list  — detections with rejected/rejection_reason populated
    pipeline_trace   dict  — "false_positive_gate" entry added
    messages         list  — one entry appended
"""

from datetime import datetime, timezone
from typing import Any, Dict, List

from src.config.settings import cfg
from src.graph.state import SmartForgeState, log_msg

# 2D surface types that are inherently depth-flat — exempt from Gate 3
_FLAT_EXEMPT = {"Scratch", "Paint chip", "Flaking", "Corrosion"}


def false_positive_gate_node(state: SmartForgeState) -> dict:
    """
    LangGraph node: 4-layer false-positive filter.

    Reads from state
    ----------------
    raw_detections, vehicle_type

    Returns partial state update
    ----------------------------
    raw_detections (with rejected flags set), pipeline_trace, messages
    """
    vehicle_type = state.get("vehicle_type", "unknown")
    detections   = state["raw_detections"]
    is_non_car   = vehicle_type not in ("car", "unknown")

    print(
        f"\n🛡️  [false_positive_gate] vehicle={vehicle_type} "
        f"is_non_car={is_non_car} | {len(detections)} detections"
    )

    filtered:       List[Dict[str, Any]] = []
    n_rejected                           = 0
    rejection_log:  List[str]            = []

    for det in detections:
        det    = dict(det)
        conf   = det.get("confidence",                  0.0)
        ar     = det.get("area_ratio",                  0.0)
        rv     = det.get("relative_deformation_index",  0.0)
        gv     = det.get("gemini_verified",             None)
        dtype  = det.get("type",                        "")
        source = det.get("source",                      "cv_model")
        reason = None

        # ── Gate 0: Gemini-discovered detections bypass YOLO gates ────────────
        if source == "gemini_discovery":
            if conf < 0.65:
                reason = (
                    f"GEMINI_DISCOVERY_LOW_CONF: Gemini confidence {conf:.3f} < 0.65 "
                    "— not confident enough to add as missed detection"
                )
            elif (
                rv < cfg.MIN_DEFORMATION_GATE
                and ar < cfg.FLAT_AREA_GATE
                and dtype not in _FLAT_EXEMPT
            ):
                reason = (
                    f"FLAT_SURFACE: gemini_discovery deformation={rv:.6f} AND "
                    f"area={ar:.5f} — likely background noise"
                )
            elif gv is False:
                reason = "GEMINI_VETO: Gemini itself rejected this detection"

            _apply_verdict(det, reason, filtered, rejection_log, source="GeminiDiscovery")
            if reason:
                n_rejected += 1
            continue   # skip YOLO gates below

        # ── Gate 1: vehicle-type confidence floor ─────────────────────────────
        if is_non_car and conf < cfg.NON_CAR_CONF_THRESHOLD:
            reason = (
                f"VEHICLE_TYPE_THRESHOLD: {vehicle_type} needs "
                f"conf≥{cfg.NON_CAR_CONF_THRESHOLD}, got {conf:.3f}"
            )

        # ── Gate 2: minimum area (non-car) ────────────────────────────────────
        elif is_non_car and ar < cfg.MIN_AREA_NON_CAR:
            reason = (
                f"AREA_TOO_SMALL: area={ar:.5f} < {cfg.MIN_AREA_NON_CAR} "
                f"on {vehicle_type}"
            )

        # ── Gate 3: depth flatness ────────────────────────────────────────────
        elif (
            rv < cfg.MIN_DEFORMATION_GATE
            and ar < cfg.FLAT_AREA_GATE
            and dtype not in _FLAT_EXEMPT
        ):
            reason = (
                f"FLAT_SURFACE: deformation={rv:.6f} AND "
                f"area={ar:.5f} — no depth variation, likely normal surface"
            )

        # ── Gate 4: Gemini explicit veto ──────────────────────────────────────
        elif gv is False:
            reason = "GEMINI_VETO: Gemini explicitly rejected this detection"

        # ── Senior Adjuster Rule: Gemini positive overrides Gate 3 ───────────
        if reason and reason.startswith("FLAT_SURFACE") and gv is True:
            print(
                f"   🔄 GEMINI OVERRIDE: {det.get('detection_id')} ({dtype}) — "
                "FLAT_SURFACE cleared by Gemini positive confirmation"
            )
            reason = None   # clear Gate 3 rejection

        _apply_verdict(det, reason, filtered, rejection_log)
        if reason:
            n_rejected += 1

    n_kept = len(filtered) - n_rejected
    print(f"\n   Gate result: {n_kept} kept / {n_rejected} rejected")

    trace_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reasoning": (
            f"4-gate filter. vehicle={vehicle_type}. "
            f"Gates: conf<{cfg.NON_CAR_CONF_THRESHOLD} (non-car), "
            f"area<{cfg.MIN_AREA_NON_CAR} (non-car), "
            f"deformation<{cfg.MIN_DEFORMATION_GATE}+area<{cfg.FLAT_AREA_GATE} "
            f"(2D types {_FLAT_EXEMPT} exempt), "
            "gemini_verified=False → reject. "
            "Gemini positive override: FLAT_SURFACE cleared if gemini_verified=True."
        ),
        "decision": (
            f"{n_kept} passed, {n_rejected} rejected. "
            + (f"Rejected: {rejection_log}" if rejection_log else "No rejections.")
        ),
        "details": {
            "vehicle_type":  vehicle_type,
            "is_non_car":    is_non_car,
            "kept":          n_kept,
            "rejected":      n_rejected,
            "rejection_log": rejection_log,
        },
    }

    return {
        "raw_detections": filtered,
        "pipeline_trace": {
            **state["pipeline_trace"],
            "false_positive_gate": trace_entry,
        },
        "messages": [
            log_msg(
                "false_positive_gate",
                f"kept={n_kept} rejected={n_rejected} vehicle={vehicle_type}",
            )
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _apply_verdict(
    det: dict,
    reason: str | None,
    filtered: list,
    rejection_log: list,
    source: str = "YOLO",
) -> None:
    """Mutate *det* in place with the gate verdict and append to *filtered*."""
    det_id = det.get("detection_id", "?")
    dtype  = det.get("type", "")
    conf   = det.get("confidence", 0.0)
    ar     = det.get("area_ratio", 0.0)
    rv     = det.get("relative_deformation_index", 0.0)

    if reason:
        det.update({
            "rejected":           True,
            "rejection_reason":   reason,
            "verification_status": "unconfirmed",
        })
        rejection_log.append(f"{det_id}: {reason}")
        print(f"   🚫 REJECTED {det_id} ({dtype}) [{source}] | {reason}")
    else:
        det.update({"rejected": False, "rejection_reason": None})
        print(
            f"   ✅ KEPT     {det_id} ({dtype}) [{source}] "
            f"conf={conf:.3f} ar={ar:.5f} rv={rv:.6f}"
        )
    filtered.append(det)
