"""
SmartForge — perception_node + perception_retry_node
======================================================
Runs the full CV perception stack: SAHI → SAM → MiDaS → Part Detection.

On first pass  (retry_count == 0): uses adaptive_sahi_conf from intake_node.
On retry       (retry_count  > 0): lowers confidence by 20 % per attempt to
                                    catch detections missed on the first pass.

Circuit breaker
---------------
If MAX_RETRIES is reached and health_monitor still fails, the router sends
the graph to reasoning with a "CircuitBreaker" stability flag rather than
looping forever.  perception_retry_node increments retry_count so the router
can read it without state mutation conflicts.

State mutations returned (perception_node)
------------------------------------------
    raw_detections     list  — fully attributed detection dicts
    depth_map          Any   — MiDaS numpy array
    image_bgr          None  — freed after use (saves ~28 MB RAM)
    pipeline_trace     dict  — "perception_agent" entry added
    messages           list  — one entry appended
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import torch

from src.config.settings import cfg
from src.cv.depth import compute_deformation_index, run_midas_depth
from src.cv.perception import (
    get_damage_location_unified,
    run_part_detection,
    run_sam_segmentation,
    run_sahi_detection,
)
from src.graph.state import SmartForgeState, log_msg


def perception_node(state: SmartForgeState) -> dict:
    """
    LangGraph node: full CV perception stack.

    Reads from state
    ----------------
    image_bgr, image_rgb, image_path, adaptive_sahi_conf,
    retry_count, validation_errors, vehicle_type, gemini_agent_ran

    Returns partial state update
    ----------------------------
    raw_detections, depth_map, image_bgr (set to None), pipeline_trace, messages
    """
    retry     = state["retry_count"]
    prev_errs = state["validation_errors"]

    if retry > 0:
        print(f"\n🔄 [perception] RE-ANALYSIS (retry {retry}/{cfg.MAX_RETRIES})")
        print(f"   Errors from HealthMonitor: {prev_errs}")
    else:
        print("\n🔵 [perception] starting CV pipeline…")

    image     = state["image_bgr"]
    image_rgb = state["image_rgb"]
    path      = state["image_path"]
    device    = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Adaptive SAHI confidence ──────────────────────────────────────────────
    # Base confidence set by intake_node from HSV analysis.
    # Each retry lowers it by 20 % to surface missed detections.
    base_conf = state.get("adaptive_sahi_conf", cfg.SAHI_CONFIDENCE)
    conf      = base_conf * (0.8 ** retry)

    # ── SAHI detection ────────────────────────────────────────────────────────
    object_preds = run_sahi_detection(path, conf=conf, device=device)
    scene        = state.get("scene_type", "unknown")
    print(
        f"   SAHI: {len(object_preds)} detections "
        f"(conf≥{conf:.2f} | base={base_conf} | scene={scene} | retry={retry})"
    )

    # ── MiDaS depth map ───────────────────────────────────────────────────────
    depth_map = run_midas_depth(image_rgb)
    global_variance = float(np.var(depth_map)) + 1e-6

    # ── Part detection (YOLO part model + COCO vehicle detector) ─────────────
    part_boxes, vd = run_part_detection(image_rgb)

    # If Gemini already classified the vehicle type on a prior retry, skip COCO
    gemini_vt = state.get("vehicle_type", "unknown")
    if gemini_vt != "unknown" and state.get("gemini_agent_ran", False):
        vd = None
        print(f"   [perception] vehicle_type from Gemini state: {gemini_vt} — skipping COCO")

    # ── Build raw_detections ──────────────────────────────────────────────────
    raw_detections: List[Dict[str, Any]] = []
    low_conf_count = 0

    for i, obj in enumerate(object_preds):
        b  = obj.bbox
        x1, y1, x2, y2 = int(b.minx), int(b.miny), int(b.maxx), int(b.maxy)
        confidence = float(obj.score.value)

        # SAM mask for precise boundary
        mask       = run_sam_segmentation(image_rgb, [x1, y1, x2, y2])
        area_ratio = float(np.sum(mask)) / (image.shape[0] * image.shape[1])

        # MiDaS deformation index
        rv = compute_deformation_index(depth_map, mask)

        # Part / zone location
        loc, loc_type = get_damage_location_unified(
            image_rgb,
            [x1, y1, x2, y2],
            part_boxes,
            vd,
            gemini_vehicle_type=gemini_vt,
        )

        low_conf = confidence < cfg.CONFIDENCE_RECHECK_LIMIT
        if low_conf:
            low_conf_count += 1

        raw_detections.append({
            "detection_id":              f"D{i + 1:03d}",
            "type":                      obj.category.name,
            "location":                  loc,
            "location_type":             loc_type,
            "bounding_box":              [x1, y1, x2, y2],
            "confidence":                round(confidence, 3),
            "low_confidence_flag":       low_conf,
            "verification_status":       "pending",
            "relative_deformation_index": round(rv, 6),
            "area_ratio":                round(area_ratio, 6),
            "source":                    "cv_model",
        })

    # ── Audit trace ───────────────────────────────────────────────────────────
    trace_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reasoning": (
            f"SAHI({len(raw_detections)} detections, conf≥{conf:.2f}). "
            f"SAM refined masks. MiDaS depth computed. "
            f"{low_conf_count} low-confidence flag(s). Retry={retry}."
        ),
        "decision": (
            "Raw detections → gemini_agent → false_positive_gate → health_monitor. "
            + (f"Re-analysis triggered by: {prev_errs}" if retry > 0 else "First pass.")
        ),
        "details": {
            "detections":     len(raw_detections),
            "low_conf":       low_conf_count,
            "conf_threshold": round(conf, 3),
            "base_conf":      round(base_conf, 3),
            "scene_type":     scene,
            "retry":          retry,
        },
    }

    print(f"✅ perception: {len(raw_detections)} detections, {low_conf_count} low-conf")

    return {
        # Free image_bgr — SAM and MiDaS consumed image_rgb; no downstream
        # node needs the BGR array and it can be ~28 MB for 4K images.
        "image_bgr":      None,
        "raw_detections": raw_detections,
        "depth_map":      depth_map,
        "pipeline_trace": {
            **state["pipeline_trace"],
            "perception_agent": trace_entry,
        },
        "messages": [
            log_msg(
                "perception_agent",
                f"{len(raw_detections)} detections. retry={retry}.",
            )
        ],
    }


def perception_retry_node(state: SmartForgeState) -> dict:
    """
    Thin wrapper that increments retry_count then delegates to perception_node.

    Keeping the increment in a separate node avoids a state-mutation race
    that would occur if perception_node incremented and returned its own
    retry_count at the same time as writing raw_detections.

    The health_monitor_router reads the NEW retry_count on the next pass
    to decide whether the circuit breaker should fire.
    """
    new_retry = state["retry_count"] + 1
    print(f"\n⚡ [circuit_check] retry_count → {new_retry}/{cfg.MAX_RETRIES}")

    # Run perception with the bumped counter injected into a temporary state copy
    perception_updates = perception_node({**state, "retry_count": new_retry})

    # Merge the incremented counter into the perception output
    return {**perception_updates, "retry_count": new_retry}
