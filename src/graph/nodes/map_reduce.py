"""
SmartForge — Batch 2: Multi-Image Map-Reduce Nodes
====================================================
Three LangGraph nodes that together implement parallel multi-image processing
using the LangGraph Send API for fan-out and NetworkX graph DB for fusion.

map_images_node
---------------
Fan-out node.  Uses langgraph.types.Send to create N independent cv_worker
sub-states in parallel — one per image in image_paths.  Each Send carries the
minimum state a worker needs plus an empty all_raw_detections accumulator.
The Annotated[List, operator.add] reducer on all_raw_detections auto-merges
results from all workers when they complete.

cv_worker_node
--------------
Per-image parallel unit.  Mirrors intake_node + perception_node but:
  - Lightweight: runs SAHI damage detection only (no SAM, no MiDaS, no Gemini).
    SAM, MiDaS, and Gemini run AFTER fusion on the best (Golden Record) frame.
  - Stamps every detection with source_image_index for fusion tracing.
  - Returns into all_raw_detections (Annotated reducer).
  - Catches all exceptions and returns an empty list so one bad image does not
    abort the whole fan-out.

fusion_node
-----------
Reduce node.  Called after all cv_workers complete.
  1. Calls cv.fusion.fuse_detections() to build the NetworkX DiGraph and
     produce one Golden Record per unique car part.
  2. Appends any image-recycling fraud flags to fraud_report.
  3. Writes fused_detections and raw_detections (downstream compat alias).
  4. Routes the fused list to gemini_agent → false_positive_gate → … as normal.

Graph wiring (from workflow.py)
--------------------------------
    map_images → [Send × N] → cv_worker → fusion → gemini_agent → …

State mutations (map_images_node)
----------------------------------
    Returns list[Send]  — not a state dict (LangGraph Send API)

State mutations (cv_worker_node)
----------------------------------
    all_raw_detections   list  — appended to global accumulator
    messages             list  — one entry appended

State mutations (fusion_node)
----------------------------------
    fused_detections     list
    raw_detections       list  — alias for downstream compat
    fraud_report         dict  — recycling flags appended if any
    pipeline_trace       dict  — "fusion_agent" entry added
    messages             list  — one entry appended
"""

from datetime import datetime, timezone
from typing import Any, Dict, List

import torch
from langgraph.types import Send

from src.config.settings import cfg
from src.cv.fusion import fuse_detections
from src.cv.perception import analyse_image_conditions, run_sahi_detection
from src.graph.state import SmartForgeState, log_msg


# ─────────────────────────────────────────────────────────────────────────────
# map_images_node
# ─────────────────────────────────────────────────────────────────────────────

def map_images_node(state: SmartForgeState) -> List[Send]:
    """
    LangGraph fan-out node: send one cv_worker per image using the Send API.

    Returns a list of Send objects — not a state dict.  LangGraph schedules
    each Send as an independent parallel sub-graph execution.
    """
    paths = state.get("image_paths", [state["image_path"]])
    print(f"\n🗺️  [map_images] Fanning out {len(paths)} parallel CV workers…")
    for i, p in enumerate(paths):
        print(f"   Worker {i}: {p}")

    return [
        Send(
            "cv_worker",
            {
                # Per-worker context
                "image_path":         p,
                "image_paths":        paths,
                "source_image_index": i,
                "adaptive_sahi_conf": state.get("adaptive_sahi_conf", cfg.SAHI_CONFIDENCE),
                "scene_type":         state.get("scene_type", "unknown"),
                "job_id":             state["job_id"],
                "vehicle_id":         state["vehicle_id"],
                "policy_id":          state["policy_id"],
                "messages":           state["messages"],
                "pipeline_trace":     state["pipeline_trace"],

                # Annotated accumulator — starts empty per worker
                "all_raw_detections": [],

                # Required SmartForgeState fields — not used per-worker
                "image_bgr":              None,
                "image_rgb":              None,
                "raw_detections":         [],
                "depth_map":              None,
                "damages_output":         [],
                "final_output":           None,
                "vehicle_type":           "unknown",
                "vehicle_type_confidence": 0.0,
                "vehicle_make_estimate":  "unknown",
                "gemini_agent_ran":       False,
                "gemini_discovered_count": 0,
                "health_score":           1.0,
                "validation_passed":      False,
                "validation_errors":      [],
                "retry_count":            0,
                "fraud_report":           state.get("fraud_report"),
                "is_fraud":               False,
                "fraud_attempts":         0,
                "claim_date":             state.get("claim_date", ""),
                "claim_lat":              state.get("claim_lat",  0.0),
                "claim_lon":              state.get("claim_lon",  0.0),
                "fused_detections":       [],
                "verified_damages":       [],
                "golden_crops":           [],
                "pipeline_stability_flag": "Stable",
                "total_loss_flag":         False,
                "financial_estimate":      None,
                "started_at":              state["started_at"],
            }
        )
        for i, p in enumerate(paths)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# cv_worker_node
# ─────────────────────────────────────────────────────────────────────────────

def cv_worker_node(state: SmartForgeState) -> dict:
    """
    LangGraph parallel worker: lightweight SAHI perception on one image.

    Skips SAM, MiDaS, and Gemini — those run after fusion on the best frame.
    Returns stamped detections into all_raw_detections (Annotated reducer).
    """
    import os
    import cv2
    import numpy as np

    img_path = state["image_path"]
    idx      = state.get("source_image_index", 0)
    print(f"\n⚡ [cv_worker-{idx}] processing → {img_path}")

    stamped: List[Dict[str, Any]] = []

    try:
        if not os.path.exists(img_path):
            raise RuntimeError(f"cv_worker-{idx}: file not found → {img_path}")

        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"cv_worker-{idx}: cannot decode → {img_path}")

        h, w = image.shape[:2]

        # Adaptive confidence for this image's conditions
        conditions = analyse_image_conditions(image, sahi_slice_size=cfg.SAHI_SLICE_SIZE)
        sahi_conf  = conditions["adaptive_sahi_conf"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        preds  = run_sahi_detection(img_path, conf=sahi_conf, device=device)

        for obj_pred in preds:
            bbox  = obj_pred.bbox
            x1    = int(bbox.minx);  y1 = int(bbox.miny)
            x2    = int(bbox.maxx);  y2 = int(bbox.maxy)
            area_ratio = max(0.0, ((x2 - x1) * (y2 - y1)) / (w * h))

            stamped.append({
                "detection_id":       f"W{idx}-{len(stamped):03d}",
                "source_image_index": idx,
                "source_image_path":  img_path,
                "label":              obj_pred.category.name,
                "type":               obj_pred.category.name,
                "confidence":         float(obj_pred.score.value),
                "bbox":               [x1, y1, x2, y2],
                "bounding_box":       [x1, y1, x2, y2],
                "area_ratio":         round(area_ratio, 5),
                "location":           "unknown",
                "location_type":      "estimated",
                "deformation_index":  0.0,
                "relative_deformation_index": 0.0,
                "low_confidence_flag": float(obj_pred.score.value) < cfg.CONFIDENCE_RECHECK_LIMIT,
                "verification_status": "pending",
                "scene_type":         conditions["scene_type"],
                "sahi_conf_used":     sahi_conf,
                "source":             "cv_model",
            })

        print(f"   ✅ cv_worker-{idx}: {len(stamped)} detections from {img_path}")

    except Exception as exc:
        print(f"   ❌ cv_worker-{idx} error: {exc}")

    return {
        "all_raw_detections": stamped,
        "messages": [
            {
                "role":      "system",
                "content":   f"cv_worker-{idx}: {len(stamped)} detections from {img_path}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# fusion_node
# ─────────────────────────────────────────────────────────────────────────────

def fusion_node(state: SmartForgeState) -> dict:
    """
    LangGraph reduce node: fuse parallel worker detections.

    Reads from state
    ----------------
    all_raw_detections (accumulated from all cv_workers via Annotated reducer)

    Returns partial state update
    ----------------------------
    fused_detections, raw_detections (alias), fraud_report (recycling flags),
    pipeline_trace, messages
    """
    all_dets = state.get("all_raw_detections", state.get("raw_detections", []))
    print(f"\n🔵 [fusion] reducing {len(all_dets)} raw detections across images…")

    fused, recycling_flags, graph_stats = fuse_detections(all_dets)

    n_images = graph_stats["images_processed"]
    n_parts  = graph_stats["parts_detected"]
    print(
        f"   Graph: {graph_stats['nodes']} nodes | {graph_stats['edges']} edges"
    )
    print(f"   Covers {n_images} image(s) → {n_parts} unique car part(s)")
    print(
        f"\n   📊 Fusion result: {len(all_dets)} raw → {len(fused)} unique damages"
    )

    # ── Append recycling fraud flags to existing fraud_report ─────────────────
    fraud_report = dict(state.get("fraud_report") or {})
    if recycling_flags:
        existing = fraud_report.get("flags", [])
        fraud_report["flags"]              = existing + recycling_flags
        fraud_report["status"]             = "SUSPICIOUS_HIGH_RISK"
        fraud_report["recycling_detected"] = True
        for fl in recycling_flags:
            print(f"   ⚠️  {fl}")

    trace_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reasoning": (
            f"Fusion reduced {len(all_dets)} raw detections across {n_images} "
            f"images to {len(fused)} unique damages. "
            f"Graph: {graph_stats['nodes']} nodes, {graph_stats['edges']} edges. "
            f"Recycling flags: {len(recycling_flags)}."
        ),
        "method":      "NetworkX Part-based Fusion",
        "graph_stats": graph_stats,
    }

    return {
        "fused_detections": fused,
        "raw_detections":   fused,   # downstream nodes read raw_detections
        "fraud_report":     fraud_report,
        "pipeline_trace": {
            **state["pipeline_trace"],
            "fusion_agent": trace_entry,
        },
        "messages": [
            {
                "role":    "system",
                "content": (
                    f"fusion: {len(all_dets)} raw → {len(fused)} unique damages "
                    f"across {n_images} images"
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
    }
