"""
SmartForge — LangGraph State Schema
=====================================
Single source of truth that flows through every graph node.
All node functions receive this typed dict and return a *partial* update —
LangGraph merges updates into the running state automatically.

Design rules
------------
- Message History uses ``Annotated[List[dict], operator.add]`` so that every
  node's log entries are *appended*, never overwritten, across the whole run.
- ``verified_damages``, ``all_raw_detections``, and ``fused_detections`` also
  use the append reducer so parallel workers (Batch 2 fan-out) can safely
  accumulate results without mutation conflicts.
- All other fields use default LangGraph merge semantics (last-write wins).

Field categories
----------------
Message History     — chronological agent logs
Workflow Context    — extracted images, arrays, detection lists
Gemini VLM Agent    — vehicle type, make, enrichment metadata
Intelligent Intake  — adaptive SAHI conf, scene type
Validation Metrics  — health_score, validation_passed, retry_count
Batch 3 Golden Frame— verified_damages, golden_crops
Batch 2 Map-Reduce  — image_paths, all_raw_detections, fused_detections
Batch 1 Fraud Layer — fraud_report, claim_date, GPS coords
Batch 4 Financial   — total_loss_flag, financial_estimate
Audit Metadata      — job_id, vehicle_id, pipeline_trace
"""

import operator
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from src.config.settings import cfg


# ─────────────────────────────────────────────────────────────────────────────
# State schema
# ─────────────────────────────────────────────────────────────────────────────

class SmartForgeState(TypedDict):
    # ── Message History (reducer: append, never overwrite) ────────────────────
    messages: Annotated[List[dict], operator.add]

    # ── Workflow Context ──────────────────────────────────────────────────────
    image_path:      str
    image_bgr:       Optional[Any]    # numpy BGR array (freed after perception)
    image_rgb:       Optional[Any]    # numpy RGB array
    raw_detections:  List[dict]       # PerceptionAgent output
    depth_map:       Optional[Any]    # MiDaS depth map numpy array
    damages_output:  List[dict]       # ReasoningAgent output
    final_output:    Optional[dict]   # complete result assembled by report_node

    # ── Gemini VLM Agent ──────────────────────────────────────────────────────
    vehicle_type:             str     # "car" | "2W" | "3W" | "truck" | "unknown"
    vehicle_type_confidence:  float   # Gemini confidence 0.0–1.0
    vehicle_make_estimate:    str     # e.g. "sedan-class", "hatchback"
    gemini_agent_ran:         bool    # True if Gemini API call succeeded
    gemini_discovered_count:  int     # damages found by full-image scan (Task 4)

    # ── Intelligent Intake Analysis ───────────────────────────────────────────
    adaptive_sahi_conf: float         # computed by intake_node from image analysis
    scene_type:         str           # "high_reflection" | "normal" | "low_contrast"

    # ── Validation Metrics (HealthMonitor reads + writes) ─────────────────────
    health_score:       float
    validation_passed:  bool
    validation_errors:  List[str]
    retry_count:        int

    # ── Batch 3: Golden Frame Verification ────────────────────────────────────
    # Annotated reducer prevents overwrite when multiple workers write in parallel
    verified_damages: Annotated[List[dict], operator.add]
    golden_crops:     List[dict]      # crop metadata for the audit trail

    # ── Batch 2: Multi-Image Map-Reduce ───────────────────────────────────────
    image_paths:         List[str]    # ALL image paths for a multi-image claim
    all_raw_detections:  Annotated[List[dict], operator.add]  # fan-out accumulator
    fused_detections:    Annotated[List[dict], operator.add]  # post-fusion records

    # ── Batch 1: Fraud & Integrity Layer ──────────────────────────────────────
    is_fraud:       bool              # explicit fraud flag for DB / dashboard routing
    fraud_attempts: int               # 3-strike counter
    fraud_report:   Optional[dict]    # trust_score, status, per-image details
    claim_date:     str               # ISO date of claimed accident
    claim_lat:      float             # claimed GPS latitude
    claim_lon:      float             # claimed GPS longitude

    # ── Batch 4: Financial Intelligence ───────────────────────────────────────
    pipeline_stability_flag: str      # "Stable" | "Unstable" | "CircuitBreaker"
    total_loss_flag:         bool     # True when repair_cost > vehicle_value × threshold
    financial_estimate:      Optional[dict]  # line_items, totals, disposition

    # ── Audit Metadata ────────────────────────────────────────────────────────
    job_id:         str
    vehicle_id:     str
    policy_id:      str
    pipeline_trace: dict
    started_at:     str


# ─────────────────────────────────────────────────────────────────────────────
# State factory
# ─────────────────────────────────────────────────────────────────────────────

def make_initial_state(
    image_path: str,
    vehicle_id: str = "",
    policy_id:  str = "",
    claim_date: str = "",
    claim_lat:  float = 0.0,
    claim_lon:  float = 0.0,
    image_paths: Optional[List[str]] = None,
) -> SmartForgeState:
    """
    Build the initial state dict before graph execution begins.

    Parameters
    ----------
    image_path  : str   — primary image path (required)
    vehicle_id  : str   — vehicle / claimant identifier
    policy_id   : str   — insurance policy number (optional)
    claim_date  : str   — ISO date of claimed accident (optional)
    claim_lat   : float — claimed GPS latitude (optional)
    claim_lon   : float — claimed GPS longitude (optional)
    image_paths : list  — all image paths for multi-image claims; when None
                          defaults to [image_path]

    Returns
    -------
    SmartForgeState — ready to pass to graph.stream() / graph.invoke()
    """
    vid    = vehicle_id or cfg.VEHICLE_ID or "VH-UNKNOWN"
    pol    = policy_id  or cfg.POLICY_ID  or ""
    now    = datetime.now(timezone.utc)
    job_id = f"{vid}-{uuid.uuid4().hex[:6].upper()}-{now.strftime('%Y%m%dT%H%M%S')}"

    paths  = image_paths if image_paths else [image_path]

    return SmartForgeState(
        # Message History
        messages = [{
            "role":      "system",
            "content":   f"Job {job_id} started",
            "timestamp": now.isoformat(),
        }],

        # Workflow Context
        image_path     = image_path,
        image_bgr      = None,
        image_rgb      = None,
        raw_detections = [],
        depth_map      = None,
        damages_output = [],
        final_output   = None,

        # Gemini VLM Agent
        vehicle_type            = "unknown",
        vehicle_type_confidence = 0.0,
        vehicle_make_estimate   = "unknown",
        gemini_agent_ran        = False,
        gemini_discovered_count = 0,

        # Intelligent Intake
        adaptive_sahi_conf = cfg.SAHI_CONFIDENCE,
        scene_type         = "unknown",

        # Validation Metrics
        health_score      = 1.0,
        validation_passed = False,
        validation_errors = [],
        retry_count       = 0,

        # Batch 3: Golden Frame
        verified_damages = [],
        golden_crops     = [],

        # Batch 2: Multi-Image Map-Reduce
        image_paths        = paths,
        all_raw_detections = [],
        fused_detections   = [],

        # Batch 1: Fraud Layer
        is_fraud       = False,
        fraud_attempts = 0,
        fraud_report   = None,
        claim_date     = claim_date or cfg.CLAIM_ACCIDENT_DATE,
        claim_lat      = claim_lat  or cfg.CLAIM_LOSS_LOCATION_LAT,
        claim_lon      = claim_lon  or cfg.CLAIM_LOSS_LOCATION_LON,

        # Batch 4: Financial Intelligence
        pipeline_stability_flag = "Stable",
        total_loss_flag         = False,
        financial_estimate      = None,

        # Audit Metadata
        job_id         = job_id,
        vehicle_id     = vid,
        policy_id      = pol,
        pipeline_trace = {},
        started_at     = now.isoformat() + "Z",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared message-building helper
# ─────────────────────────────────────────────────────────────────────────────

def log_msg(agent: str, content: str) -> dict:
    """
    Build a single message dict for appending to state["messages"].

    Usage inside any node:
        return {
            "messages": [log_msg("intake_agent", "Image accepted.")],
            ...
        }
    """
    return {
        "role":      agent,
        "content":   content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
