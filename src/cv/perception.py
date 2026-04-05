"""
SmartForge — Perception CV Helpers
=====================================
Pure CV functions for image analysis, SAHI inference, SAM segmentation,
part detection, damage location, severity classification, and cost estimation.
No LangGraph state, no file I/O side-effects beyond model loading.

Public API
----------
    analyse_image_conditions(image_bgr, sahi_slice_size) → dict
    run_sahi_detection(img_path, conf, device)            → list[ObjectPrediction]
    run_sam_segmentation(image_rgb, bbox)                 → np.ndarray (mask)
    run_part_detection(image_rgb)                         → (list[dict], YOLO|None)
    compute_iou(a, b)                                     → float
    get_damage_location_unified(image_rgb, bbox,
        part_boxes, vd, gemini_vehicle_type)              → (str, str)
    compute_severity(damage_type, rel_deformation,
        area_ratio)                                       → (severity, category)
    severity_to_score(severity)                           → int
    estimate_cost(damage_type, severity)                  → (cost_range, repair_type)
"""

import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.config.settings import cfg


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive image-condition analysis  (Task 1.1 + 1.2 from the notebook)
# ─────────────────────────────────────────────────────────────────────────────

def analyse_image_conditions(
    image_bgr: np.ndarray,
    sahi_slice_size: int = 640,
) -> Dict[str, Any]:
    """
    Analyse an image to determine the optimal SAHI confidence and whether
    downsampling would destroy fine-detail (scratch) information.

    Specular Highlight Detection (Task 1.1)
    ----------------------------------------
    Converts to HSV and computes variance of the V (Value/brightness) channel.
    High variance → shiny/reflective surface → raise SAHI confidence to reduce
    specular false positives.
    Low variance  → dark/matte surface       → lower confidence to catch subtle
    damage.

    Adaptive Downsampling Flag (Task 1.2)
    --------------------------------------
    If downsampling a large image would reduce the short side below
    2 × sahi_slice_size, the vehicle is likely small in frame — downsampling
    would lose sub-tile detail needed for scratch detection.
    When True, the caller should skip the 4096-cap resize.

    Parameters
    ----------
    image_bgr      : OpenCV BGR numpy array
    sahi_slice_size : SAHI tile size (default 640)

    Returns
    -------
    dict with keys:
        v_variance          float
        scene_type          "high_reflection" | "normal" | "low_contrast"
        adaptive_sahi_conf  float
        skip_downsampling   bool
        reasoning           str
    """
    h, w = image_bgr.shape[:2]

    # ── Specular highlight detection ──────────────────────────────────────────
    hsv        = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v_channel  = hsv[:, :, 2].astype(np.float32)
    v_variance = float(np.var(v_channel))

    if v_variance > 3000:
        scene_type    = "high_reflection"
        adaptive_conf = 0.45
        reasoning = (
            f"V-channel variance={v_variance:.0f} > 3000 — shiny/reflective "
            f"surface detected. SAHI confidence raised to {adaptive_conf} to "
            "reduce specular false positives."
        )
    elif v_variance < 1000:
        scene_type    = "low_contrast"
        adaptive_conf = 0.25
        reasoning = (
            f"V-channel variance={v_variance:.0f} < 1000 — dark or low-contrast "
            f"image. SAHI confidence lowered to {adaptive_conf} to detect subtle "
            "damage on matte surfaces."
        )
    else:
        scene_type    = "normal"
        adaptive_conf = cfg.SAHI_CONFIDENCE
        reasoning = (
            f"V-channel variance={v_variance:.0f} in normal range (1000–3000). "
            f"Using default SAHI confidence={adaptive_conf}."
        )

    # ── Adaptive downsampling check ───────────────────────────────────────────
    skip_downsampling = False
    if max(h, w) > 4096:
        scale_factor     = 4096 / max(h, w)
        short_side_after = min(h, w) * scale_factor
        if short_side_after < sahi_slice_size * 2:
            skip_downsampling = True
            reasoning += (
                f" | Image is {w}×{h}px. After 4096-cap downsampling, short "
                f"side would be {short_side_after:.0f}px < "
                f"{sahi_slice_size * 2}px (2×SAHI tile). Skipping downsampling "
                "to preserve scratch-level detail."
            )

    return {
        "v_variance":         round(v_variance, 1),
        "scene_type":         scene_type,
        "adaptive_sahi_conf": round(adaptive_conf, 3),
        "skip_downsampling":  skip_downsampling,
        "reasoning":          reasoning,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SAHI detection
# ─────────────────────────────────────────────────────────────────────────────

def run_sahi_detection(
    img_path: str,
    conf: float,
    device: str = "cpu",
) -> list:
    """
    Run SAHI sliced inference using the damage segmentation model.

    Parameters
    ----------
    img_path : str   — path to the image file
    conf     : float — YOLO confidence threshold (adaptive, from analyse_image_conditions)
    device   : str   — "cuda" or "cpu"

    Returns
    -------
    list of sahi ObjectPrediction objects
    """
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    damage_model = AutoDetectionModel.from_pretrained(
        model_type           = "ultralytics",
        model_path           = cfg.DAMAGE_MODEL_PATH,
        confidence_threshold = conf,
        device               = device,
    )
    result = get_sliced_prediction(
        img_path,
        damage_model,
        slice_height         = cfg.SAHI_SLICE_SIZE,
        slice_width          = cfg.SAHI_SLICE_SIZE,
        overlap_height_ratio = cfg.SAHI_OVERLAP,
        overlap_width_ratio  = cfg.SAHI_OVERLAP,
    )
    return result.object_prediction_list


# ─────────────────────────────────────────────────────────────────────────────
# SAM segmentation
# ─────────────────────────────────────────────────────────────────────────────

def run_sam_segmentation(
    image_rgb: np.ndarray,
    bbox: List[int],
) -> np.ndarray:
    """
    Run SAM (Segment Anything Model) on a single bounding box.

    Downloads the checkpoint automatically if it is not on disk.

    Parameters
    ----------
    image_rgb : np.ndarray  — RGB image (H, W, 3)
    bbox      : list[int]   — [x1, y1, x2, y2] in pixels

    Returns
    -------
    np.ndarray — binary mask of shape (H, W), dtype bool
    """
    from segment_anything import SamPredictor, sam_model_registry

    # Auto-download checkpoint if absent
    if not os.path.exists(cfg.SAM_CHECKPOINT):
        print(f"   [SAM] Downloading checkpoint to {cfg.SAM_CHECKPOINT}…")
        subprocess.run(
            ["wget", "-q", cfg.SAM_URL, "-O", cfg.SAM_CHECKPOINT],
            check=True,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam    = sam_model_registry["vit_b"](checkpoint=cfg.SAM_CHECKPOINT)
    sam.to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    masks, _, _ = predictor.predict(
        box                 = np.array(bbox),
        multimask_output    = False,
    )
    return masks[0]  # shape (H, W), dtype bool


# ─────────────────────────────────────────────────────────────────────────────
# Part detection
# ─────────────────────────────────────────────────────────────────────────────

def run_part_detection(
    image_rgb: np.ndarray,
) -> Tuple[List[Dict[str, Any]], Optional[Any]]:
    """
    Run the custom part-detection YOLO model + the standard COCO detector.

    Returns
    -------
    part_boxes : list of {"name": str, "bbox": [x1,y1,x2,y2]}
    vd         : YOLO model for vehicle-type detection (or None on error)
    """
    from ultralytics import YOLO

    part_boxes: List[Dict[str, Any]] = []
    vd = None

    try:
        pm = YOLO(cfg.PART_MODEL_PATH)
        part_boxes = [
            {
                "name": pm.names[int(b.cls[0])],
                "bbox": list(map(int, b.xyxy[0])),
            }
            for r in pm(image_rgb)
            for b in r.boxes
        ]
        vd = YOLO("yolov8n.pt")   # COCO detector for vehicle-type fallback
    except Exception as exc:
        print(f"   [part_detection] Warning: {exc}")

    return part_boxes, vd


# ─────────────────────────────────────────────────────────────────────────────
# Geometric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou(
    a: List[int],
    b: List[int],
) -> float:
    """
    Compute Intersection-over-Union between two bounding boxes.

    Parameters
    ----------
    a, b : [x1, y1, x2, y2]

    Returns
    -------
    float in [0, 1]
    """
    xA = max(a[0], b[0]);  yA = max(a[1], b[1])
    xB = min(a[2], b[2]);  yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def get_damage_location_unified(
    image_rgb: np.ndarray,
    bbox: List[int],
    part_boxes: List[Dict[str, Any]],
    vd: Optional[Any],
    gemini_vehicle_type: str = "unknown",
) -> Tuple[str, str]:
    """
    Determine the vehicle part / spatial zone where a damage was detected.

    Priority
    --------
    1. Gemini vehicle type already classified → skip COCO inference overhead.
    2. COCO detector (vd) available           → infer vehicle type from image.
    3. Fallback                               → spatial zone mapping.

    For cars with confirmed part boxes: use IoU matching against part_boxes.
    For non-car / zone fallback: map centre-of-bbox to a 3×3×3 spatial grid.

    Returns
    -------
    (location_str, location_type)
    location_type: "detected" | "estimated"
    """
    x1, y1, x2, y2 = bbox
    h, w, _ = image_rgb.shape

    # ── Determine vehicle type ────────────────────────────────────────────────
    vt = gemini_vehicle_type
    if vt in ("unknown", "") and vd is not None:
        vt = "unknown"
        try:
            for r in vd(image_rgb):
                for box in r.boxes:
                    lbl = vd.names[int(box.cls[0])]
                    if lbl == "car":
                        vt = "car"; break
                    elif lbl in ("motorcycle", "bicycle"):
                        vt = "2w"; break
                if vt != "unknown":
                    break
        except Exception:
            pass

    # ── Car + part boxes → IoU-based location ─────────────────────────────────
    if vt == "car" and part_boxes:
        best_part = "Unknown"
        best_iou  = 0.0
        for p in part_boxes:
            iou = compute_iou(bbox, p["bbox"])
            if iou > best_iou:
                best_iou  = iou
                best_part = p["name"]
        canonical = cfg.PART_NAME_MAP.get(best_part, best_part)
        return canonical, "detected"

    # ── Spatial zone fallback ─────────────────────────────────────────────────
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    h_ = "Left"   if cx < 0.33 else ("Center" if cx < 0.66 else "Right")
    v_ = "Front"  if cy < 0.33 else ("Middle" if cy < 0.66 else "Rear")
    ht = "Upper Section" if cy < 0.3 else ("Lower Section" if cy > 0.7 else "Main Body")
    zone = f"{v_} {h_} {ht}"
    return cfg.ZONE_LANGUAGE_MAP.get(zone, zone), "estimated"


# ─────────────────────────────────────────────────────────────────────────────
# Severity classification
# ─────────────────────────────────────────────────────────────────────────────

def compute_severity(
    damage_type: str,
    rel_deformation: float,
    area_ratio: float,
) -> Tuple[str, str]:
    """
    Classify damage severity and category from CV signals.

    Parameters
    ----------
    damage_type       : YOLO class name (e.g. "Dent", "Scratch")
    rel_deformation   : relative depth-variance index from MiDaS (0+)
    area_ratio        : bbox area / image area (0–1)

    Returns
    -------
    (severity, category)
    severity : "Low" | "Medium" | "High"
    category : "Cosmetic" | "Functional" | "Moderate"
    """
    t  = damage_type
    rv = rel_deformation
    ar = area_ratio

    if t in ("Missing part", "Broken part", "Cracked"):
        return "High", "Functional"

    if t == "Dent":
        if rv > 0.02:  return "High",   "Functional"
        if rv > 0.005: return "Medium",  "Functional"
        return "Low", "Cosmetic"

    if t in ("Scratch", "Paint chip", "Flaking", "Corrosion"):
        if ar < 0.005: return "Low",    "Cosmetic"
        if ar < 0.02:  return "Medium", "Cosmetic"
        return "Medium", "Moderate"

    return "Low", "Cosmetic"


def severity_to_score(severity: str) -> int:
    """
    Convert a severity label to a health-score penalty.

    Used by reasoning_node and decision_node to compute the
    overall_assessment_score (100 - sum_of_penalties).
    """
    return {"Low": 5, "Medium": 20, "High": 40}.get(severity, 5)


# ─────────────────────────────────────────────────────────────────────────────
# Cost estimation (quick INR range — pre Batch-4 financial engine)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_cost(
    damage_type: str,
    severity: str,
) -> Tuple[str, str]:
    """
    Look up a rough INR repair-cost range from the COST_TABLE.

    Returns
    -------
    (cost_range, repair_description)
    Falls back to DEFAULT_COST when the combination is not in the table.
    """
    return cfg.COST_TABLE.get(
        (damage_type, severity),
        cfg.DEFAULT_COST,
    )
