"""
SmartForge cv sub-package.
Pure computer-vision helpers — no LangGraph state, no I/O side-effects.
Each module is independently importable for testing.
"""

from .fraud_checks import (
    haversine_km,
    parse_exif_gps,
    parse_exif_datetime,
    load_fraud_hash_db,
    save_fraud_hash_db,
    compute_phash,
    check_phash_against_db,
    check_reverse_image_serpapi,
    detect_screen_capture,
    perform_ela_check,
    check_ai_generation_with_fallback,
)
from .perception import (
    analyse_image_conditions,
    run_sahi_detection,
    run_sam_segmentation,
    run_part_detection,
    compute_iou,
    get_damage_location_unified,
    compute_severity,
    severity_to_score,
    estimate_cost,
)
from .depth import run_midas_depth, compute_deformation_index
from .fusion import build_claims_graph, fuse_detections, claims_graph

__all__ = [
    # fraud
    "haversine_km", "parse_exif_gps", "parse_exif_datetime",
    "load_fraud_hash_db", "save_fraud_hash_db",
    "compute_phash", "check_phash_against_db", "check_reverse_image_serpapi",
    "detect_screen_capture", "perform_ela_check",
    "check_ai_generation_with_fallback",
    # perception
    "analyse_image_conditions", "run_sahi_detection",
    "run_sam_segmentation", "run_part_detection",
    "compute_iou", "get_damage_location_unified",
    "compute_severity", "severity_to_score", "estimate_cost",
    # depth
    "run_midas_depth", "compute_deformation_index",
    # fusion
    "build_claims_graph", "fuse_detections", "claims_graph",
]
