"""
SmartForge — fraud_node + fraud_router
========================================
Batch 1: Multi-Agent Fraud & Integrity Layer.

Inserted immediately after intake_node — runs BEFORE any GPU or API
resources are consumed, so suspicious claims are halted at zero cost.

Five checks (all parallel within the node, results aggregated)
--------------------------------------------------------------
CHECK 1  Temporal Consistency   (+20 pts)
    EXIF DateTimeOriginal vs claim_date.
    Photo taken > 1 day before claim date → −25 pts.

CHECK 2  GPS Location Consistency   (+20 pts)
    EXIF GPS vs claimed loss location (Haversine).
    Distance > FRAUD_GPS_MAX_DISTANCE_KM → −30 pts.

CHECK 3  Software / Source Integrity   (+10 pts)
    EXIF Image Software tag.
    Adobe / Canva / Photoshop → −30 pts.
    Mobile camera origin → +10 pts.

CHECK 4  Reverse Image / Perceptual Hash   (+20 pts)
    pHash vs local fraud DB for cross-claim recycling detection.
    If SERPAPI_ENABLED: also queries Google Lens for internet matches.
    Duplicate found → −40 pts.

CHECK 5  Screen & AI-Generation Forensics   (+10 pts)
    5a. FFT Moiré + colour banding → detects photo-of-a-screen → −35 pts.
    5b. Winston AI (or ELA or Laplacian) → detects AI fakes → −35 pts.

Trust Score routing
-------------------
    trust_score ≥ FRAUD_TRUST_THRESHOLD → VERIFIED → perception / map_images
    trust_score <  FRAUD_TRUST_THRESHOLD → SUSPICIOUS_HIGH_RISK → human_audit

BYPASS_FRAUD
------------
    When cfg.BYPASS_FRAUD is True, all 5 checks are skipped instantly.
    The Gradio UI sets this to False when a user files an insurance claim
    (want_insurance = True) so the full layer activates for real submissions.

State mutations returned
------------------------
    fraud_report      dict  — trust_score, status, flags, per-check details
    is_fraud          bool
    pipeline_trace    dict  — "fraud_agent" entry added
    messages          list  — one entry appended
"""

import exifread
from datetime import datetime, timezone

from src.config.settings import cfg
from src.cv.fraud_checks import (
    check_ai_generation_with_fallback,
    check_phash_against_db,
    check_reverse_image_serpapi,
    detect_screen_capture,
    haversine_km,
    parse_exif_datetime,
    parse_exif_gps,
)
from src.graph.state import SmartForgeState, log_msg


# EXIF software tags that indicate post-processing / editing
_EDIT_TOOLS = (
    "Adobe", "Photoshop", "Lightroom", "Canva",
    "GIMP", "Snapseed", "PicsArt",
)
# EXIF software tags that indicate an original mobile capture
_MOBILE_HINTS = (
    "iPhone", "Android", "samsung", "xiaomi", "huawei",
    "Google", "realme", "oppo", "vivo", "OnePlus", "Nokia",
)


def fraud_node(state: SmartForgeState) -> dict:
    """
    LangGraph node: 5-check fraud & integrity layer.

    Reads from state
    ----------------
    image_path, claim_date, claim_lat, claim_lon, image_paths

    Returns partial state update
    ----------------------------
    fraud_report, is_fraud, pipeline_trace, messages
    """
    # ── BYPASS mode (demo / assessment-only) ──────────────────────────────────
    if cfg.BYPASS_FRAUD:
        print("\n⏩ [fraud_node] BYPASS_FRAUD=True — skipping all forensic checks.")
        image_paths = state.get("image_paths", [state["image_path"]])
        next_nd     = "map_images" if len(image_paths) > 1 else "perception"
        bypass_rep  = {
            "trust_score": 100,
            "status":      "VERIFIED",
            "flags":       [],
            "details":     {"note": "BYPASS_FRAUD=True — all checks skipped"},
            "next_node":   next_nd,
            "checked_at":  datetime.now(timezone.utc).isoformat(),
            "checks_run":  0,
        }
        trace_bp = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reasoning": "BYPASS_FRAUD=True — all 5 forensic checks skipped.",
            "decision":  f"Routing to {next_nd} (bypassed).",
            "details":   bypass_rep,
        }
        return {
            "fraud_report":   bypass_rep,
            "is_fraud":       False,
            "pipeline_trace": {**state["pipeline_trace"], "fraud_agent": trace_bp},
            "messages":       [log_msg("fraud_agent", "BYPASS_FRAUD=True — instant VERIFIED")],
        }

    # ── Full 5-check scan ─────────────────────────────────────────────────────
    img_path   = state["image_path"]
    claim_date = state.get("claim_date", cfg.CLAIM_ACCIDENT_DATE)
    claim_lat  = state.get("claim_lat",  cfg.CLAIM_LOSS_LOCATION_LAT)
    claim_lon  = state.get("claim_lon",  cfg.CLAIM_LOSS_LOCATION_LON)

    print(f"\n🛡️  [fraud] ── 5-Check Integrity Scan {'─'*30}")
    print(f"   Image      : {img_path}")
    print(f"   Claim date : {claim_date} | Location: ({claim_lat:.4f}, {claim_lon:.4f})")

    trust_score = 0
    flags:   list = []
    details: dict = {}

    # ── EXIF extraction (shared by checks 1-3) ────────────────────────────────
    try:
        with open(img_path, "rb") as fh:
            tags = exifread.process_file(fh, details=True)
    except Exception as exc:
        tags = {}
        flags.append(f"EXIF_READ_ERROR: {exc}")

    # ── CHECK 1: Temporal Consistency ─────────────────────────────────────────
    print("\n   [Check 1/5] Temporal Consistency…")
    photo_dt = parse_exif_datetime(tags)
    if photo_dt:
        trust_score += 20
        details["photo_datetime"] = photo_dt.isoformat()
        try:
            claim_dt   = datetime.strptime(claim_date, "%Y-%m-%d")
            delta_days = (claim_dt - photo_dt).days
            if delta_days > 1:
                trust_score -= 25
                flags.append(
                    f"TEMPORAL_MISMATCH: Photo taken {delta_days}d before claim "
                    f"({photo_dt.date()} vs {claim_date}). −25pts"
                )
                print(f"   ⚠️  FAIL — photo predates claim by {delta_days} days")
            else:
                print(f"   ✅ PASS (photo date: {photo_dt.date()})")
        except ValueError:
            flags.append("CLAIM_DATE_PARSE_ERROR")
    else:
        flags.append("NO_EXIF_DATETIME — cannot verify photo age")
        print("   ⚠️  WARN — no EXIF datetime found")

    # ── CHECK 2: GPS Location Consistency ────────────────────────────────────
    print("   [Check 2/5] GPS Location Consistency…")
    photo_lat, photo_lon = parse_exif_gps(tags)
    if photo_lat is not None:
        trust_score += 20
        dist_km = haversine_km(photo_lat, photo_lon, claim_lat, claim_lon)
        details["photo_gps"]       = {"lat": round(photo_lat, 5), "lon": round(photo_lon, 5)}
        details["gps_distance_km"] = round(dist_km, 2)
        if dist_km > cfg.FRAUD_GPS_MAX_DISTANCE_KM:
            trust_score -= 30
            flags.append(
                f"GPS_MISMATCH: Photo {dist_km:.1f}km from claim location "
                f"(max {cfg.FRAUD_GPS_MAX_DISTANCE_KM}km). −30pts"
            )
            print(f"   ⚠️  FAIL — {dist_km:.1f}km from claimed loss location")
        else:
            print(f"   ✅ PASS ({dist_km:.1f}km from claimed location)")
    else:
        flags.append("NO_EXIF_GPS — possible stock/internet image")
        print("   ⚠️  WARN — no GPS metadata (stock-image risk signal)")

    # ── CHECK 3: Software / Source Integrity ──────────────────────────────────
    print("   [Check 3/5] Software & Source Integrity…")
    software_tag = str(tags.get("Image Software", "None")).strip()
    details["exif_software"] = software_tag

    if any(t.lower() in software_tag.lower() for t in _EDIT_TOOLS):
        trust_score -= 30
        flags.append(f"EDITING_SOFTWARE: '{software_tag}' detected. −30pts")
        print(f"   ⚠️  FAIL — editing software detected: {software_tag}")
    elif software_tag == "None" or any(
        h.lower() in software_tag.lower() for h in _MOBILE_HINTS
    ):
        trust_score += 10
        details["source_type"] = "original_mobile_capture"
        print(f"   ✅ PASS (software: {software_tag or 'None — original capture'})")
    else:
        flags.append(f"UNKNOWN_SOFTWARE: '{software_tag}'")
        print(f"   ⚠️  WARN — unrecognised software: {software_tag}")

    # ── CHECK 4: Reverse Image / Perceptual Hash ──────────────────────────────
    print("   [Check 4/5] Reverse Image & Perceptual Hash…")
    hash_result = check_phash_against_db(img_path)
    details["phash_check"] = hash_result

    if hash_result["status"] == "DUPLICATE_DETECTED":
        trust_score -= 40
        flags.append(
            f"RECYCLED_IMAGE: pHash match "
            f"(Hamming={hash_result['hamming_distance']}) "
            f"against prior claim '{hash_result['matched_claim']}'. −40pts"
        )
        print(
            f"   🚨 FAIL — near-duplicate of known claim image "
            f"(Hamming={hash_result['hamming_distance']})"
        )
    elif hash_result["status"] == "UNIQUE":
        trust_score += 20
        print(f"   ✅ PASS — unique image enrolled (pHash: {hash_result['phash'][:16]}…)")
    else:
        flags.append(f"PHASH_ERROR: {hash_result.get('status', 'unknown')}")
        print(f"   ⚠️  WARN — pHash check failed: {hash_result.get('status')}")

    # Optional SerpAPI Google Lens cross-check
    if cfg.SERPAPI_ENABLED:
        print("      [4b] SerpAPI Google Lens reverse-image check…")
        serp_result = check_reverse_image_serpapi(img_path)
        details["serpapi_check"] = serp_result
        if serp_result.get("found_online"):
            trust_score -= 35
            flags.append(
                f"INTERNET_IMAGE: Google Lens matched "
                f"'{serp_result.get('match_title', '')}' at "
                f"{serp_result.get('match_url', '')}. −35pts"
            )
            print(f"   🚨 FAIL — image found on internet: {serp_result.get('match_title')}")
        elif serp_result.get("found_online") is False:
            print("      ✅ Not found in Google Lens index")
        else:
            print(f"      ⚠️  SerpAPI skipped/error: {serp_result.get('reason', '')}")

    # ── CHECK 5: Screen & AI-Generation Forensics ─────────────────────────────
    print("   [Check 5/5] Screen Capture & AI-Generation Forensics…")

    # 5a — Screen / Display Detection
    screen_result = detect_screen_capture(img_path)
    details["screen_detection"] = screen_result
    if screen_result["is_screen"]:
        trust_score -= 35
        flags.append(
            f"SCREEN_CAPTURE: FFT/colour analysis detected photo-of-a-screen "
            f"(confidence={screen_result['confidence']:.2f}). "
            f"Signals: {screen_result['signals']}. −35pts"
        )
        print(f"   🚨 FAIL — screen capture detected (confidence={screen_result['confidence']:.2f})")
        for sig in screen_result["signals"]:
            print(f"      • {sig}")
    else:
        trust_score += 10
        print(
            f"   ✅ PASS — real-world photo confirmed "
            f"(screen confidence={screen_result['confidence']:.2f})"
        )

    # 5b — AI Generation / Deepfake Detection
    print("      [5b] AI-generation forensics (Winston AI / ELA / Laplacian)…")
    ai_result = check_ai_generation_with_fallback(img_path)
    details["ai_generation_check"] = ai_result

    if ai_result.get("is_ai_generated"):
        trust_score -= 35
        flags.append(
            f"AI_GENERATED: {ai_result.get('method', 'unknown')} reports "
            f"AI probability={ai_result.get('ai_probability', 'N/A')}. −35pts"
        )
        print(
            f"   🚨 FAIL — AI-generated image detected via "
            f"{ai_result.get('method', 'unknown')} "
            f"(p={ai_result.get('ai_probability', 'N/A')})"
        )
    elif ai_result.get("is_ai_generated") is False:
        print(
            f"   ✅ PASS — authentic image "
            f"({ai_result.get('method', 'unknown')}, "
            f"ai_prob={ai_result.get('ai_probability', 'N/A')})"
        )
    else:
        print(f"   ⚠️  WARN — AI check inconclusive: {ai_result.get('method', 'error')}")

    # ── Final trust score & routing ───────────────────────────────────────────
    trust_score = max(0, min(100, trust_score))
    status      = (
        "VERIFIED"
        if trust_score >= cfg.FRAUD_TRUST_THRESHOLD
        else "SUSPICIOUS_HIGH_RISK"
    )
    image_paths = state.get("image_paths", [state["image_path"]])
    next_node   = (
        ("map_images" if len(image_paths) > 1 else "perception")
        if status == "VERIFIED"
        else "human_audit"
    )

    fraud_report = {
        "trust_score": trust_score,
        "status":      status,
        "flags":       flags,
        "details":     details,
        "next_node":   next_node,
        "checked_at":  datetime.now(timezone.utc).isoformat(),
        "checks_run":  5,
    }

    icon = "✅" if status == "VERIFIED" else "🚨"
    print(f"\n{'═'*60}")
    print(f"  {icon}  [fraud] Trust Score: {trust_score}/100 → {status}")
    if flags:
        print("  🔴 Active Flags:")
        for fl in flags:
            print(f"     • {fl}")
    print(f"  ➡️  Next node: {next_node}")
    print(f"{'═'*60}")

    trace_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reasoning": (
            f"5-check fraud scan on {img_path}. "
            f"Trust={trust_score}/100. Status={status}. "
            f"Flags={len(flags)}: {flags}"
        ),
        "decision": f"Routing to {next_node}.",
        "details":  fraud_report,
    }

    return {
        "fraud_report":   fraud_report,
        "is_fraud":       status == "SUSPICIOUS_HIGH_RISK",
        "pipeline_trace": {**state["pipeline_trace"], "fraud_agent": trace_entry},
        "messages": [
            log_msg(
                "fraud_agent",
                f"Trust={trust_score}/100 status={status} flags={len(flags)}",
            )
        ],
    }


def fraud_router(state: SmartForgeState) -> str:
    """
    Conditional edge function after fraud_node.

    Routes
    ------
    SUSPICIOUS_HIGH_RISK → "human_audit"
    VERIFIED + multi-image → "map_images"   (Batch 2 fan-out)
    VERIFIED + single-image → "perception"
    """
    fr          = state.get("fraud_report") or {}
    if fr.get("status") == "SUSPICIOUS_HIGH_RISK":
        return "human_audit"
    image_paths = state.get("image_paths", [state.get("image_path", "")])
    if len(image_paths) > 1:
        return "map_images"
    return "perception"
