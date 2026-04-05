"""
SmartForge — gemini_agent_node
================================
Gemini VLM as a supporting AI agent.  Positioned between perception_node
and false_positive_gate_node so CV detections are enriched before any
filtering or health-checking occurs.

Three API calls maximum (down from N + M + 1 in earlier versions)
------------------------------------------------------------------
Call A  Task 1  Vehicle type classification
            Sends image + CV context JSON.  Returns vehicle_type,
            vehicle_make_estimate, confidence.

Call B  Tasks 2 + 3  BATCHED location enrichment + low-conf verification
            One call regardless of how many detections need enrichment.
            - Enriches location for detections where location_type=="estimated"
              (YOLO part model missed the part or zone-fallback was used).
            - Confirms or rejects detections flagged low_confidence_flag==True.
            - Detections with location_type=="detected" are skipped entirely
              (YOLO part-model result is trusted without Gemini overhead).

Call C  Task 4  Full-image scan for missed damages
            Compares existing detections to the image and reports new damage
            regions the CV model missed.  Strict rules embedded in the prompt:
            ignore road surface, background objects, shadows, reflections,
            number plates.  Confidence threshold 65 % minimum.

Failure handling
----------------
Every call is wrapped in _call_gemini() which returns {"_error": "..."} on
any failure.  The node degrades gracefully: missing Call A → zone locations
used as-is; missing Call B → low-conf flags remain pending; missing Call C
→ no synthetic detections added.  The pipeline never halts due to Gemini.

State mutations returned
------------------------
    raw_detections            list  — enriched + Gemini-discovered detections
    vehicle_type              str
    vehicle_type_confidence   float
    vehicle_make_estimate     str
    gemini_agent_ran          bool
    gemini_discovered_count   int
    pipeline_trace            dict  — "gemini_agent" entry added
    messages                  list  — one entry appended
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from src.config.settings import cfg
from src.models.gemini_client import call_gemini
from src.graph.state import SmartForgeState, log_msg


def gemini_agent_node(state: SmartForgeState) -> dict:
    """
    LangGraph node: Gemini VLM supporting agent (batched, max 3 API calls).
    """
    print("\n🔮 [gemini_agent] starting VLM enrichment (batched)…")

    if not cfg.GEMINI_ENABLED:
        print("   ⏭️  Gemini disabled (no API key) — skipping enrichment")
        trace = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reasoning": "GEMINI_ENABLED=False. No API key configured.",
            "decision":  "Skipped. CV pipeline outputs used as-is.",
            "details":   {"skipped": True},
        }
        return {
            "gemini_agent_ran":        False,
            "gemini_discovered_count": 0,
            "pipeline_trace": {**state["pipeline_trace"], "gemini_agent": trace},
            "messages": [log_msg("gemini_agent", "Skipped — no API key.")],
        }

    image_path  = state["image_path"]
    detections  = state["raw_detections"]
    n_det       = len(detections)
    low_conf    = sum(1 for d in detections if d["low_confidence_flag"])
    n_estimated = sum(1 for d in detections if d["location_type"] == "estimated")

    gemini_ran       = False
    vehicle_type     = "unknown"
    vehicle_conf     = 0.0
    vehicle_make     = "unknown"
    enriched_details: List[str] = []

    # ── Compact CV context injected into every prompt ─────────────────────────
    pipeline_json = json.dumps({
        "cv_pipeline_output": {
            "total_detections":         n_det,
            "low_confidence_count":     low_conf,
            "estimated_location_count": n_estimated,
            "detections": [
                {
                    "id":                      d["detection_id"],
                    "type":                    d["type"],
                    "yolo_confidence":         d["confidence"],
                    "bounding_box":            d["bounding_box"],
                    "area_ratio":              d["area_ratio"],
                    "midas_depth_deformation": d["relative_deformation_index"],
                    "location_type":           d["location_type"],
                    "cv_location":             d["location"],
                }
                for d in detections
            ],
        }
    }, indent=2)

    # ── Call A: Vehicle Type Classification ──────────────────────────────────
    print("   Call A — Task 1: vehicle classification…")
    vt_prompt = (
        f"You are an automotive AI agent in a multi-stage insurance claims pipeline.\n\n"
        f"=== CV PIPELINE OUTPUT (YOLO + SAHI + MiDaS) ===\n{pipeline_json}\n\n"
        f"=== YOUR TASK ===\n"
        "Using the vehicle image AND the structured CV output above:\n"
        "1. Identify the vehicle type from the image.\n"
        "2. Use bounding boxes and area_ratio as spatial hints.\n"
        "3. Use midas_depth_deformation to understand damage severity context.\n"
        'vehicle_type options: "car" | "2W" | "3W" | "truck" | "unknown"\n'
        "2W = motorcycle/scooter/bicycle, 3W = auto-rickshaw/tuk-tuk"
    )
    vt_schema = (
        '{"vehicle_type": "car|2W|3W|truck|unknown", '
        '"vehicle_make_estimate": "sedan-class|hatchback|SUV|motorcycle|etc", '
        '"confidence": 0.0_to_1.0, "reasoning": "one sentence"}'
    )
    vt_result = call_gemini(vt_prompt, image_path, vt_schema)

    if "_error" not in vt_result:
        vehicle_type = vt_result.get("vehicle_type", "unknown")
        vehicle_conf = float(vt_result.get("confidence", 0.0))
        vehicle_make = vt_result.get("vehicle_make_estimate", "unknown")
        gemini_ran   = True
        print(f"   ✅ Vehicle: {vehicle_type} ({vehicle_make}) conf={vehicle_conf:.2f}")
    else:
        print(f"   ⚠️  Vehicle classification failed: {vt_result.get('_error','?')[:60]}")

    # ── Call B: Batch location enrichment + low-conf verification ────────────
    batch_items = [
        {
            "id":                      d["detection_id"],
            "type":                    d["type"],
            "bounding_box":            d["bounding_box"],
            "area_ratio":              round(d["area_ratio"], 5),
            "midas_depth_deformation": round(d["relative_deformation_index"], 6),
            "cv_location":             d["location"],
            "needs_location":          (
                d["location_type"] == "estimated"
                and not d.get("gemini_location")
            ),
            "needs_verification":      d["low_confidence_flag"],
            "yolo_confidence":         round(d["confidence"], 3),
        }
        for d in detections
        if d["location_type"] != "detected"   # skip part-model confirmed locations
    ]

    batch_results_map: Dict[str, Any] = {}

    if batch_items and gemini_ran:
        n_loc = sum(1 for b in batch_items if b["needs_location"])
        n_ver = sum(1 for b in batch_items if b["needs_verification"])
        print(
            f"   Call B — Tasks 2+3 BATCH: {len(batch_items)} item(s) "
            f"({n_loc} location enrichments + {n_ver} verifications → 1 call)…"
        )
        batch_prompt = (
            f"You are an automotive AI agent. Vehicle type from image: {vehicle_type}.\n\n"
            f"=== CV PIPELINE CONTEXT ===\n{pipeline_json}\n\n"
            "=== BATCH TASK ===\n"
            "For each detection in the list below, perform the indicated tasks:\n"
            "  - needs_location=true  → identify the EXACT vehicle part name\n"
            "  - needs_verification=true → confirm if the bbox region shows real vehicle damage\n"
            "  - A detection may need both, either, or neither.\n\n"
            f"Detections:\n{json.dumps(batch_items, indent=2)}\n\n"
            "Rules:\n"
            "- Use precise automotive terminology (e.g. 'Front left door panel')\n"
            "- For verification, look carefully at the exact pixel region of the bbox\n"
            "- verified=null if needs_verification=false\n"
            "- Return exactly one result object per detection id"
        )
        batch_schema = (
            '{"results": [{"id": "D001", '
            '"enriched_location": "specific vehicle part name", '
            '"location_confidence": 0.0_to_1.0, '
            '"verified": true_or_false_or_null, '
            '"verification_confidence": 0.0_to_1.0, '
            '"reasoning": "one sentence covering both tasks"}]}'
        )
        batch_resp = call_gemini(batch_prompt, image_path, batch_schema)
        if "_error" not in batch_resp:
            for item in batch_resp.get("results", []):
                det_id = item.get("id")
                if det_id:
                    batch_results_map[det_id] = item
            print(f"   ✅ Batch returned {len(batch_results_map)} result(s)")
        else:
            print(f"   ⚠️  Batch call failed: {batch_resp.get('_error','?')[:80]} — zone fallback used")
    elif batch_items and not gemini_ran:
        print("   ⏭️  Call B skipped — Task 1 failed, zone locations used as-is")
    else:
        print("   ⏭️  Call B skipped — all detections have confirmed part-model locations")

    # ── Apply batch results to each detection ─────────────────────────────────
    enriched_detections: List[Dict[str, Any]] = []
    for det in detections:
        det = dict(det)

        if det["location_type"] == "detected":
            det.update({
                "gemini_location":        None,
                "gemini_location_source": "cv_primary",
                "gemini_verified":        None,
                "gemini_reasoning":       "CV part model location used (location_type=detected).",
            })
            enriched_detections.append(det)
            continue

        result = batch_results_map.get(det["detection_id"])

        # Location enrichment
        if det["location_type"] == "estimated" and result and "_error" not in result:
            enriched_loc = result.get("enriched_location")
            if enriched_loc:
                det.update({
                    "gemini_location":        enriched_loc,
                    "gemini_location_source": "gemini_enriched",
                    "gemini_reasoning":       result.get("reasoning", ""),
                    "location":               enriched_loc,
                    "location_type":          "gemini_enriched",
                })
                enriched_details.append(f"{det['detection_id']}: {enriched_loc}")
            else:
                det.update({
                    "gemini_location":        None,
                    "gemini_location_source": "zone_fallback",
                    "gemini_reasoning":       "Batch result had no enriched_location.",
                })
        elif det["location_type"] == "gemini_enriched" and det.get("gemini_location"):
            pass  # already enriched on a prior retry — preserve as-is
        else:
            det.update({
                "gemini_location":        None,
                "gemini_location_source": "zone_fallback",
                "gemini_reasoning": (
                    "Gemini not available for enrichment."
                    if not gemini_ran
                    else "Batch result not found for this detection."
                ),
            })

        # Low-confidence verification
        det["gemini_verified"] = None
        if det["low_confidence_flag"] and result and "_error" not in result:
            raw_verified = result.get("verified")
            if raw_verified is None:
                raw_verified = result.get("real_damage", result.get("is_damage_real"))
            if raw_verified is not None:
                confirmed = bool(raw_verified)
                det["gemini_verified"]  = confirmed
                det["gemini_reasoning"] = (
                    det.get("gemini_reasoning", "")
                    + " | Gemini verify: "
                    + result.get("reasoning", "")
                )
                det["verification_status"] = "confirmed" if confirmed else "unconfirmed"
                icon = "✅" if confirmed else "🚩"
                print(f"   {icon} {det['detection_id']} ({det['type']}) {'verified' if confirmed else 'REJECTED'} by Gemini (batched)")

        enriched_detections.append(det)

    # ── Call C: Full-image scan for missed damages ────────────────────────────
    gemini_discovered: List[Dict[str, Any]] = []

    if gemini_ran:
        print("   Call C — Task 4: full-image missing-damage scan…")
        img_h, img_w = state["image_rgb"].shape[:2]
        img_area     = img_h * img_w

        existing_summary = "\n".join([
            f"  - {d['detection_id']}: {d['type']} bbox={d['bounding_box']} "
            f"conf={d['confidence']:.3f} loc={d['location']}"
            for d in enriched_detections
        ]) or "  (no detections yet)"

        fn_prompt = (
            "You are a Quality Assurance Claims Adjuster AI.\n\n"
            "The primary CV model has already detected these damages on this vehicle:\n"
            f"{existing_summary}\n\n"
            "Your task: Inspect the VEHICLE BODY PANELS for any clearly missed damage.\n"
            "Are there any OTHER visible damages that the CV model MISSED?\n\n"
            "STRICT RULES:\n"
            "- ONLY report damage ON THE VEHICLE BODY PANELS.\n"
            "  Ignore: road surface, floor, background, trees, walls, sky, shadows,\n"
            "  reflections in the paint, dirt marks, water stains, tyre tread, number plates.\n"
            "- Confidence threshold: only report if you are ≥65% confident.\n"
            "- Do NOT re-report the existing detections listed above.\n"
            f"- Use bounding box pixel coordinates [x1, y1, x2, y2]. "
            f"Image size: {img_w}×{img_h} pixels.\n"
            "- If no clearly missed body-panel damage exists, return empty list.\n\n"
            "Be STRICT and conservative — a false positive wastes adjuster time."
        )
        fn_schema = (
            '{"missed_damages": [{"type": "Scratch|Dent|Cracked|Broken part|Missing part|'
            'Paint chip|Flaking|Corrosion", '
            '"location": "specific vehicle part name", '
            '"bounding_box": [x1, y1, x2, y2], '
            '"confidence": 0.0_to_1.0, '
            '"reasoning": "one sentence — what you can see"}]}'
        )
        fn_result = call_gemini(fn_prompt, image_path, fn_schema)

        if "_error" not in fn_result:
            missed_list = fn_result.get("missed_damages", [])
            if not missed_list:
                print("   ✅ Task 4: no missed damages — CV detections are complete.")
            else:
                print(f"   🔍 Task 4: Gemini found {len(missed_list)} missed damage(s):")
                for i, m in enumerate(missed_list):
                    raw_bbox = m.get("bounding_box", [])
                    if not (isinstance(raw_bbox, list) and len(raw_bbox) == 4):
                        print(f"      ⚠️  GD{i+1:03d}: invalid bbox {raw_bbox} — skipped")
                        continue
                    x1, y1, x2, y2 = [max(0, int(v)) for v in raw_bbox]
                    x2 = min(x2, img_w);  y2 = min(y2, img_h)
                    if x2 <= x1 or y2 <= y1:
                        print(f"      ⚠️  GD{i+1:03d}: degenerate bbox — skipped")
                        continue
                    conf       = min(1.0, max(0.0, float(m.get("confidence", 0.7))))
                    area_ratio = ((x2 - x1) * (y2 - y1)) / (img_area + 1e-6)
                    d_id       = f"GD{i+1:03d}"
                    gemini_discovered.append({
                        "detection_id":               d_id,
                        "type":                       m.get("type", "Unknown"),
                        "location":                   m.get("location", "Unknown"),
                        "location_type":              "gemini_enriched",
                        "bounding_box":               [x1, y1, x2, y2],
                        "confidence":                 round(conf, 3),
                        "low_confidence_flag":        conf < 0.65,
                        "verification_status":        "gemini_confirmed",
                        "relative_deformation_index": 0.0,
                        "area_ratio":                 round(area_ratio, 6),
                        "gemini_location":            m.get("location", "Unknown"),
                        "gemini_location_source":     "gemini_discovery",
                        "gemini_reasoning":           m.get("reasoning", ""),
                        "gemini_verified":            True,
                        "rejected":                   False,
                        "rejection_reason":           None,
                        "source":                     "gemini_discovery",
                    })
                    print(
                        f"      ✅ {d_id}: {m.get('type')} @ {m.get('location')} "
                        f"(conf={conf:.2f})"
                    )
        else:
            print(f"   ⚠️  Task 4 failed gracefully: {fn_result.get('_error','?')[:80]}")
    else:
        print("   ⏭️  Call C skipped — Gemini not available.")

    all_detections = enriched_detections + gemini_discovered
    n_discovered   = len(gemini_discovered)
    api_calls_made = (
        (1 if gemini_ran else 0)
        + (1 if batch_items and gemini_ran else 0)
        + (1 if gemini_ran else 0)
    )

    trace_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reasoning": (
            f"Gemini Flash multimodal analysis (BATCHED). "
            f"Call A: vehicle_type={vehicle_type} ({vehicle_make}, conf={vehicle_conf:.2f}). "
            f"Call B: {len(batch_results_map)} detections enriched+verified in 1 call "
            f"(saved {max(0, n_estimated + low_conf - 1)} individual calls). "
            f"Call C: full-image scan found {n_discovered} missed damage(s)."
        ),
        "decision": (
            f"Vehicle identified as {vehicle_type}. "
            f"Enriched: {enriched_details if enriched_details else 'none'}. "
            f"Task4 added {n_discovered} Gemini-discovered detection(s)."
        ),
        "details": {
            "vehicle_type":       vehicle_type,
            "vehicle_make":       vehicle_make,
            "vehicle_conf":       vehicle_conf,
            "gemini_ran":         gemini_ran,
            "api_calls_made":     api_calls_made,
            "api_calls_saved":    max(0, n_estimated + low_conf - 1),
            "locations_enriched": len(enriched_details),
            "low_conf_verified":  low_conf,
            "gemini_discovered":  n_discovered,
        },
    }

    print(
        f"✅ gemini_agent: vehicle={vehicle_type} | enriched={len(enriched_details)} | "
        f"verified={low_conf} low-conf | task4={n_discovered} | ran={gemini_ran}"
    )

    return {
        "raw_detections":           all_detections,
        "vehicle_type":             vehicle_type,
        "vehicle_type_confidence":  vehicle_conf,
        "vehicle_make_estimate":    vehicle_make,
        "gemini_agent_ran":         gemini_ran,
        "gemini_discovered_count":  n_discovered,
        "pipeline_trace": {
            **state["pipeline_trace"],
            "gemini_agent": trace_entry,
        },
        "messages": [
            log_msg(
                "gemini_agent",
                f"vehicle={vehicle_type} enriched={len(enriched_details)} "
                f"verified={low_conf} saved={max(0, n_estimated + low_conf - 1)} calls "
                f"ran={gemini_ran}",
            )
        ],
    }
