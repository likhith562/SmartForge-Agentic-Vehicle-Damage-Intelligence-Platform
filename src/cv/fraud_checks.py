"""
SmartForge — CV Fraud-Check Helpers
=====================================
Pure functions implementing all five fraud-detection checks.
None of these functions touch LangGraph state — they accept plain Python
values and return plain dicts.  The fraud_node in src/graph/nodes/fraud.py
orchestrates them and writes results into state.

Checks
------
1. Temporal Consistency   — EXIF DateTimeOriginal vs claim_date
2. GPS Consistency        — EXIF GPS vs claim location (Haversine)
3. Software Integrity     — EXIF Image Software tag (editing software flag)
4. pHash Duplicate        — perceptual hash vs local fraud DB + optional SerpAPI
5. Screen & AI Forensics  — FFT Moiré + ELA + Laplacian variance

Public API
----------
    haversine_km(lat1, lon1, lat2, lon2)         → float (km)
    parse_exif_gps(tags)                          → (lat, lon) | (None, None)
    parse_exif_datetime(tags)                     → datetime | None
    load_fraud_hash_db()                          → dict
    save_fraud_hash_db(db)
    compute_phash(img_path)                       → str | None
    check_phash_against_db(img_path)              → dict
    check_reverse_image_serpapi(img_path)         → dict
    detect_screen_capture(img_path)               → dict
    perform_ela_check(img_path)                   → float
    check_ai_generation_with_fallback(img_path)   → dict
"""

import json
import math
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from src.config.settings import cfg


# ─────────────────────────────────────────────────────────────────────────────
# Check 1 & 2 helpers — EXIF parsing
# ─────────────────────────────────────────────────────────────────────────────

def haversine_km(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """
    Return the great-circle distance in kilometres between two GPS coordinates
    using the Haversine formula.
    """
    R   = 6371.0
    φ1  = math.radians(lat1)
    φ2  = math.radians(lat2)
    dφ  = math.radians(lat2 - lat1)
    dλ  = math.radians(lon2 - lon1)
    a   = (
        math.sin(dφ / 2) ** 2
        + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def parse_exif_gps(tags: dict) -> Tuple[Optional[float], Optional[float]]:
    """
    Convert EXIF GPS tags to decimal degrees.

    Returns (lat, lon) on success, or (None, None) when GPS tags are absent
    or malformed.
    """
    try:
        def _dms_to_decimal(dms_tag, ref_tag) -> float:
            dms = dms_tag.values
            ref = str(ref_tag)
            d   = float(dms[0].num) / float(dms[0].den)
            m   = float(dms[1].num) / float(dms[1].den)
            s   = float(dms[2].num) / float(dms[2].den)
            dec = d + m / 60 + s / 3600
            if ref in ("S", "W"):
                dec = -dec
            return dec

        lat = _dms_to_decimal(tags["GPS GPSLatitude"],  tags["GPS GPSLatitudeRef"])
        lon = _dms_to_decimal(tags["GPS GPSLongitude"], tags["GPS GPSLongitudeRef"])
        return lat, lon
    except Exception:
        return None, None


def parse_exif_datetime(tags: dict) -> Optional[datetime]:
    """
    Parse EXIF DateTimeOriginal or Image DateTime into a Python datetime.
    Returns None when no datetime tag is present or parsing fails.
    """
    for key in ("EXIF DateTimeOriginal", "Image DateTime"):
        tag = tags.get(key)
        if tag:
            try:
                return datetime.strptime(str(tag), "%Y:%m:%d %H:%M:%S")
            except ValueError:
                pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Check 4 helpers — pHash + Reverse Image Search
# ─────────────────────────────────────────────────────────────────────────────

def load_fraud_hash_db() -> Dict[str, str]:
    """
    Load the local perceptual-hash fraud database from disk.
    Returns an empty dict if the file does not exist or is unreadable.
    """
    path = cfg.FRAUD_HASH_DB_PATH
    if os.path.exists(path):
        try:
            with open(path, "r") as fh:
                return json.load(fh)
        except Exception:
            pass
    return {}


def save_fraud_hash_db(db: Dict[str, str]) -> None:
    """Persist the fraud hash database dict to disk."""
    with open(cfg.FRAUD_HASH_DB_PATH, "w") as fh:
        json.dump(db, fh, indent=2)


def compute_phash(img_path: str) -> Optional[str]:
    """
    Compute the perceptual hash (pHash) of an image file.
    Returns a hex string, or None on error.
    """
    try:
        import imagehash
        from PIL import Image as PILImage
        pil_img = PILImage.open(img_path).convert("RGB")
        return str(imagehash.phash(pil_img))
    except Exception as exc:
        print(f"   [phash] Error: {exc}")
        return None


def check_phash_against_db(img_path: str) -> Dict[str, Any]:
    """
    Core reverse-image check using perceptual hashing.

    Algorithm
    ---------
    1. Compute pHash of the submitted image.
    2. Compare against every entry in the local fraud DB using Hamming distance.
    3. If any entry has Hamming distance ≤ PHASH_HAMMING_THRESHOLD → DUPLICATE.
    4. Otherwise enrol the new hash into the DB for future cross-claim checks.

    Returns a dict with keys:
        status            "DUPLICATE_DETECTED" | "UNIQUE" | "HASH_ERROR"
        phash             str | None
        matched_claim     str  (path of prior claim, present on duplicate)
        hamming_distance  int | None
    """
    import imagehash

    ph = compute_phash(img_path)
    if ph is None:
        return {"status": "HASH_ERROR", "phash": None}

    db         = load_fraud_hash_db()
    best_match = None
    best_dist  = 999

    for stored_path, stored_hash_str in db.items():
        try:
            dist = imagehash.hex_to_hash(ph) - imagehash.hex_to_hash(stored_hash_str)
            if dist < best_dist:
                best_dist  = dist
                best_match = stored_path
        except Exception:
            continue

    if best_dist <= cfg.PHASH_HAMMING_THRESHOLD:
        return {
            "status":           "DUPLICATE_DETECTED",
            "phash":            ph,
            "matched_claim":    best_match,
            "hamming_distance": best_dist,
        }

    # Unique — enrol in DB for future checks
    db[img_path] = ph
    save_fraud_hash_db(db)
    return {"status": "UNIQUE", "phash": ph, "hamming_distance": None}


def check_reverse_image_serpapi(img_path: str) -> Dict[str, Any]:
    """
    Optional: Google Reverse Image Search via SerpAPI (Google Lens endpoint).

    Requires SERPAPI_KEY in environment / .env.
    Free tier: 100 searches/month at https://serpapi.com

    Returns dict with keys:
        found_online  bool | None  — True if an internet match was found
        match_url     str          — URL of the matching page
        match_title   str          — title of the matching page
        match_count   int          — total visual matches found
        reason        str          — present on skip/error
    """
    if not cfg.SERPAPI_ENABLED:
        return {"found_online": None, "reason": "SERPAPI_NOT_CONFIGURED"}

    try:
        import base64
        import requests

        with open(img_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()

        resp = requests.get(
            "https://serpapi.com/search",
            params={
                "engine":  "google_lens",
                "api_key": cfg.SERPAPI_KEY,
                "url":     f"data:image/jpeg;base64,{b64[:500]}",
            },
            timeout=10,
        )

        if resp.status_code == 401:
            msg = "[SERPAPI_ERROR]: 401 Unauthorized — check your SERPAPI_KEY"
            print(f"      ⚠️  {msg}")
            return {"found_online": None, "reason": msg}
        if resp.status_code == 429:
            msg = "[SERPAPI_ERROR]: 429 Rate Limit — monthly quota exceeded"
            print(f"      ⚠️  {msg}")
            return {"found_online": None, "reason": msg}

        data    = resp.json()
        matches = data.get("visual_matches", [])
        if matches:
            top = matches[0]
            return {
                "found_online": True,
                "match_url":    top.get("link",  ""),
                "match_title":  top.get("title", ""),
                "match_count":  len(matches),
            }
        return {"found_online": False}

    except Exception as exc:
        msg = f"[SERPAPI_ERROR]: {str(exc)[:80]}"
        print(f"      ⚠️  {msg}")
        return {"found_online": None, "reason": msg}


# ─────────────────────────────────────────────────────────────────────────────
# Check 5a — Screen / Display Detection (FFT Moiré + colour banding)
# ─────────────────────────────────────────────────────────────────────────────

def detect_screen_capture(img_path: str) -> Dict[str, Any]:
    """
    Detect whether a photo was taken OF A SCREEN rather than a real vehicle.

    A common fraud vector: claimants photograph a stock/internet car image
    displayed on a phone or monitor to defeat basic reverse-image detection.

    Two-signal approach
    -------------------
    Signal A — FFT Moiré Detection
        Real-world photos have smooth frequency spectra.  Screen photos
        exhibit a periodic pixel-grid pattern creating strong peaks in the
        FFT magnitude at mid-frequencies (Moiré effect).
        Threshold: mid-frequency energy ratio > 0.38.

    Signal B — Colour Banding / Gamma Anomaly
        Screens have a limited colour gamut and add gamma banding visible
        as regular empty bins in the R-channel histogram.
        Threshold: > 30 empty bins in the non-tail region (20–235).

    Returns dict with keys:
        is_screen   bool
        confidence  float  (0.0 – 1.0; ≥ 0.5 → is_screen = True)
        signals     list[str]
    """
    import cv2
    import numpy as np

    result: Dict[str, Any] = {
        "is_screen":  False,
        "confidence": 0.0,
        "signals":    [],
    }

    try:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return result

        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w    = gray.shape

        # ── Signal A: FFT Moiré ───────────────────────────────────────────────
        fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
        cy, cx  = h // 2, w // 2
        max_r   = np.sqrt(cy ** 2 + cx ** 2)
        r_lo    = 0.10 * max_r
        r_hi    = 0.40 * max_r
        ys, xs  = np.ogrid[-cy:h - cy, -cx:w - cx]
        radii   = np.sqrt(xs ** 2 + ys ** 2)
        mid_mask    = (radii >= r_lo) & (radii <= r_hi)
        mid_energy  = float(np.sum(fft_mag[mid_mask]))
        total_energy = float(np.sum(fft_mag)) + 1e-9
        moire_ratio = mid_energy / total_energy

        MOIRE_THRESHOLD = 0.38
        if moire_ratio > MOIRE_THRESHOLD:
            result["signals"].append(
                f"FFT_MOIRE: mid-freq energy ratio={moire_ratio:.3f} > {MOIRE_THRESHOLD}"
            )
            result["confidence"] += 0.5

        # ── Signal B: Colour banding in R channel ─────────────────────────────
        r_channel   = img_bgr[:, :, 2].ravel()
        hist, _     = np.histogram(r_channel, bins=256, range=(0, 256))
        inner       = hist[20:235]
        zero_bins   = int(np.sum(inner == 0))
        COMB_THRESHOLD = 30
        if zero_bins > COMB_THRESHOLD:
            result["signals"].append(
                f"COLOUR_BANDING: {zero_bins} empty R-histogram bins > {COMB_THRESHOLD}"
            )
            result["confidence"] += 0.4

        result["is_screen"] = result["confidence"] >= 0.5

    except Exception as exc:
        result["signals"].append(f"SCREEN_DETECT_ERROR: {exc}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Check 5b — Error Level Analysis (ELA)
# ─────────────────────────────────────────────────────────────────────────────

def perform_ela_check(img_path: str) -> float:
    """
    Error Level Analysis — detects pixel-level editing artefacts.

    Concept
    -------
    Re-saving a JPEG at a known quality level produces consistent compression
    artefacts.  Regions that were digitally edited (e.g. a scratch painted in
    Photoshop) have a DIFFERENT error level from the surrounding untouched
    areas.

    Method
    ------
    1. Re-save the image at 90 % JPEG quality to a temp file.
    2. Compute per-pixel absolute difference (original − resaved).
    3. Return the mean absolute difference across all channels.

    Interpretation
    --------------
    ela_score < 2.0   → consistent, likely unedited
    ela_score 2–5.0   → normal compression artefacts
    ela_score > 5.0   → high inconsistency → likely edited / spliced

    Returns
    -------
    float — mean absolute diff; 0.0 on error
    """
    import cv2
    import numpy as np

    try:
        original = cv2.imread(img_path)
        if original is None:
            return 0.0

        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        cv2.imwrite(tmp.name, original, [cv2.IMWRITE_JPEG_QUALITY, 90])
        resaved = cv2.imread(tmp.name)
        os.unlink(tmp.name)

        if resaved is None or resaved.shape != original.shape:
            return 0.0

        diff = cv2.absdiff(
            original.astype(np.float32),
            resaved.astype(np.float32),
        )
        return float(np.mean(diff))

    except Exception as exc:
        print(f"   [ela] Error: {exc}")
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Check 5b — AI-Generation detection (3-stage unbreakable pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def check_ai_generation_with_fallback(img_path: str) -> Dict[str, Any]:
    """
    3-Stage Unbreakable AI-Generation & Forensic Check.

    Stage 1 — Winston AI API (cloud, best accuracy)
        Requires WINSTON_AI_KEY.  Free tier: 2000 credits/month.
        Returns a 0–1 AI probability score.
        Automatically skipped when key is absent or quota exceeded.

    Stage 2 — ELA: Error Level Analysis (local, no key needed)
        Detects pixel-by-pixel editing artefacts (Photoshop / GIMP splices).
        Triggered when Stage 1 is unavailable.
        ela_score > 5.0 → treated same as AI-generated.

    Stage 3 — Laplacian Variance: Smoothness Check (local, fastest)
        AI-generated images (DALL-E, Midjourney, SD) lack camera sensor noise.
        Real damaged-car photos: lap_var > 300.
        AI images: typically < 100.
        Used as the final fallback when Stages 1 & 2 both fail.

    Returns dict with keys:
        is_ai_generated   bool | None
        ai_probability    float | None
        method            str  — which stage produced the verdict
        stages_run        list[str]
        ela_score         float  (Stage 2 only)
        laplacian_variance float (Stage 3 only)
        reasoning         str
    """
    import cv2
    import requests

    stages_run: list = []
    base_result: Dict[str, Any] = {
        "is_ai_generated": None,
        "ai_probability":  None,
        "method":          "none",
        "stages_run":      stages_run,
    }

    # ── Stage 1: Winston AI API ───────────────────────────────────────────────
    if cfg.WINSTON_AI_ENABLED:
        stages_run.append("winston_ai_api")
        try:
            with open(img_path, "rb") as fh:
                resp = requests.post(
                    "https://api.gowinston.ai/v2/image",
                    headers={"Authorization": f"Bearer {cfg.WINSTON_AI_KEY}"},
                    files={"file": (
                        os.path.basename(img_path), fh, "image/jpeg"
                    )},
                    timeout=15,
                )

            data = resp.json()
            if resp.status_code == 200 and (
                "score" in data or "ai_probability" in data
            ):
                ai_prob = float(
                    data.get("score", data.get("ai_probability", 0.0))
                )
                return {
                    "is_ai_generated": ai_prob >= cfg.WINSTON_AI_THRESHOLD,
                    "ai_probability":  round(ai_prob, 3),
                    "method":          "winston_ai_api",
                    "stages_run":      stages_run,
                    "raw_response":    {
                        k: v for k, v in data.items() if k != "image_data"
                    },
                }
            else:
                print(
                    f"   [winston_ai] Unexpected response {resp.status_code}"
                    " — switching to ELA"
                )
        except Exception as exc:
            quota_hit = any(
                k in str(exc).lower()
                for k in ("quota", "429", "limit", "billing")
            )
            print(
                f"   [winston_ai] "
                f"{'Quota exceeded' if quota_hit else 'API error'}: "
                f"{str(exc)[:80]}"
            )
            print("   ➡️  Switching to Stage 2: ELA forensics")

    # ── Stage 2: Error Level Analysis ────────────────────────────────────────
    stages_run.append("ela_forensics")
    try:
        ela_score = perform_ela_check(img_path)
        ELA_THRESHOLD = 5.0

        if ela_score > ELA_THRESHOLD:
            return {
                "is_ai_generated": True,
                "ai_probability":  round(min(1.0, ela_score / 20.0), 3),
                "ela_score":       round(ela_score, 3),
                "method":          "ela_forensics",
                "stages_run":      stages_run,
                "reasoning": (
                    f"ELA score {ela_score:.2f} > {ELA_THRESHOLD} "
                    "— high compression inconsistency (editing artefacts detected)"
                ),
            }
        elif ela_score > 0:
            return {
                "is_ai_generated": False,
                "ai_probability":  round(ela_score / 20.0, 3),
                "ela_score":       round(ela_score, 3),
                "method":          "ela_forensics",
                "stages_run":      stages_run,
                "reasoning": (
                    f"ELA score {ela_score:.2f} ≤ {ELA_THRESHOLD} "
                    "— consistent compression (no editing artefacts)"
                ),
            }
    except Exception as exc:
        print(f"   [ela] Stage 2 error: {exc} — switching to Laplacian fallback")

    # ── Stage 3: Laplacian Variance (final fallback) ──────────────────────────
    stages_run.append("laplacian_variance")
    try:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return {**base_result, "method": "all_stages_failed"}

        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        AI_SMOOTH_THRESHOLD = 100.0
        ai_likely = lap_var < AI_SMOOTH_THRESHOLD
        ai_prob   = round(max(0.0, 1.0 - lap_var / 800.0), 3)

        return {
            "is_ai_generated":    ai_likely,
            "ai_probability":     ai_prob,
            "laplacian_variance": round(lap_var, 2),
            "method":             "laplacian_variance",
            "stages_run":         stages_run,
            "reasoning": (
                f"Laplacian variance={lap_var:.1f} "
                f"({'below' if ai_likely else 'above'} threshold "
                f"{AI_SMOOTH_THRESHOLD}) → "
                f"{'hyper-smooth (AI-like)' if ai_likely else 'natural texture (real photo)'}"
            ),
        }
    except Exception as exc:
        return {**base_result, "method": f"all_stages_failed: {exc}"}
