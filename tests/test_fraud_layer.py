"""
tests/test_fraud_layer.py
==========================
Unit and integration tests for the Batch 1 Fraud & Integrity Layer.

Covers
------
- haversine_km          — distance calculation accuracy
- parse_exif_datetime   — happy path + missing tag
- parse_exif_gps        — happy path + missing tag
- perform_ela_check     — real JPEG round-trip produces a stable score
- detect_screen_capture — FFT Moiré signal on a synthetic grid image
- check_phash_against_db — enrol → UNIQUE; re-enrol → DUPLICATE_DETECTED
- check_ai_generation_with_fallback — Laplacian stage on real/synthetic image
- fraud_node            — BYPASS path; full scan with a recycled-image fixture
- fraud_router          — routing logic for all three outcomes

All tests use only stdlib + numpy + Pillow for image fixtures.
No GPU, no real API keys, no live MongoDB required.
"""

import json
import math
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image as PILImage


# ── Helpers to create synthetic test images ───────────────────────────────────

def _make_jpeg(path: str, width: int = 320, height: int = 240,
               mode: str = "solid") -> str:
    """
    Write a synthetic JPEG to *path* and return the path.

    mode
    ----
    "solid"  — uniform grey (low ELA score, low Laplacian variance)
    "noise"  — random pixel noise (high Laplacian variance — real-photo-like)
    "grid"   — periodic black/white grid (high FFT Moiré signal — screen-like)
    """
    if mode == "solid":
        arr = np.full((height, width, 3), 128, dtype=np.uint8)
    elif mode == "noise":
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    elif mode == "grid":
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[::4, :] = 255   # horizontal white stripes every 4 px
        arr[:, ::4] = 255   # vertical white stripes every 4 px
    else:
        raise ValueError(f"Unknown mode: {mode}")

    PILImage.fromarray(arr).save(path, format="JPEG", quality=95)
    return path


def _make_jpeg_tmp(mode: str = "solid") -> str:
    """Create a named temp JPEG and return its path (caller must delete)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    return _make_jpeg(tmp.name, mode=mode)


# ─────────────────────────────────────────────────────────────────────────────
# 1. haversine_km
# ─────────────────────────────────────────────────────────────────────────────

class TestHaversine(unittest.TestCase):
    """haversine_km — pure math, no dependencies."""

    def setUp(self):
        from src.cv.fraud_checks import haversine_km
        self.haversine_km = haversine_km

    def test_same_point_is_zero(self):
        self.assertAlmostEqual(self.haversine_km(13.08, 80.27, 13.08, 80.27), 0.0, places=3)

    def test_known_distance_chennai_bangalore(self):
        # Chennai (13.08, 80.27) to Bengaluru (12.97, 77.59) ≈ 290 km
        dist = self.haversine_km(13.08, 80.27, 12.97, 77.59)
        self.assertGreater(dist, 250)
        self.assertLess(dist, 320)

    def test_north_pole_to_south_pole(self):
        # Half circumference ≈ 20015 km
        dist = self.haversine_km(90, 0, -90, 0)
        self.assertAlmostEqual(dist, 20015, delta=10)

    def test_small_displacement(self):
        # 1 degree latitude ≈ 111 km
        dist = self.haversine_km(0, 0, 1, 0)
        self.assertAlmostEqual(dist, 111, delta=2)

    def test_negative_coordinates(self):
        dist = self.haversine_km(-33.87, 151.21, -37.81, 144.96)  # Sydney→Melbourne
        self.assertGreater(dist, 700)
        self.assertLess(dist, 800)


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXIF parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestExifParsing(unittest.TestCase):

    def setUp(self):
        from src.cv.fraud_checks import parse_exif_datetime, parse_exif_gps
        self.parse_dt  = parse_exif_datetime
        self.parse_gps = parse_exif_gps

    # ── datetime ──────────────────────────────────────────────────────────────

    def test_datetime_from_exif_original(self):
        tag = MagicMock()
        tag.__str__ = lambda s: "2024:03:15 10:30:00"
        tags = {"EXIF DateTimeOriginal": tag}
        dt = self.parse_dt(tags)
        self.assertIsNotNone(dt)
        self.assertEqual(dt.year,   2024)
        self.assertEqual(dt.month,  3)
        self.assertEqual(dt.day,    15)
        self.assertEqual(dt.hour,   10)

    def test_datetime_falls_back_to_image_datetime(self):
        tag = MagicMock()
        tag.__str__ = lambda s: "2023:01:01 08:00:00"
        tags = {"Image DateTime": tag}
        dt = self.parse_dt(tags)
        self.assertIsNotNone(dt)
        self.assertEqual(dt.year, 2023)

    def test_datetime_returns_none_when_missing(self):
        self.assertIsNone(self.parse_dt({}))

    def test_datetime_returns_none_on_bad_format(self):
        tag = MagicMock()
        tag.__str__ = lambda s: "not-a-date"
        self.assertIsNone(self.parse_dt({"EXIF DateTimeOriginal": tag}))

    # ── GPS ───────────────────────────────────────────────────────────────────

    def _make_dms_tag(self, degrees, minutes, seconds):
        """Build a mock EXIF DMS tag that parse_exif_gps can consume."""
        def _ratio(n):
            r = MagicMock()
            r.num = int(n * 1000)
            r.den = 1000
            return r
        tag = MagicMock()
        tag.values = [_ratio(degrees), _ratio(minutes), _ratio(seconds)]
        return tag

    def test_gps_north_east(self):
        lat_tag = self._make_dms_tag(13, 5, 0)     # 13° 5' 0" N ≈ 13.083
        lon_tag = self._make_dms_tag(80, 16, 0)    # 80° 16' 0" E ≈ 80.267
        ref_n   = MagicMock(); ref_n.__str__ = lambda s: "N"
        ref_e   = MagicMock(); ref_e.__str__ = lambda s: "E"
        tags = {
            "GPS GPSLatitude":     lat_tag,
            "GPS GPSLatitudeRef":  ref_n,
            "GPS GPSLongitude":    lon_tag,
            "GPS GPSLongitudeRef": ref_e,
        }
        lat, lon = self.parse_gps(tags)
        self.assertAlmostEqual(lat, 13.083, delta=0.01)
        self.assertAlmostEqual(lon, 80.267, delta=0.01)

    def test_gps_south_west_is_negative(self):
        lat_tag = self._make_dms_tag(33, 52, 0)
        lon_tag = self._make_dms_tag(151, 12, 0)
        ref_s   = MagicMock(); ref_s.__str__ = lambda s: "S"
        ref_w   = MagicMock(); ref_w.__str__ = lambda s: "W"
        tags = {
            "GPS GPSLatitude":     lat_tag,
            "GPS GPSLatitudeRef":  ref_s,
            "GPS GPSLongitude":    lon_tag,
            "GPS GPSLongitudeRef": ref_w,
        }
        lat, lon = self.parse_gps(tags)
        self.assertLess(lat, 0)
        self.assertLess(lon, 0)

    def test_gps_returns_none_none_when_missing(self):
        lat, lon = self.parse_gps({})
        self.assertIsNone(lat)
        self.assertIsNone(lon)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ELA (Error Level Analysis)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestELA(unittest.TestCase):

    def setUp(self):
        from src.cv.fraud_checks import perform_ela_check
        self.ela = perform_ela_check

    def tearDown(self):
        if hasattr(self, "_tmp") and os.path.exists(self._tmp):
            os.unlink(self._tmp)

    def test_solid_image_has_low_ela(self):
        self._tmp = _make_jpeg_tmp("solid")
        score = self.ela(self._tmp)
        # Solid colour → very consistent compression → low ELA
        self.assertGreaterEqual(score, 0.0)
        self.assertLess(score, 5.0)

    def test_noise_image_has_positive_ela(self):
        self._tmp = _make_jpeg_tmp("noise")
        score = self.ela(self._tmp)
        self.assertGreater(score, 0.0)

    def test_missing_file_returns_zero(self):
        score = self.ela("/nonexistent/path/image.jpg")
        self.assertEqual(score, 0.0)

    def test_returns_float(self):
        self._tmp = _make_jpeg_tmp("solid")
        score = self.ela(self._tmp)
        self.assertIsInstance(score, float)


# ─────────────────────────────────────────────────────────────────────────────
# 4. FFT Moiré / screen capture detection
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestScreenCapture(unittest.TestCase):

    def setUp(self):
        from src.cv.fraud_checks import detect_screen_capture
        self.detect = detect_screen_capture

    def tearDown(self):
        if hasattr(self, "_tmp") and os.path.exists(self._tmp):
            os.unlink(self._tmp)

    def test_solid_image_not_flagged_as_screen(self):
        self._tmp = _make_jpeg_tmp("solid")
        result = self.detect(self._tmp)
        self.assertIn("is_screen",  result)
        self.assertIn("confidence", result)
        self.assertIn("signals",    result)
        # Solid grey has no Moiré and no colour banding
        self.assertFalse(result["is_screen"])
        self.assertEqual(result["confidence"], 0.0)

    def test_grid_image_triggers_moire(self):
        self._tmp = _make_jpeg_tmp("grid")
        result = self.detect(self._tmp)
        # A pixel grid should push mid-frequency energy above the Moiré threshold
        moire_signals = [s for s in result["signals"] if "MOIRE" in s or "FFT" in s]
        self.assertGreater(len(moire_signals), 0,
            "Expected FFT_MOIRE signal on a periodic grid image")

    def test_missing_file_returns_safe_default(self):
        result = self.detect("/nonexistent/image.jpg")
        self.assertFalse(result["is_screen"])
        self.assertEqual(result["confidence"], 0.0)

    def test_result_keys_always_present(self):
        self._tmp = _make_jpeg_tmp("noise")
        result = self.detect(self._tmp)
        for key in ("is_screen", "confidence", "signals"):
            self.assertIn(key, result)


# ─────────────────────────────────────────────────────────────────────────────
# 5. pHash duplicate detection
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestPHash(unittest.TestCase):

    def setUp(self):
        from src.cv.fraud_checks import (
            check_phash_against_db,
            compute_phash,
            load_fraud_hash_db,
            save_fraud_hash_db,
        )
        self.check  = check_phash_against_db
        self.phash  = compute_phash
        self.load   = load_fraud_hash_db
        self.save   = save_fraud_hash_db

        # Redirect the fraud DB to a temp file for isolation
        self._db_tmp  = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._db_tmp.close()
        self._img_tmp = _make_jpeg_tmp("solid")

        # Patch cfg.FRAUD_HASH_DB_PATH
        import src.cv.fraud_checks as _fc
        self._orig_path = _fc.cfg.FRAUD_HASH_DB_PATH
        _fc.cfg.FRAUD_HASH_DB_PATH = self._db_tmp.name

    def tearDown(self):
        import src.cv.fraud_checks as _fc
        _fc.cfg.FRAUD_HASH_DB_PATH = self._orig_path
        for p in (self._db_tmp.name, self._img_tmp):
            if os.path.exists(p):
                os.unlink(p)

    def test_compute_phash_returns_hex_string(self):
        ph = self.phash(self._img_tmp)
        self.assertIsNotNone(ph)
        self.assertIsInstance(ph, str)
        self.assertGreater(len(ph), 0)

    def test_compute_phash_missing_file_returns_none(self):
        ph = self.phash("/nonexistent/image.jpg")
        self.assertIsNone(ph)

    def test_first_submission_is_unique(self):
        result = self.check(self._img_tmp)
        self.assertEqual(result["status"], "UNIQUE")
        self.assertIsNotNone(result["phash"])

    def test_identical_image_flagged_as_duplicate(self):
        # First submission enrolls the hash
        self.check(self._img_tmp)
        # Second submission of the same file → Hamming distance = 0
        result = self.check(self._img_tmp)
        self.assertEqual(result["status"], "DUPLICATE_DETECTED")
        self.assertEqual(result["hamming_distance"], 0)
        self.assertIsNotNone(result["matched_claim"])

    def test_slightly_different_image_not_flagged(self):
        # Enrol original
        self.check(self._img_tmp)
        # Create a visually different image
        noise_tmp = _make_jpeg_tmp("noise")
        try:
            result = self.check(noise_tmp)
            # Noise vs solid → large Hamming distance → UNIQUE
            self.assertEqual(result["status"], "UNIQUE",
                "Visually different images should not be flagged as duplicates")
        finally:
            os.unlink(noise_tmp)

    def test_db_persists_between_calls(self):
        self.check(self._img_tmp)
        db = self.load()
        self.assertIn(self._img_tmp, db)

    def test_hash_error_on_corrupt_file(self):
        corrupt = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        corrupt.write(b"this is not a jpeg")
        corrupt.close()
        try:
            ph = self.phash(corrupt.name)
            self.assertIsNone(ph)
        finally:
            os.unlink(corrupt.name)


# ─────────────────────────────────────────────────────────────────────────────
# 6. AI-generation detection (Laplacian stage only — no API key needed)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestAIGenerationCheck(unittest.TestCase):

    def setUp(self):
        from src.cv.fraud_checks import check_ai_generation_with_fallback
        self.check = check_ai_generation_with_fallback

    def tearDown(self):
        if hasattr(self, "_tmp") and os.path.exists(self._tmp):
            os.unlink(self._tmp)

    def test_result_always_has_required_keys(self):
        self._tmp = _make_jpeg_tmp("solid")
        result = self.check(self._tmp)
        for key in ("is_ai_generated", "ai_probability", "method", "stages_run"):
            self.assertIn(key, result)

    def test_stages_run_is_a_list(self):
        self._tmp = _make_jpeg_tmp("solid")
        result = self.check(self._tmp)
        self.assertIsInstance(result["stages_run"], list)
        self.assertGreater(len(result["stages_run"]), 0)

    def test_solid_grey_flagged_as_ai_like_by_laplacian(self):
        """A uniform solid image has near-zero Laplacian variance → AI-like."""
        self._tmp = _make_jpeg_tmp("solid")
        result = self.check(self._tmp)
        # With no Winston AI key, should fall through to ELA or Laplacian
        self.assertIn(result["method"], ("ela_forensics", "laplacian_variance"))
        if result["method"] == "laplacian_variance":
            # Solid images have very low variance → classified as AI-like
            self.assertTrue(result["is_ai_generated"],
                "Solid uniform image should be flagged as AI-like by Laplacian stage")

    def test_noise_image_not_ai_by_laplacian(self):
        """High-variance noise image should NOT be flagged by Laplacian."""
        self._tmp = _make_jpeg_tmp("noise")
        result = self.check(self._tmp)
        if result["method"] == "laplacian_variance":
            self.assertFalse(result["is_ai_generated"],
                "Random noise has high Laplacian variance — real-photo-like")

    def test_missing_file_does_not_raise(self):
        result = self.check("/nonexistent/image.jpg")
        # Should degrade to "all_stages_failed" without raising
        self.assertIn("method", result)


# ─────────────────────────────────────────────────────────────────────────────
# 7. fraud_node — BYPASS and full-scan paths
# ─────────────────────────────────────────────────────────────────────────────

class TestFraudNode(unittest.TestCase):
    """
    Tests for fraud_node and fraud_router.
    No GPU, no API keys.  The full-scan path uses real CV helpers on
    a synthetic JPEG (no EXIF metadata → most checks add 0 pts or warn).
    """

    def _make_state(self, image_path: str, **overrides) -> dict:
        """Minimal SmartForgeState-compatible dict for fraud_node."""
        base = {
            "image_path":     image_path,
            "image_paths":    [image_path],
            "claim_date":     "2024-06-01",
            "claim_lat":      13.08,
            "claim_lon":      80.27,
            "pipeline_trace": {},
            "messages":       [],
            "fraud_attempts": 0,
        }
        base.update(overrides)
        return base

    # ── BYPASS path ───────────────────────────────────────────────────────────

    def test_bypass_returns_verified_instantly(self):
        from src.cv.fraud_checks import load_fraud_hash_db
        from src.graph.nodes.fraud import fraud_node
        import src.graph.nodes.fraud as _fn

        _orig = _fn.cfg.BYPASS_FRAUD
        _fn.cfg.BYPASS_FRAUD = True
        try:
            tmp = _make_jpeg_tmp("solid")
            state  = self._make_state(tmp)
            result = fraud_node(state)
            fr = result["fraud_report"]
            self.assertEqual(fr["status"],      "VERIFIED")
            self.assertEqual(fr["trust_score"], 100)
            self.assertEqual(fr["checks_run"],  0)
            self.assertFalse(result["is_fraud"])
        finally:
            _fn.cfg.BYPASS_FRAUD = _orig
            os.unlink(tmp)

    def test_bypass_routes_single_image_to_perception(self):
        from src.graph.nodes.fraud import fraud_node, fraud_router
        import src.graph.nodes.fraud as _fn

        _orig = _fn.cfg.BYPASS_FRAUD
        _fn.cfg.BYPASS_FRAUD = True
        try:
            tmp    = _make_jpeg_tmp("solid")
            state  = self._make_state(tmp)
            result = fraud_node(state)
            merged = {**state, **result}
            route  = fraud_router(merged)
            self.assertEqual(route, "perception")
        finally:
            _fn.cfg.BYPASS_FRAUD = _orig
            os.unlink(tmp)

    def test_bypass_routes_multi_image_to_map_images(self):
        from src.graph.nodes.fraud import fraud_node, fraud_router
        import src.graph.nodes.fraud as _fn

        _orig = _fn.cfg.BYPASS_FRAUD
        _fn.cfg.BYPASS_FRAUD = True
        try:
            tmp1   = _make_jpeg_tmp("solid")
            tmp2   = _make_jpeg_tmp("noise")
            state  = self._make_state(tmp1, image_paths=[tmp1, tmp2])
            result = fraud_node(state)
            merged = {**state, **result}
            route  = fraud_router(merged)
            self.assertEqual(route, "map_images")
        finally:
            _fn.cfg.BYPASS_FRAUD = _orig
            for p in (tmp1, tmp2):
                if os.path.exists(p): os.unlink(p)

    # ── Full-scan path (no API keys — only local checks run) ──────────────────

    def test_full_scan_returns_fraud_report_dict(self):
        from src.graph.nodes.fraud import fraud_node
        import src.graph.nodes.fraud as _fn

        _orig = _fn.cfg.BYPASS_FRAUD
        _fn.cfg.BYPASS_FRAUD = False

        # Patch DB path to isolate pHash DB
        db_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        db_tmp.close()
        import src.cv.fraud_checks as _fc
        _orig_db = _fc.cfg.FRAUD_HASH_DB_PATH
        _fc.cfg.FRAUD_HASH_DB_PATH = db_tmp.name

        try:
            tmp   = _make_jpeg_tmp("solid")
            state = self._make_state(tmp)
            result = fraud_node(state)
            fr = result["fraud_report"]

            self.assertIn("trust_score",  fr)
            self.assertIn("status",       fr)
            self.assertIn("flags",        fr)
            self.assertIn("details",      fr)
            self.assertIn("checks_run",   fr)
            self.assertIsInstance(fr["trust_score"], int)
            self.assertIn(fr["status"], ("VERIFIED", "SUSPICIOUS_HIGH_RISK"))
        finally:
            _fn.cfg.BYPASS_FRAUD = _orig
            _fc.cfg.FRAUD_HASH_DB_PATH = _orig_db
            if os.path.exists(db_tmp.name): os.unlink(db_tmp.name)
            if os.path.exists(tmp): os.unlink(tmp)

    def test_recycled_image_flagged_as_suspicious(self):
        """Submitting the same image twice should lower trust score significantly."""
        from src.graph.nodes.fraud import fraud_node
        import src.graph.nodes.fraud as _fn

        _orig = _fn.cfg.BYPASS_FRAUD
        _fn.cfg.BYPASS_FRAUD = False

        db_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        db_tmp.close()
        import src.cv.fraud_checks as _fc
        _orig_db = _fc.cfg.FRAUD_HASH_DB_PATH
        _fc.cfg.FRAUD_HASH_DB_PATH = db_tmp.name

        try:
            tmp = _make_jpeg_tmp("solid")
            state = self._make_state(tmp)

            result1 = fraud_node(state)   # first submission — enrols hash
            score1  = result1["fraud_report"]["trust_score"]

            result2 = fraud_node(state)   # second submission — duplicate!
            score2  = result2["fraud_report"]["trust_score"]

            self.assertLess(score2, score1,
                "Recycled image should produce a lower trust score on second submission")

            flags2 = result2["fraud_report"]["flags"]
            recycled = any("RECYCLED" in f or "DUPLICATE" in f for f in flags2)
            self.assertTrue(recycled,
                "Expected RECYCLED_IMAGE or DUPLICATE_DETECTED flag on second submission")
        finally:
            _fn.cfg.BYPASS_FRAUD = _orig
            _fc.cfg.FRAUD_HASH_DB_PATH = _orig_db
            if os.path.exists(db_tmp.name): os.unlink(db_tmp.name)
            if os.path.exists(tmp): os.unlink(tmp)

    # ── fraud_router ──────────────────────────────────────────────────────────

    def test_router_suspicious_goes_to_human_audit(self):
        from src.graph.nodes.fraud import fraud_router
        state = {
            "fraud_report": {"status": "SUSPICIOUS_HIGH_RISK"},
            "image_path":   "/tmp/x.jpg",
            "image_paths":  ["/tmp/x.jpg"],
        }
        self.assertEqual(fraud_router(state), "human_audit")

    def test_router_verified_single_image_goes_to_perception(self):
        from src.graph.nodes.fraud import fraud_router
        state = {
            "fraud_report": {"status": "VERIFIED"},
            "image_path":   "/tmp/x.jpg",
            "image_paths":  ["/tmp/x.jpg"],
        }
        self.assertEqual(fraud_router(state), "perception")

    def test_router_verified_multi_image_goes_to_map_images(self):
        from src.graph.nodes.fraud import fraud_router
        state = {
            "fraud_report": {"status": "VERIFIED"},
            "image_path":   "/tmp/a.jpg",
            "image_paths":  ["/tmp/a.jpg", "/tmp/b.jpg"],
        }
        self.assertEqual(fraud_router(state), "map_images")

    def test_router_no_fraud_report_defaults_to_perception(self):
        from src.graph.nodes.fraud import fraud_router
        state = {
            "fraud_report": None,
            "image_path":   "/tmp/x.jpg",
            "image_paths":  ["/tmp/x.jpg"],
        }
        self.assertEqual(fraud_router(state), "perception")


if __name__ == "__main__":
    unittest.main()
