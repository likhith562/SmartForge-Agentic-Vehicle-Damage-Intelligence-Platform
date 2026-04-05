"""
tests/test_graph_state.py
==========================
Tests for the LangGraph state schema and graph routing.

Covers
------
- SmartForgeState   — TypedDict field presence and Annotated reducer behaviour
- make_initial_state — correct defaults, job_id format, multi-image paths
- log_msg            — message dict structure
- human_audit_node   — terminal node output shape + is_fraud flag
- Graph routing      — fraud_router correctly routes SUSPICIOUS → human_audit
                       (integration: imports compiled graph, does NOT stream it)
- health_monitor_router — PASS, FAIL-with-retries, circuit-breaker paths

No GPU, no real API keys, no MongoDB required.
"""

import operator
import os
import tempfile
import unittest
from datetime import datetime, timezone
from typing import get_type_hints
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# 1. SmartForgeState schema
# ─────────────────────────────────────────────────────────────────────────────

class TestSmartForgeState(unittest.TestCase):
    """Validate the TypedDict schema and Annotated reducer fields."""

    def setUp(self):
        from src.graph.state import SmartForgeState, make_initial_state
        self.State          = SmartForgeState
        self.make_state     = make_initial_state

    def test_all_required_top_level_keys_present(self):
        state = self.make_state("/tmp/fake.jpg")
        required = [
            "messages", "image_path", "raw_detections", "final_output",
            "vehicle_type", "health_score", "validation_passed",
            "retry_count", "fraud_report", "is_fraud", "fraud_attempts",
            "job_id", "vehicle_id", "policy_id", "pipeline_trace",
            "started_at", "financial_estimate", "total_loss_flag",
            "verified_damages", "all_raw_detections", "fused_detections",
            "image_paths", "adaptive_sahi_conf", "scene_type",
        ]
        for key in required:
            self.assertIn(key, state, f"Missing required key: {key}")

    def test_messages_is_list_with_one_entry(self):
        state = self.make_state("/tmp/fake.jpg")
        self.assertIsInstance(state["messages"], list)
        self.assertEqual(len(state["messages"]), 1)
        self.assertIn("role",      state["messages"][0])
        self.assertIn("content",   state["messages"][0])
        self.assertIn("timestamp", state["messages"][0])

    def test_job_id_contains_vehicle_id(self):
        state = self.make_state("/tmp/x.jpg", vehicle_id="VH042")
        self.assertIn("VH042", state["job_id"])

    def test_job_id_is_unique_across_calls(self):
        s1 = self.make_state("/tmp/x.jpg", vehicle_id="VH001")
        s2 = self.make_state("/tmp/x.jpg", vehicle_id="VH001")
        self.assertNotEqual(s1["job_id"], s2["job_id"])

    def test_image_paths_defaults_to_single_element_list(self):
        state = self.make_state("/tmp/car.jpg")
        self.assertEqual(state["image_paths"], ["/tmp/car.jpg"])

    def test_image_paths_accepts_multi_image_list(self):
        paths = ["/tmp/a.jpg", "/tmp/b.jpg", "/tmp/c.jpg"]
        state = self.make_state("/tmp/a.jpg", image_paths=paths)
        self.assertEqual(state["image_paths"], paths)

    def test_default_numeric_fields(self):
        state = self.make_state("/tmp/x.jpg")
        self.assertEqual(state["retry_count"],       0)
        self.assertEqual(state["fraud_attempts"],    0)
        self.assertAlmostEqual(state["health_score"], 1.0)
        self.assertFalse(state["validation_passed"])
        self.assertFalse(state["is_fraud"])
        self.assertFalse(state["total_loss_flag"])

    def test_annotated_reducer_fields_start_empty(self):
        state = self.make_state("/tmp/x.jpg")
        self.assertEqual(state["all_raw_detections"], [])
        self.assertEqual(state["fused_detections"],   [])
        self.assertEqual(state["verified_damages"],   [])

    def test_pipeline_trace_starts_empty_dict(self):
        state = self.make_state("/tmp/x.jpg")
        self.assertIsInstance(state["pipeline_trace"], dict)
        self.assertEqual(len(state["pipeline_trace"]), 0)

    def test_claim_coords_default_to_zero(self):
        state = self.make_state("/tmp/x.jpg")
        self.assertEqual(state["claim_lat"], 0.0)
        self.assertEqual(state["claim_lon"], 0.0)

    def test_claim_coords_passed_through(self):
        state = self.make_state("/tmp/x.jpg", claim_lat=13.08, claim_lon=80.27)
        self.assertAlmostEqual(state["claim_lat"], 13.08)
        self.assertAlmostEqual(state["claim_lon"], 80.27)

    def test_started_at_is_iso_string(self):
        state = self.make_state("/tmp/x.jpg")
        self.assertIsInstance(state["started_at"], str)
        # Should be parseable as ISO datetime
        dt = datetime.fromisoformat(state["started_at"].rstrip("Z"))
        self.assertIsNotNone(dt)

    def test_pipeline_stability_flag_default(self):
        state = self.make_state("/tmp/x.jpg")
        self.assertEqual(state["pipeline_stability_flag"], "Stable")


# ─────────────────────────────────────────────────────────────────────────────
# 2. log_msg helper
# ─────────────────────────────────────────────────────────────────────────────

class TestLogMsg(unittest.TestCase):

    def setUp(self):
        from src.graph.state import log_msg
        self.log_msg = log_msg

    def test_returns_dict_with_required_keys(self):
        msg = self.log_msg("test_agent", "hello world")
        self.assertIn("role",      msg)
        self.assertIn("content",   msg)
        self.assertIn("timestamp", msg)

    def test_role_and_content_match_args(self):
        msg = self.log_msg("my_agent", "some content")
        self.assertEqual(msg["role"],    "my_agent")
        self.assertEqual(msg["content"], "some content")

    def test_timestamp_is_recent_iso_string(self):
        before = datetime.now(timezone.utc)
        msg    = self.log_msg("agent", "msg")
        after  = datetime.now(timezone.utc)
        ts     = datetime.fromisoformat(msg["timestamp"])
        ts     = ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts
        self.assertGreaterEqual(ts, before)
        self.assertLessEqual(ts, after)


# ─────────────────────────────────────────────────────────────────────────────
# 3. human_audit_node
# ─────────────────────────────────────────────────────────────────────────────

class TestHumanAuditNode(unittest.TestCase):

    def _make_state(self, trust_score: int = 10, flags: list = None) -> dict:
        return {
            "job_id":     "VH001-TEST-20240101T000000",
            "vehicle_id": "VH001",
            "image_path": "/tmp/fake.jpg",
            "fraud_report": {
                "trust_score": trust_score,
                "status":      "SUSPICIOUS_HIGH_RISK",
                "flags":       flags or ["RECYCLED_IMAGE: pHash match"],
                "checks_run":  5,
                "checked_at":  datetime.now(timezone.utc).isoformat(),
            },
            "pipeline_trace": {},
            "messages":       [],
        }

    def test_sets_is_fraud_true(self):
        from src.graph.nodes.human_audit import human_audit_node
        state  = self._make_state()
        result = human_audit_node(state)
        self.assertTrue(result["is_fraud"])

    def test_final_output_has_required_keys(self):
        from src.graph.nodes.human_audit import human_audit_node
        state  = self._make_state()
        result = human_audit_node(state)
        fo     = result["final_output"]
        for key in ("status", "fraud_report", "message",
                    "claim_ruling_code", "job_id"):
            self.assertIn(key, fo, f"Missing key in final_output: {key}")

    def test_final_output_status_is_human_audit_required(self):
        from src.graph.nodes.human_audit import human_audit_node
        result = human_audit_node(self._make_state())
        self.assertEqual(result["final_output"]["status"], "HUMAN_AUDIT_REQUIRED")

    def test_ruling_code_is_clm_manual(self):
        from src.graph.nodes.human_audit import human_audit_node
        result = human_audit_node(self._make_state())
        self.assertEqual(result["final_output"]["claim_ruling_code"], "CLM_MANUAL")

    def test_messages_list_has_one_entry(self):
        from src.graph.nodes.human_audit import human_audit_node
        result = human_audit_node(self._make_state())
        self.assertIsInstance(result["messages"], list)
        self.assertEqual(len(result["messages"]), 1)

    def test_does_not_raise_on_empty_fraud_report(self):
        from src.graph.nodes.human_audit import human_audit_node
        state = self._make_state()
        state["fraud_report"] = {}
        try:
            human_audit_node(state)
        except Exception as exc:
            self.fail(f"human_audit_node raised on empty fraud_report: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Graph routing: SUSPICIOUS → human_audit (via fraud_router)
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphRouting(unittest.TestCase):
    """
    Validates that the compiled graph correctly routes fraud-flagged claims
    to human_audit without executing any CV or Gemini nodes.

    Strategy: patch fraud_node to return SUSPICIOUS_HIGH_RISK directly, then
    stream the graph and confirm final_output has HUMAN_AUDIT_REQUIRED status.
    This is a lightweight integration test — no image loading, no GPU.
    """

    def _make_initial_state(self, image_path: str) -> dict:
        from src.graph.state import make_initial_state
        return make_initial_state(image_path, vehicle_id="VH-TEST")

    def test_suspicious_fraud_report_routes_to_human_audit(self):
        """
        Mock intake_node to skip image loading, mock fraud_node to return
        SUSPICIOUS, then verify the graph terminates at human_audit_node.
        """
        from src.graph.workflow import graph

        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        # Write minimal valid JPEG header so OpenCV does not crash if intake runs
        from PIL import Image as PILImage
        import numpy as np
        PILImage.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(tmp.name)

        def _mock_intake(state):
            import cv2
            img = cv2.imread(tmp.name)
            return {
                "image_path":         tmp.name,
                "image_bgr":          img,
                "image_rgb":          cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                "adaptive_sahi_conf": 0.3,
                "scene_type":         "normal",
                "pipeline_trace":     {**state["pipeline_trace"], "intake_agent": {}},
                "messages":           [{"role": "intake_agent",
                                        "content": "mocked",
                                        "timestamp": datetime.now(timezone.utc).isoformat()}],
            }

        def _mock_fraud(state):
            return {
                "fraud_report": {
                    "trust_score": 0,
                    "status":      "SUSPICIOUS_HIGH_RISK",
                    "flags":       ["RECYCLED_IMAGE: pHash match (Hamming=0)"],
                    "checks_run":  5,
                    "checked_at":  datetime.now(timezone.utc).isoformat(),
                    "details":     {},
                },
                "is_fraud":       True,
                "pipeline_trace": {**state["pipeline_trace"], "fraud_agent": {}},
                "messages":       [{"role": "fraud_agent",
                                    "content": "SUSPICIOUS",
                                    "timestamp": datetime.now(timezone.utc).isoformat()}],
            }

        state  = self._make_initial_state(tmp.name)
        thread = {"configurable": {"thread_id": state["job_id"]}}

        try:
            with (
                patch("src.graph.nodes.intake.intake_node",   side_effect=_mock_intake),
                patch("src.graph.nodes.fraud.fraud_node",     side_effect=_mock_fraud),
            ):
                final = None
                for event in graph.stream(state, thread, stream_mode="values"):
                    final = event

            self.assertIsNotNone(final)
            fo = final.get("final_output") or {}
            self.assertEqual(fo.get("status"), "HUMAN_AUDIT_REQUIRED",
                f"Expected HUMAN_AUDIT_REQUIRED, got: {fo.get('status')}")
            self.assertTrue(final.get("is_fraud"),
                "is_fraud should be True after routing to human_audit")
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)


# ─────────────────────────────────────────────────────────────────────────────
# 5. health_monitor_router — all routing branches
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthMonitorRouter(unittest.TestCase):

    def setUp(self):
        from src.graph.nodes.health_monitor import health_monitor_router
        from src.config.settings import cfg
        self.router     = health_monitor_router
        self.max_retries = cfg.MAX_RETRIES

    def _state(self, passed: bool, retry: int) -> dict:
        return {
            "validation_passed": passed,
            "retry_count":       retry,
            "validation_errors": [] if passed else ["INVALID_AREA_RATIO: D001 area=0"],
        }

    def test_pass_routes_to_reasoning(self):
        route = self.router(self._state(passed=True, retry=0))
        self.assertEqual(route, "reasoning")

    def test_pass_after_retry_still_routes_to_reasoning(self):
        route = self.router(self._state(passed=True, retry=1))
        self.assertEqual(route, "reasoning")

    def test_fail_with_retries_remaining_routes_to_perception_retry(self):
        route = self.router(self._state(passed=False, retry=0))
        self.assertEqual(route, "perception_retry")

    def test_fail_at_max_retries_routes_to_reasoning_circuit_breaker(self):
        route = self.router(self._state(passed=False, retry=self.max_retries))
        self.assertEqual(route, "reasoning",
            "Circuit breaker should degrade to reasoning, not loop forever")

    def test_fail_just_below_max_retries_still_retries(self):
        if self.max_retries < 1:
            self.skipTest("MAX_RETRIES < 1 — circuit breaker fires immediately")
        route = self.router(self._state(passed=False, retry=self.max_retries - 1))
        self.assertEqual(route, "perception_retry")


# ─────────────────────────────────────────────────────────────────────────────
# 6. health_monitor_node — validation check logic
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthMonitorNode(unittest.TestCase):

    def setUp(self):
        from src.graph.nodes.health_monitor import health_monitor_node
        self.node = health_monitor_node

    def _base_state(self, detections=None):
        return {
            "raw_detections":  detections or [],
            "retry_count":     0,
            "pipeline_trace":  {},
            "messages":        [],
        }

    def _make_det(self, det_id="D001", conf=0.8, ar=0.05, rv=0.01, low_conf=False):
        return {
            "detection_id":               det_id,
            "confidence":                 conf,
            "area_ratio":                 ar,
            "relative_deformation_index": rv,
            "low_confidence_flag":        low_conf,
            "gemini_verified":            None,
            "source":                     "cv_model",
        }

    def test_valid_single_detection_passes(self):
        state  = self._base_state([self._make_det()])
        result = self.node(state)
        self.assertTrue(result["validation_passed"])
        self.assertAlmostEqual(result["health_score"], 1.0)
        self.assertEqual(result["pipeline_stability_flag"], "Stable")

    def test_invalid_area_ratio_zero_fails(self):
        state  = self._base_state([self._make_det(ar=0.0)])
        result = self.node(state)
        self.assertFalse(result["validation_passed"])
        errs = result["validation_errors"]
        self.assertTrue(any("INVALID_AREA_RATIO" in e for e in errs))

    def test_invalid_area_ratio_above_one_fails(self):
        state  = self._base_state([self._make_det(ar=1.5)])
        result = self.node(state)
        self.assertFalse(result["validation_passed"])

    def test_invalid_deformation_negative_fails(self):
        state  = self._base_state([self._make_det(rv=-0.001)])
        result = self.node(state)
        self.assertFalse(result["validation_passed"])
        errs = result["validation_errors"]
        self.assertTrue(any("INVALID_DEFORMATION" in e for e in errs))

    def test_high_confidence_variance_fails(self):
        dets = [
            self._make_det("D001", conf=0.95, ar=0.05),
            self._make_det("D002", conf=0.31, ar=0.05),
        ]
        state  = self._base_state(dets)
        result = self.node(state)
        # var([0.95, 0.31]) ≈ 0.1 > 0.08 → should fail
        errs = result["validation_errors"]
        self.assertTrue(any("HIGH_CONF_VARIANCE" in e for e in errs))

    def test_gemini_veto_preserved_on_update(self):
        det = {**self._make_det(), "gemini_verified": False}
        state  = self._base_state([det])
        result = self.node(state)
        updated = result["raw_detections"][0]
        self.assertEqual(updated["verification_status"], "unconfirmed",
            "gemini_verified=False should force unconfirmed regardless of health")

    def test_empty_detections_passes(self):
        state  = self._base_state([])
        result = self.node(state)
        self.assertTrue(result["validation_passed"])

    def test_circuit_breaker_flag_set_when_max_retries_hit(self):
        from src.config.settings import cfg
        state = {
            **self._base_state([self._make_det(ar=0.0)]),  # will fail
            "retry_count": cfg.MAX_RETRIES,
        }
        result = self.node(state)
        self.assertEqual(result["pipeline_stability_flag"], "CircuitBreaker")


if __name__ == "__main__":
    unittest.main()
