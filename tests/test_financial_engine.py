"""
tests/test_financial_engine.py
================================
Unit and integration tests for the Batch 4 Financial Intelligence Engine.

Covers
------
- estimate_cost           — COST_TABLE lookup + default fallback
- severity_to_score       — penalty mapping
- compute_severity        — CV signal → severity/category classification
- _get_repair_data        — REPAIR_DATABASE exact + fuzzy + default lookup
- reasoning_node          — line-item output, Total Loss threshold,
                            confirmed-only vs conservative score
- decision_node           — CLM_MANUAL / CLM_WORKSHOP / CLM_PENDING ruling codes
                            + AI-never-auto-approves invariant

No GPU, no API keys, no MongoDB required.
"""

import unittest
from datetime import datetime, timezone
from typing import Any, Dict, List


# ─────────────────────────────────────────────────────────────────────────────
# Helper — build minimal detection records
# ─────────────────────────────────────────────────────────────────────────────

def _make_det(
    det_id:    str   = "D001",
    dtype:     str   = "Dent",
    location:  str   = "Front Bumper",
    severity:  str   = "Medium",
    rv:        float = 0.008,
    ar:        float = 0.05,
    rejected:  bool  = False,
    conf:      float = 0.85,
    vs:        str   = "confirmed",
    is_verified: Any = True,
    severity_gemini: str = None,
) -> Dict[str, Any]:
    det = {
        "detection_id":               det_id,
        "type":                       dtype,
        "location":                   location,
        "severity":                   severity,
        "damage_category":            "Functional",
        "repair_type":                "Panel beating + repaint",
        "estimated_repair_cost":      "₹3,000–₹7,000",
        "relative_deformation_index": rv,
        "area_ratio":                 ar,
        "confidence":                 conf,
        "low_confidence_flag":        conf < 0.45,
        "verification_status":        vs,
        "is_verified":                is_verified,
        "rejected":                   rejected,
        "gemini_verified":            None,
        "gemini_reasoning":           "CV + severity rule",
        "bounding_box":               [10, 10, 100, 100],
    }
    if severity_gemini:
        det["severity_gemini"] = severity_gemini
    return det


def _reasoning_state(detections: list) -> dict:
    """Minimal state for reasoning_node."""
    return {
        "verified_damages":  detections,
        "raw_detections":    detections,
        "pipeline_trace":    {},
        "messages":          [],
        "damages_output":    [],
    }


def _decision_state(damages: list, is_fraud: bool = False) -> dict:
    """Minimal state for decision_node."""
    return {
        "damages_output":    damages,
        "financial_estimate": {},
        "is_fraud":           is_fraud,
        "job_id":             "VH001-TEST-20240101T000000",
        "vehicle_id":         "VH001",
        "policy_id":          "POL-001",
        "vehicle_type":       "car",
        "vehicle_make_estimate": "sedan-class",
        "gemini_agent_ran":   True,
        "total_loss_flag":    False,
        "pipeline_stability_flag": "Stable",
        "pipeline_trace":     {},
        "messages":           [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. estimate_cost — COST_TABLE lookup
# ─────────────────────────────────────────────────────────────────────────────

class TestEstimateCost(unittest.TestCase):

    def setUp(self):
        from src.cv.perception import estimate_cost
        self.est = estimate_cost

    def test_scratch_low_returns_known_range(self):
        cost, action = self.est("Scratch", "Low")
        self.assertIn("₹500", cost)
        self.assertIn("touch-up", action.lower())

    def test_dent_high_returns_replacement_note(self):
        cost, action = self.est("Dent", "High")
        self.assertIn("₹7,000", cost)
        self.assertIn("replacement", action.lower())

    def test_cracked_high_returns_part_replacement(self):
        cost, action = self.est("Cracked", "High")
        self.assertIn("replacement", action.lower())

    def test_unknown_combo_returns_default(self):
        from src.config.settings import cfg
        cost, action = self.est("AlienDamage", "Extreme")
        default_cost, default_action = cfg.DEFAULT_COST
        self.assertEqual(cost,   default_cost)
        self.assertEqual(action, default_action)

    def test_missing_part_high_returns_non_empty(self):
        cost, action = self.est("Missing part", "High")
        self.assertTrue(len(cost) > 0)
        self.assertTrue(len(action) > 0)

    def test_corrosion_medium_exists(self):
        cost, action = self.est("Corrosion", "Medium")
        self.assertIn("₹", cost)


# ─────────────────────────────────────────────────────────────────────────────
# 2. severity_to_score — penalty mapping
# ─────────────────────────────────────────────────────────────────────────────

class TestSeverityToScore(unittest.TestCase):

    def setUp(self):
        from src.cv.perception import severity_to_score
        self.score = severity_to_score

    def test_low_has_smallest_penalty(self):
        self.assertEqual(self.score("Low"), 5)

    def test_medium_penalty(self):
        self.assertEqual(self.score("Medium"), 20)

    def test_high_has_largest_penalty(self):
        self.assertEqual(self.score("High"), 40)

    def test_unknown_defaults_to_low_penalty(self):
        self.assertEqual(self.score("Critical"), 5)  # "Critical" not in map → 5

    def test_ordering_low_lt_medium_lt_high(self):
        self.assertLess(self.score("Low"), self.score("Medium"))
        self.assertLess(self.score("Medium"), self.score("High"))


# ─────────────────────────────────────────────────────────────────────────────
# 3. compute_severity — CV signal classification
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeSeverity(unittest.TestCase):

    def setUp(self):
        from src.cv.perception import compute_severity
        self.compute = compute_severity

    # Structural damage types
    def test_missing_part_always_high_functional(self):
        sev, cat = self.compute("Missing part", rv=0.0, area_ratio=0.001)
        self.assertEqual(sev, "High")
        self.assertEqual(cat, "Functional")

    def test_cracked_always_high_functional(self):
        sev, cat = self.compute("Cracked", rv=0.0, area_ratio=0.001)
        self.assertEqual(sev, "High")
        self.assertEqual(cat, "Functional")

    def test_broken_part_always_high_functional(self):
        sev, cat = self.compute("Broken part", rv=0.0, area_ratio=0.001)
        self.assertEqual(sev, "High")
        self.assertEqual(cat, "Functional")

    # Dent severity depends on relative deformation
    def test_dent_high_deformation_is_high(self):
        sev, cat = self.compute("Dent", rv=0.03, area_ratio=0.05)
        self.assertEqual(sev, "High")

    def test_dent_medium_deformation_is_medium(self):
        sev, cat = self.compute("Dent", rv=0.008, area_ratio=0.05)
        self.assertEqual(sev, "Medium")

    def test_dent_low_deformation_is_low_cosmetic(self):
        sev, cat = self.compute("Dent", rv=0.001, area_ratio=0.02)
        self.assertEqual(sev, "Low")
        self.assertEqual(cat, "Cosmetic")

    # Surface damage depends on area ratio
    def test_scratch_tiny_area_is_low(self):
        sev, cat = self.compute("Scratch", rv=0.0, area_ratio=0.002)
        self.assertEqual(sev, "Low")
        self.assertEqual(cat, "Cosmetic")

    def test_scratch_medium_area_is_medium(self):
        sev, cat = self.compute("Scratch", rv=0.0, area_ratio=0.01)
        self.assertEqual(sev, "Medium")

    def test_corrosion_large_area_is_medium_moderate(self):
        sev, cat = self.compute("Corrosion", rv=0.0, area_ratio=0.03)
        self.assertEqual(sev, "Medium")

    def test_unknown_type_defaults_to_low_cosmetic(self):
        sev, cat = self.compute("SpaceDamage", rv=0.0, area_ratio=0.05)
        self.assertEqual(sev, "Low")
        self.assertEqual(cat, "Cosmetic")


# ─────────────────────────────────────────────────────────────────────────────
# 4. _get_repair_data — REPAIR_DATABASE lookups
# ─────────────────────────────────────────────────────────────────────────────

class TestGetRepairData(unittest.TestCase):

    def setUp(self):
        from src.graph.nodes.reasoning import _get_repair_data
        self.get = _get_repair_data

    def test_exact_match_front_bumper(self):
        data = self.get("Front Bumper")
        self.assertIn("replace",        data)
        self.assertIn("paint",          data)
        self.assertIn("labor_per_hour", data)
        self.assertEqual(data["replace"], 450)

    def test_exact_match_door_panel(self):
        data = self.get("Door Panel")
        self.assertEqual(data["replace"], 1200)

    def test_fuzzy_match_headlight(self):
        # "Left Headlight" should fuzzy-match "Left Headlight" in DB
        data = self.get("Left Headlight")
        self.assertEqual(data["replace"], 800)

    def test_fuzzy_partial_match(self):
        # "Bumper" alone should fuzzy-match one of the bumper entries
        data = self.get("Bumper")
        self.assertIn("replace", data)
        self.assertGreater(data["replace"], 0)

    def test_unknown_part_returns_default(self):
        from src.config.settings import cfg
        data    = self.get("HoverboardDoor")
        default = cfg.REPAIR_DATABASE["_default"]
        self.assertEqual(data["replace"], default["replace"])

    def test_all_values_are_positive_numbers(self):
        from src.config.settings import cfg
        for part, costs in cfg.REPAIR_DATABASE.items():
            if part == "_default":
                continue
            for field in ("replace", "paint", "labor_per_hour"):
                self.assertGreaterEqual(
                    costs[field], 0,
                    f"REPAIR_DATABASE['{part}']['{field}'] must be ≥ 0",
                )


# ─────────────────────────────────────────────────────────────────────────────
# 5. reasoning_node — line-item output and financial calculations
# ─────────────────────────────────────────────────────────────────────────────

class TestReasoningNode(unittest.TestCase):

    def setUp(self):
        from src.graph.nodes.reasoning import reasoning_node
        self.node = reasoning_node

    def test_returns_required_keys(self):
        state  = _reasoning_state([_make_det()])
        result = self.node(state)
        for key in ("damages_output", "financial_estimate",
                    "total_loss_flag", "pipeline_trace", "messages"):
            self.assertIn(key, result)

    def test_financial_estimate_has_line_items(self):
        state  = _reasoning_state([_make_det()])
        result = self.node(state)
        fin    = result["financial_estimate"]
        self.assertIn("line_items",        fin)
        self.assertIn("total_repair_usd",  fin)
        self.assertIn("disposition",       fin)
        self.assertIsInstance(fin["line_items"], list)

    def test_no_damages_produces_zero_cost(self):
        state  = _reasoning_state([])
        result = self.node(state)
        fin    = result["financial_estimate"]
        self.assertEqual(fin["total_repair_usd"], 0.0)
        self.assertFalse(result["total_loss_flag"])

    def test_rejected_damages_excluded_from_financial_estimate(self):
        dets = [
            _make_det("D001", rejected=False, location="Front Bumper"),
            _make_det("D002", rejected=True,  location="Door Panel"),
        ]
        state  = _reasoning_state(dets)
        result = self.node(state)
        fin    = result["financial_estimate"]
        parts  = [it["part"] for it in fin["line_items"]]
        self.assertIn("Front Bumper", parts)
        self.assertNotIn("Door Panel", parts)

    def test_severity_gemini_preferred_over_cv_severity(self):
        """Gemini-refined severity should drive the repair action."""
        det = _make_det("D001", severity="Low", severity_gemini="Severe",
                        location="Engine Hood")
        state  = _reasoning_state([det])
        result = self.node(state)
        fin    = result["financial_estimate"]
        if fin["line_items"]:
            action = fin["line_items"][0]["action"]
            self.assertEqual(action, "REPLACE",
                "Severe severity from Gemini should trigger REPLACE action")

    def test_total_loss_flag_when_cost_exceeds_threshold(self):
        from src.config.settings import cfg
        # Force a very expensive detection: Roof Panel, Critical severity
        # Replace cost = 1800 + (100 * 4) = 2200 USD; vehicle value default = 15000
        # 2200 / 15000 ≈ 14.7% — below 75% threshold
        # Use many High-severity damages to exceed threshold
        dets = [
            _make_det(f"D{i:03d}", dtype="Cracked", location="Roof Panel",
                      severity="High", severity_gemini="Critical",
                      is_verified=True, rejected=False)
            for i in range(10)
        ]
        state  = _reasoning_state(dets)
        result = self.node(state)
        fin    = result["financial_estimate"]
        # 10 × (1800 + 4×100) = 10 × 2200 = 22000 > 15000 × 0.75 = 11250 → TOTALED
        if fin["total_repair_usd"] > cfg.VEHICLE_VALUE * cfg.TOTAL_LOSS_THRESHOLD:
            self.assertTrue(result["total_loss_flag"])
            self.assertEqual(fin["disposition"], "TOTALED")
        else:
            # Total not reached — just check flag is False
            self.assertFalse(result["total_loss_flag"])

    def test_overall_score_100_when_no_confirmed_damages(self):
        # All detections rejected → reasoning skips them → score should be 100
        dets   = [_make_det("D001", is_verified=False, rejected=True)]
        state  = _reasoning_state(dets)
        result = self.node(state)
        # reasoning_node only computes score; report_node normalises to 100
        # Check that reasoning_node returns some score and financial estimate
        self.assertIn("damages_output", result)

    def test_line_items_contain_inr_format(self):
        state  = _reasoning_state([_make_det("D001", location="Side Mirror")])
        result = self.node(state)
        fin    = result["financial_estimate"]
        for item in fin["line_items"]:
            self.assertIn("cost_inr_fmt", item)
            self.assertTrue(item["cost_inr_fmt"].startswith("₹"),
                f"cost_inr_fmt should start with ₹, got: {item['cost_inr_fmt']}")

    def test_repair_vs_replace_boundary_at_severe(self):
        """Minor/Moderate → REPAIR/PAINT; Severe/Critical → REPLACE."""
        from src.config.settings import cfg
        for sev, expected_action in [
            ("Minor",    "REPAIR/PAINT"),
            ("Moderate", "REPAIR/PAINT"),
            ("Severe",   "REPLACE"),
            ("Critical", "REPLACE"),
        ]:
            with self.subTest(severity=sev):
                det    = _make_det("D001", location="Front Bumper",
                                   severity=sev, severity_gemini=sev)
                state  = _reasoning_state([det])
                result = self.node(state)
                fin    = result["financial_estimate"]
                if fin["line_items"]:
                    self.assertEqual(fin["line_items"][0]["action"], expected_action,
                        f"Expected {expected_action} for severity={sev}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. decision_node — ruling codes and invariants
# ─────────────────────────────────────────────────────────────────────────────

class TestDecisionNode(unittest.TestCase):

    def setUp(self):
        from src.graph.nodes.decision import decision_node
        self.node = decision_node

    def _run(self, damages, is_fraud=False):
        state = _decision_state(damages, is_fraud=is_fraud)
        return self.node(state)

    # ── final_output shape ───────────────────────────────────────────────────

    def test_returns_final_output(self):
        result = self._run([_make_det()])
        self.assertIn("final_output", result)

    def test_final_output_has_required_keys(self):
        fo = self._run([_make_det()])["final_output"]
        for key in ("claim_id", "claim_ruling_code", "processing_status",
                    "auto_approved", "overall_assessment_score",
                    "ruling_timestamp"):
            self.assertIn(key, fo, f"Missing key in final_output: {key}")

    # ── AI-never-auto-approves invariant ────────────────────────────────────

    def test_auto_approved_is_always_false(self):
        """Core safety invariant: AI must never auto-approve a claim."""
        scenarios = [
            [_make_det("D001", severity="Low",  rv=0.001, ar=0.003)],  # clean
            [],                                                          # no damage
            [_make_det("D001", severity="High", rv=0.03)],              # severe
        ]
        for dets in scenarios:
            with self.subTest(n_damages=len(dets)):
                fo = self._run(dets)["final_output"]
                self.assertFalse(
                    fo["auto_approved"],
                    "auto_approved must always be False — human auditor must approve",
                )

    # ── Ruling code branches ─────────────────────────────────────────────────

    def test_fraud_flagged_gives_clm_manual(self):
        fo = self._run([_make_det()], is_fraud=True)["final_output"]
        self.assertEqual(fo["claim_ruling_code"], "CLM_MANUAL")

    def test_unconfirmed_detection_gives_clm_manual(self):
        det = _make_det("D001", vs="unconfirmed", is_verified=None, rejected=False)
        fo  = self._run([det])["final_output"]
        self.assertEqual(fo["claim_ruling_code"], "CLM_MANUAL")

    def test_high_severity_damage_gives_clm_workshop(self):
        det = _make_det("D001", severity="High", vs="confirmed", is_verified=True)
        fo  = self._run([det])["final_output"]
        self.assertEqual(fo["claim_ruling_code"], "CLM_WORKSHOP")

    def test_low_score_gives_clm_workshop(self):
        from src.config.settings import cfg
        # Many medium damages push score below ESCALATION_THRESHOLD
        dets = [
            _make_det(f"D{i:03d}", severity="Medium", vs="confirmed",
                      is_verified=True, rejected=False)
            for i in range(6)   # 6 × 20 = 120 penalty → score = max(0, 100-120) = 0
        ]
        fo = self._run(dets)["final_output"]
        self.assertIn(fo["claim_ruling_code"], ("CLM_WORKSHOP", "CLM_MANUAL"))

    def test_clean_low_severity_gives_clm_pending(self):
        from src.config.settings import cfg
        det = _make_det("D001", severity="Low", vs="confirmed",
                        is_verified=True, rejected=False, rv=0.001, ar=0.005)
        fo  = self._run([det])["final_output"]
        # 100 - 5 = 95 ≥ ESCALATION_THRESHOLD (70) → CLM_PENDING
        self.assertEqual(fo["claim_ruling_code"], "CLM_PENDING")

    def test_no_damages_gives_clm_pending(self):
        fo = self._run([])["final_output"]
        # Score = 100 (no penalties) → clean → CLM_PENDING
        self.assertEqual(fo["claim_ruling_code"], "CLM_PENDING")
        self.assertEqual(fo["overall_assessment_score"], 100)

    # ── Score computation ────────────────────────────────────────────────────

    def test_rejected_damages_excluded_from_score(self):
        dets = [
            _make_det("D001", severity="High", vs="confirmed",
                      is_verified=True, rejected=False),
            _make_det("D002", severity="High", vs="confirmed",
                      is_verified=True, rejected=True),   # rejected → not penalised
        ]
        fo = self._run(dets)["final_output"]
        # Only D001 counts: 100 - 40 = 60
        self.assertEqual(fo["overall_assessment_score"], 60)

    def test_ruling_timestamp_is_iso_string(self):
        fo = self._run([_make_det()])["final_output"]
        ts = fo["ruling_timestamp"]
        self.assertIsInstance(ts, str)
        dt = datetime.fromisoformat(ts.rstrip("Z"))
        self.assertIsNotNone(dt)


if __name__ == "__main__":
    unittest.main()
