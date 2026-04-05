"""
tests/test_db_layer.py
========================
Unit and integration tests for the hybrid MongoDB / SQLite persistence layer.

Strategy
--------
All tests redirect cfg.SQLITE_PATH (and the module-level _SQLITE_PATH in
mongo_client) to a fresh temporary file so the real database is never touched.
MongoDB is never required — the test suite forces SQLite mode regardless of
whether SMARTFORGE_MONGO_URI is set, by patching _USE_MONGO to False.

Covers
------
- db_upsert       — insert and update, JSON serialisation of dict fields,
                    boolean coercion of is_fraud
- db_get          — fetch by case_id, missing case returns {}
- db_find         — vehicle_id partial match, status exact match,
                    is_fraud filter, date_from filter, limit
- db_count        — aggregated status counts
- db_mark_auditor — decision write-back + status transition
- db_backend_info — returns expected string
- Round-trip       — upsert → get → verify field fidelity for JSON fields
"""

import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch


# ─────────────────────────────────────────────────────────────────────────────
# Base class — creates an isolated temp DB for every test
# ─────────────────────────────────────────────────────────────────────────────

class DBTestBase(unittest.TestCase):
    """
    Sets up a fresh SQLite temp file and patches the db module to use it.
    Every test method gets a clean, empty database.
    """

    def setUp(self):
        # Create isolated temp DB
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self._db_path = self._tmp.name

        # Patch the module-level variables so all db_* calls hit the temp file
        import src.db.mongo_client as _mc
        self._orig_sqlite   = _mc._SQLITE_PATH
        self._orig_use_mongo = _mc._USE_MONGO

        _mc._SQLITE_PATH = self._db_path
        _mc._USE_MONGO   = False          # force SQLite regardless of env vars

        # Re-import public API after patching (same module object)
        from src.db.mongo_client import (
            db_upsert, db_get, db_find, db_count,
            db_mark_auditor, db_backend_info,
        )
        self.upsert       = db_upsert
        self.get          = db_get
        self.find         = db_find
        self.count        = db_count
        self.mark_auditor = db_mark_auditor
        self.backend_info = db_backend_info

    def tearDown(self):
        import src.db.mongo_client as _mc
        _mc._SQLITE_PATH = self._orig_sqlite
        _mc._USE_MONGO   = self._orig_use_mongo
        if os.path.exists(self._db_path):
            os.unlink(self._db_path)

    # ── Convenience helpers ───────────────────────────────────────────────────

    def _insert(self, case_id: str = "CASE-001", **extra) -> str:
        defaults = {
            "vehicle_id": "VH001",
            "status":     "uploaded",
        }
        defaults.update(extra)
        self.upsert(case_id, **defaults)
        return case_id


# ─────────────────────────────────────────────────────────────────────────────
# 1. db_backend_info
# ─────────────────────────────────────────────────────────────────────────────

class TestBackendInfo(DBTestBase):

    def test_returns_sqlite_string_when_mongo_disabled(self):
        info = self.backend_info()
        self.assertIn("SQLite", info)
        self.assertIn(self._db_path, info)

    def test_returns_string_type(self):
        self.assertIsInstance(self.backend_info(), str)


# ─────────────────────────────────────────────────────────────────────────────
# 2. db_upsert — insert
# ─────────────────────────────────────────────────────────────────────────────

class TestUpsertInsert(DBTestBase):

    def test_insert_creates_record(self):
        self._insert("CASE-001")
        rec = self.get("CASE-001")
        self.assertEqual(rec["case_id"], "CASE-001")

    def test_insert_sets_vehicle_id(self):
        self._insert("CASE-002", vehicle_id="VH042")
        rec = self.get("CASE-002")
        self.assertEqual(rec["vehicle_id"], "VH042")

    def test_insert_sets_status(self):
        self._insert("CASE-003", status="analyzed")
        rec = self.get("CASE-003")
        self.assertEqual(rec["status"], "analyzed")

    def test_insert_sets_updated_at(self):
        self._insert("CASE-004")
        rec = self.get("CASE-004")
        self.assertIsNotNone(rec.get("updated_at"))

    def test_insert_sets_created_at(self):
        self._insert("CASE-005")
        rec = self.get("CASE-005")
        self.assertIsNotNone(rec.get("created_at"))

    def test_is_fraud_stored_as_integer_zero(self):
        self._insert("CASE-006", is_fraud=False)
        conn = sqlite3.connect(self._db_path)
        row  = conn.execute(
            "SELECT is_fraud FROM claims WHERE case_id=?", ("CASE-006",)
        ).fetchone()
        conn.close()
        self.assertEqual(row[0], 0)

    def test_is_fraud_stored_as_integer_one(self):
        self._insert("CASE-007", is_fraud=True)
        conn = sqlite3.connect(self._db_path)
        row  = conn.execute(
            "SELECT is_fraud FROM claims WHERE case_id=?", ("CASE-007",)
        ).fetchone()
        conn.close()
        self.assertEqual(row[0], 1)

    def test_json_field_serialised_and_deserialised(self):
        payload = {"key": "value", "nested": {"a": 1}}
        self._insert("CASE-008", user_data=payload)
        rec = self.get("CASE-008")
        self.assertEqual(rec["user_data"], payload)

    def test_list_json_field_round_trips(self):
        history = [["hello", "world"], ["foo", "bar"]]
        self._insert("CASE-009", chat_history=history)
        rec = self.get("CASE-009")
        self.assertEqual(rec["chat_history"], history)

    def test_unknown_column_does_not_raise(self):
        """Extra kwargs for columns that don't exist should be silently skipped."""
        try:
            self._insert("CASE-010", nonexistent_field="oops")
        except Exception as exc:
            self.fail(f"Unknown column raised an exception: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. db_upsert — update (idempotent upsert)
# ─────────────────────────────────────────────────────────────────────────────

class TestUpsertUpdate(DBTestBase):

    def test_second_upsert_updates_status(self):
        self._insert("CASE-100", status="uploaded")
        self.upsert("CASE-100", status="analyzed")
        rec = self.get("CASE-100")
        self.assertEqual(rec["status"], "analyzed")

    def test_upsert_preserves_existing_fields(self):
        self._insert("CASE-101", vehicle_id="VH001", status="uploaded")
        self.upsert("CASE-101", status="analyzed")
        rec = self.get("CASE-101")
        self.assertEqual(rec["vehicle_id"], "VH001")

    def test_upsert_updates_updated_at(self):
        import time
        self._insert("CASE-102")
        first_updated = self.get("CASE-102").get("updated_at", "")
        time.sleep(0.01)  # ensure clock advances
        self.upsert("CASE-102", status="analyzed")
        second_updated = self.get("CASE-102").get("updated_at", "")
        self.assertGreaterEqual(second_updated, first_updated)

    def test_upsert_overwrites_json_field(self):
        self._insert("CASE-103", user_data={"v": 1})
        self.upsert("CASE-103", user_data={"v": 2, "extra": True})
        rec = self.get("CASE-103")
        self.assertEqual(rec["user_data"]["v"], 2)
        self.assertTrue(rec["user_data"]["extra"])

    def test_multiple_upserts_do_not_duplicate_rows(self):
        for _ in range(5):
            self.upsert("CASE-104", status="uploaded")
        conn = sqlite3.connect(self._db_path)
        cnt  = conn.execute(
            "SELECT COUNT(*) FROM claims WHERE case_id=?", ("CASE-104",)
        ).fetchone()[0]
        conn.close()
        self.assertEqual(cnt, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. db_get
# ─────────────────────────────────────────────────────────────────────────────

class TestGet(DBTestBase):

    def test_returns_dict(self):
        self._insert("CASE-200")
        rec = self.get("CASE-200")
        self.assertIsInstance(rec, dict)

    def test_returns_empty_dict_for_missing_case(self):
        rec = self.get("DOES-NOT-EXIST")
        self.assertEqual(rec, {})

    def test_case_id_in_result(self):
        self._insert("CASE-201")
        rec = self.get("CASE-201")
        self.assertEqual(rec["case_id"], "CASE-201")

    def test_complex_json_round_trip(self):
        fo = {
            "claim_ruling_code": "CLM_PENDING",
            "overall_assessment_score": 85,
            "damages": [{"detection_id": "D001", "type": "Dent"}],
            "financial_estimate": {"total_repair_usd": 450.0},
        }
        self._insert("CASE-202", final_output=fo)
        rec = self.get("CASE-202")
        self.assertEqual(rec["final_output"]["claim_ruling_code"],        "CLM_PENDING")
        self.assertEqual(rec["final_output"]["overall_assessment_score"], 85)
        self.assertEqual(len(rec["final_output"]["damages"]),             1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. db_find
# ─────────────────────────────────────────────────────────────────────────────

class TestFind(DBTestBase):

    def _seed(self):
        """Insert a varied set of records for filter testing."""
        records = [
            ("CASE-A1", "VH001", "uploaded",        False),
            ("CASE-A2", "VH001", "analyzed",         False),
            ("CASE-A3", "VH002", "claim_submitted",  False),
            ("CASE-A4", "VH002", "fraud_flagged",    True),
            ("CASE-A5", "VH003", "approved",         False),
            ("CASE-A6", "VH003", "rejected",         True),
        ]
        for cid, vid, status, fraud in records:
            self.upsert(cid, vehicle_id=vid, status=status, is_fraud=fraud)

    def test_empty_filter_returns_all(self):
        self._seed()
        results = self.find({}, limit=100)
        self.assertEqual(len(results), 6)

    def test_vehicle_id_partial_match(self):
        self._seed()
        results = self.find({"vehicle_id": "VH001"})
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(r["vehicle_id"], "VH001")

    def test_status_exact_match(self):
        self._seed()
        results = self.find({"status": "approved"})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["case_id"], "CASE-A5")

    def test_status_all_skips_filter(self):
        self._seed()
        results_all   = self.find({"status": "All"}, limit=100)
        results_empty = self.find({},               limit=100)
        self.assertEqual(len(results_all), len(results_empty))

    def test_is_fraud_true_filter(self):
        self._seed()
        results = self.find({"is_fraud": True})
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertTrue(r["is_fraud"])

    def test_is_fraud_false_filter(self):
        self._seed()
        results = self.find({"is_fraud": False})
        self.assertEqual(len(results), 4)

    def test_combined_vehicle_and_status_filter(self):
        self._seed()
        results = self.find({"vehicle_id": "VH001", "status": "analyzed"})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["case_id"], "CASE-A2")

    def test_limit_respected(self):
        self._seed()
        results = self.find({}, limit=3)
        self.assertLessEqual(len(results), 3)

    def test_returns_list_of_dicts(self):
        self._seed()
        results = self.find({})
        self.assertIsInstance(results, list)
        for r in results:
            self.assertIsInstance(r, dict)

    def test_empty_db_returns_empty_list(self):
        results = self.find({})
        self.assertEqual(results, [])

    def test_date_from_filter(self):
        """Records with created_at ≥ date_from should be returned."""
        import time
        self._insert("CASE-EARLY", vehicle_id="VH099", status="uploaded")
        # Stamp an earlier created_at directly
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "UPDATE claims SET created_at=? WHERE case_id=?",
            ("2020-01-01T00:00:00", "CASE-EARLY"),
        )
        conn.commit(); conn.close()

        self._insert("CASE-RECENT", vehicle_id="VH099", status="analyzed")
        results = self.find({"date_from": "2023-01-01"})
        ids = [r["case_id"] for r in results]
        self.assertIn("CASE-RECENT", ids)
        self.assertNotIn("CASE-EARLY", ids)


# ─────────────────────────────────────────────────────────────────────────────
# 6. db_count
# ─────────────────────────────────────────────────────────────────────────────

class TestCount(DBTestBase):

    def _seed(self):
        statuses = [
            "uploaded", "uploaded",
            "analyzed", "analyzed", "analyzed",
            "claim_submitted",
            "approved",
            "rejected",
        ]
        for i, s in enumerate(statuses):
            self.upsert(f"CASE-C{i:03d}", vehicle_id="VH001", status=s)

    def test_total_count_matches_insertions(self):
        self._seed()
        counts = self.count()
        self.assertEqual(counts["total"], 8)

    def test_analyzed_count(self):
        self._seed()
        counts = self.count()
        self.assertEqual(counts["analyzed"], 3)

    def test_approved_count(self):
        self._seed()
        counts = self.count()
        self.assertEqual(counts["approved"], 1)

    def test_rejected_count(self):
        self._seed()
        counts = self.count()
        self.assertEqual(counts["rejected"], 1)

    def test_pending_includes_uploaded_and_claim_submitted(self):
        self._seed()
        counts = self.count()
        # pending = uploaded(2) + claim_submitted(1) = 3
        self.assertEqual(counts["pending"], 3)

    def test_empty_db_returns_zeros(self):
        counts = self.count()
        self.assertEqual(counts["total"],    0)
        self.assertEqual(counts["approved"], 0)
        self.assertEqual(counts["rejected"], 0)

    def test_count_returns_all_expected_keys(self):
        counts = self.count()
        for key in ("total", "analyzed", "fraud", "approved", "rejected", "pending"):
            self.assertIn(key, counts)

    def test_all_counts_are_non_negative_integers(self):
        self._seed()
        counts = self.count()
        for key, val in counts.items():
            self.assertIsInstance(val, int, f"counts['{key}'] should be int")
            self.assertGreaterEqual(val, 0, f"counts['{key}'] should be ≥ 0")


# ─────────────────────────────────────────────────────────────────────────────
# 7. db_mark_auditor
# ─────────────────────────────────────────────────────────────────────────────

class TestMarkAuditor(DBTestBase):

    def setUp(self):
        super().setUp()
        self._insert("CASE-M001", vehicle_id="VH001", status="claim_submitted")

    def test_confirm_fraud_sets_status_rejected(self):
        self.mark_auditor("CASE-M001", "Confirm Fraud", "Looks fake")
        rec = self.get("CASE-M001")
        self.assertEqual(rec["status"], "rejected")

    def test_clear_not_fraud_sets_status_analyzed(self):
        self.mark_auditor("CASE-M001", "Clear — Not Fraud", "Verified OK")
        rec = self.get("CASE-M001")
        self.assertEqual(rec["status"], "analyzed")

    def test_approve_claim_sets_status_approved(self):
        self.mark_auditor("CASE-M001", "Approve Claim", "All checks pass")
        rec = self.get("CASE-M001")
        self.assertEqual(rec["status"], "approved")

    def test_reject_claim_sets_status_rejected(self):
        self.mark_auditor("CASE-M001", "Reject Claim", "Invalid docs")
        rec = self.get("CASE-M001")
        self.assertEqual(rec["status"], "rejected")

    def test_auditor_review_field_persisted(self):
        self.mark_auditor("CASE-M001", "Approve Claim", "Legit claim")
        rec = self.get("CASE-M001")
        ar  = rec.get("auditor_review") or {}
        self.assertEqual(ar.get("decision"), "Approve Claim")
        self.assertEqual(ar.get("note"),     "Legit claim")
        self.assertIn("reviewed_at", ar)

    def test_reviewed_at_is_iso_string(self):
        self.mark_auditor("CASE-M001", "Reject Claim")
        rec = self.get("CASE-M001")
        ts  = (rec.get("auditor_review") or {}).get("reviewed_at", "")
        dt  = datetime.fromisoformat(ts)
        self.assertIsNotNone(dt)

    def test_note_defaults_to_empty_string(self):
        self.mark_auditor("CASE-M001", "Approve Claim")
        rec = self.get("CASE-M001")
        ar  = rec.get("auditor_review") or {}
        self.assertEqual(ar.get("note", ""), "")

    def test_unknown_decision_falls_back_to_analyzed(self):
        self.mark_auditor("CASE-M001", "SomeUnknownDecision")
        rec = self.get("CASE-M001")
        self.assertEqual(rec["status"], "analyzed")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Round-trip fidelity — complex nested JSON
# ─────────────────────────────────────────────────────────────────────────────

class TestRoundTripFidelity(DBTestBase):

    def test_full_final_output_round_trip(self):
        fo = {
            "claim_id":                 "CLM-VH001-ABC123",
            "claim_ruling_code":        "CLM_PENDING",
            "overall_assessment_score": 78,
            "confirmed_damage_count":   3,
            "auto_approved":            False,
            "damages": [
                {"detection_id": "D001", "type": "Dent",    "location": "Front Bumper"},
                {"detection_id": "D002", "type": "Scratch", "location": "Door Panel"},
            ],
            "financial_estimate": {
                "total_repair_usd":     725.0,
                "total_repair_inr_fmt": "₹60,175",
                "disposition":          "REPAIRABLE",
                "total_loss_flag":      False,
                "line_items": [
                    {"part": "Front Bumper", "action": "REPAIR/PAINT",
                     "cost_usd": 370.0, "cost_inr_fmt": "₹30,710"},
                ],
            },
            "executive_summary": "Two confirmed damages found on the vehicle.",
        }
        self.upsert("CASE-RT001", vehicle_id="VH001",
                    status="analyzed", final_output=fo)
        rec = self.get("CASE-RT001")
        stored_fo = rec["final_output"]

        self.assertEqual(stored_fo["claim_ruling_code"],        "CLM_PENDING")
        self.assertEqual(stored_fo["overall_assessment_score"], 78)
        self.assertEqual(len(stored_fo["damages"]),             2)
        self.assertAlmostEqual(
            stored_fo["financial_estimate"]["total_repair_usd"], 725.0
        )
        self.assertEqual(
            stored_fo["financial_estimate"]["line_items"][0]["part"],
            "Front Bumper",
        )

    def test_fraud_report_round_trip(self):
        fr = {
            "trust_score": 25,
            "status":      "SUSPICIOUS_HIGH_RISK",
            "flags":       ["RECYCLED_IMAGE: pHash match", "SCREEN_CAPTURE"],
            "details": {
                "phash_check":         {"status": "DUPLICATE_DETECTED", "hamming_distance": 0},
                "screen_detection":    {"is_screen": True, "confidence": 0.9},
                "ai_generation_check": {"ela_score": 1.25, "method": "ela_forensics"},
            },
            "checks_run": 5,
        }
        self.upsert("CASE-RT002", fraud_report=fr, is_fraud=True)
        rec = self.get("CASE-RT002")

        self.assertEqual(rec["fraud_report"]["trust_score"], 25)
        self.assertEqual(len(rec["fraud_report"]["flags"]),  2)
        self.assertEqual(
            rec["fraud_report"]["details"]["phash_check"]["hamming_distance"], 0
        )
        self.assertTrue(rec["is_fraud"])

    def test_agent_trace_round_trip(self):
        trace = {
            "intake_agent":  {"timestamp": "2024-01-01T00:00:00", "decision": "Image accepted"},
            "fraud_agent":   {"timestamp": "2024-01-01T00:00:01", "decision": "VERIFIED"},
            "report_agent":  {"timestamp": "2024-01-01T00:00:10", "decision": "Report assembled"},
        }
        self.upsert("CASE-RT003", agent_trace=trace)
        rec = self.get("CASE-RT003")
        self.assertEqual(
            rec["agent_trace"]["intake_agent"]["decision"], "Image accepted"
        )
        self.assertEqual(len(rec["agent_trace"]), 3)


if __name__ == "__main__":
    unittest.main()
