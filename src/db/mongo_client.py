"""
SmartForge — Database Layer
============================
Hybrid persistence: always writes to SQLite first (zero-latency, no network
dependency), then attempts a best-effort sync to MongoDB Atlas.

This guarantees:
  - Zero data loss even when MongoDB is unreachable or quota-exceeded
  - Real-time Auditor Dashboard reads work via SQLite instantly
  - MongoDB provides cloud persistence and cross-session search

Public API
----------
    db_upsert(case_id, **fields)          Insert or update a case
    db_get(case_id)   → dict             Fetch one case by ID
    db_find(filters, limit) → list       Query cases with a filter dict
    db_count()        → dict             Status counts for stat cards
    db_mark_auditor(case_id, decision, note)  Auditor write-back
    db_backend_info() → str              "MongoDB Atlas" | "SQLite (path)"

Filter keys understood by db_find
----------------------------------
    vehicle_id   str   — partial-match search
    status       str   — exact match (or "All" to skip)
    is_fraud     bool  — True = fraud-only
    date_from    str   — ISO date string (created_at >= date_from)

Status pipeline
---------------
    uploaded → pref_saved → analyzed → claim_submitted
             → fraud_checked → approved / rejected
"""

import json as _json
import os as _os
import sqlite3 as _sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.config.settings import cfg

# ── JSON fields (serialised as strings in SQLite, dicts in MongoDB) ───────────
_JSON_FIELDS = {
    "user_data", "final_output", "checkpoint_dump", "fraud_report",
    "insurance", "agent_trace", "chat_history", "auditor_review",
}

# ─────────────────────────────────────────────────────────────────────────────
# MongoDB initialisation (optional — graceful degradation to SQLite)
# ─────────────────────────────────────────────────────────────────────────────

_USE_MONGO: bool = False
_mongo_col = None   # pymongo Collection handle, set below if connection succeeds

_MONGO_URI: str = _os.environ.get("SMARTFORGE_MONGO_URI", cfg.MONGO_URI or "")

if _MONGO_URI and _MONGO_URI.strip():
    try:
        from pymongo import MongoClient, DESCENDING
        from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

        _client    = MongoClient(_MONGO_URI, serverSelectionTimeoutMS=4000)
        _client.admin.command("ping")           # test connection (4 s timeout)
        _db        = _client["smartforge"]
        _mongo_col = _db["claims"]

        # Indexes for fast lookup patterns used by both dashboards
        _mongo_col.create_index("vehicle_id")
        _mongo_col.create_index("status")
        _mongo_col.create_index([("created_at", DESCENDING)])

        _USE_MONGO = True
        print(f"✅ [DB] MongoDB connected → {_MONGO_URI[:40]}…")

    except Exception as _me:
        print(f"⚠️  [DB] MongoDB unavailable ({_me}) — SQLite fallback active.")
else:
    print("⚠️  [DB] MONGO_URI empty — SQLite fallback active.")

# ─────────────────────────────────────────────────────────────────────────────
# SQLite helpers
# ─────────────────────────────────────────────────────────────────────────────

_SQLITE_PATH: str = cfg.SQLITE_PATH


def _sqlite_init() -> None:
    """Create the claims table if it does not exist (idempotent)."""
    conn = _sqlite3.connect(_SQLITE_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS claims (
            case_id         TEXT PRIMARY KEY,
            user_id         TEXT,
            vehicle_id      TEXT,
            status          TEXT DEFAULT 'uploaded',
            created_at      TEXT,
            updated_at      TEXT,
            user_data       TEXT,
            final_output    TEXT,
            checkpoint_dump TEXT,
            fraud_report    TEXT,
            fraud_hash      TEXT,
            insurance       TEXT,
            agent_trace     TEXT,
            chat_history    TEXT,
            is_fraud        INTEGER DEFAULT 0,
            fraud_attempts  INTEGER DEFAULT 0,
            auditor_review  TEXT,
            images          TEXT
        )
    """)
    conn.commit()
    conn.close()


def _sqlite_row_to_dict(cursor: _sqlite3.Cursor, row: tuple) -> Dict[str, Any]:
    """Convert a SQLite row + cursor description to a dict, deserialising JSON fields."""
    desc = [d[0] for d in cursor.description] if cursor.description else []
    result: Dict[str, Any] = {}
    for col, val in zip(desc, row):
        if col in _JSON_FIELDS and val:
            try:
                val = _json.loads(val)
            except Exception:
                pass
        result[col] = val
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def db_upsert(case_id: str, **fields: Any) -> None:
    """
    Hybrid upsert.

    Step 1 — always writes to SQLite immediately (zero-latency, no network).
    Step 2 — attempts MongoDB Atlas sync (best-effort, never raises).

    Parameters
    ----------
    case_id : str
        Unique identifier for the claim case.
    **fields
        Any column/field values to insert or update.
        JSON-serialisable dicts are automatically serialised for SQLite
        and kept as dicts for MongoDB.
    """
    now = datetime.now(timezone.utc).isoformat()
    fields.setdefault("updated_at", now)

    # ── Step 1: SQLite (always) ───────────────────────────────────────────────
    _sqlite_init()
    conn = _sqlite3.connect(_SQLITE_PATH)
    exists = conn.execute(
        "SELECT 1 FROM claims WHERE case_id=?", (case_id,)
    ).fetchone()

    if not exists:
        conn.execute(
            "INSERT INTO claims (case_id, created_at, updated_at) VALUES (?,?,?)",
            (case_id, now, now),
        )

    for col, val in fields.items():
        _val = val
        if col in _JSON_FIELDS and isinstance(_val, (dict, list)):
            _val = _json.dumps(_val, default=str)
        if col == "is_fraud":
            _val = int(bool(_val))
        try:
            conn.execute(
                f"UPDATE claims SET {col}=? WHERE case_id=?", (_val, case_id)
            )
        except _sqlite3.OperationalError:
            pass   # unknown column — skip gracefully
    conn.commit()
    conn.close()

    # ── Step 2: MongoDB Atlas sync (best-effort) ──────────────────────────────
    if _USE_MONGO:
        try:
            mongo_fields: Dict[str, Any] = dict(fields)
            mongo_fields["case_id"] = case_id
            if "vehicle_id" in mongo_fields:
                mongo_fields["user_id"] = mongo_fields["vehicle_id"]

            # Deserialise JSON strings back to dicts (no double-encoding in Mongo)
            for k in list(mongo_fields.keys()):
                if k in _JSON_FIELDS and isinstance(mongo_fields[k], str):
                    try:
                        mongo_fields[k] = _json.loads(mongo_fields[k])
                    except Exception:
                        pass

            _mongo_col.update_one(
                {"case_id": case_id},
                {
                    "$set":         mongo_fields,
                    "$setOnInsert": {"created_at": now},
                },
                upsert=True,
            )
        except Exception as _me:
            # MongoDB sync failed — SQLite already written, so no data loss
            print(f"⚠️ [DB] MongoDB sync failed (SQLite OK): {_me}")


def db_get(case_id: str) -> Dict[str, Any]:
    """
    Fetch one case by its exact case_id.

    Returns an empty dict if the case does not exist.
    Reads from MongoDB when available, otherwise SQLite.
    """
    if _USE_MONGO:
        doc = _mongo_col.find_one({"case_id": case_id}, {"_id": 0})
        return doc or {}

    _sqlite_init()
    conn = _sqlite3.connect(_SQLITE_PATH)
    cur  = conn.execute("SELECT * FROM claims WHERE case_id=?", (case_id,))
    row  = cur.fetchone()
    conn.close()
    if not row:
        return {}
    return _sqlite_row_to_dict(cur, row)


def db_find(
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Return a list of case documents matching *filters*, sorted newest first.

    Supported filter keys
    ---------------------
    vehicle_id  str   partial match (LIKE %value%)
    status      str   exact match; "All" or "" skips this filter
    is_fraud    bool  True = fraud-flagged cases only
    date_from   str   ISO date; returns cases created on or after this date

    Role enforcement is the caller's responsibility:
      - User dashboard passes {"vehicle_id": current_user_id}
      - Auditor dashboard passes {} (no filter) for full visibility
    """
    filters = filters or {}

    if _USE_MONGO:
        from pymongo import DESCENDING as _DESC
        query: Dict[str, Any] = {}
        if filters.get("vehicle_id"):
            query["vehicle_id"] = {
                "$regex": filters["vehicle_id"], "$options": "i"
            }
        if filters.get("status") and filters["status"] not in ("All", ""):
            query["status"] = filters["status"]
        if filters.get("is_fraud") is not None:
            query["is_fraud"] = bool(filters["is_fraud"])
        if filters.get("date_from"):
            query["created_at"] = {"$gte": filters["date_from"]}

        cursor = (
            _mongo_col.find(query, {"_id": 0})
            .sort("created_at", _DESC)
            .limit(limit)
        )
        return list(cursor)

    # SQLite path
    _sqlite_init()
    where:  List[str] = ["1=1"]
    params: List[Any] = []

    if filters.get("vehicle_id"):
        where.append("vehicle_id LIKE ?")
        params.append(f"%{filters['vehicle_id']}%")
    if filters.get("status") and filters["status"] not in ("All", ""):
        where.append("status=?")
        params.append(filters["status"])
    if filters.get("is_fraud") is not None:
        where.append("is_fraud=?")
        params.append(int(filters["is_fraud"]))
    if filters.get("date_from"):
        where.append("created_at >= ?")
        params.append(filters["date_from"])

    sql = (
        f"SELECT * FROM claims WHERE {' AND '.join(where)} "
        f"ORDER BY created_at DESC LIMIT {int(limit)}"
    )
    conn = _sqlite3.connect(_SQLITE_PATH)
    cur  = conn.execute(sql, params)
    rows = cur.fetchall()
    conn.close()

    return [_sqlite_row_to_dict(cur, row) for row in rows]


def db_count(filters: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    """
    Return status counts for the Auditor Dashboard stat cards.

    Returns
    -------
    dict with keys: total, analyzed, fraud, approved, rejected, pending
    """
    _ = filters  # reserved for future role-scoped counts

    if _USE_MONGO:
        pipeline = [{"$group": {"_id": "$status", "n": {"$sum": 1}}}]
        rows     = list(_mongo_col.aggregate(pipeline))
        counts   = {r["_id"]: r["n"] for r in rows if r.get("_id") is not None}
    else:
        _sqlite_init()
        conn   = _sqlite3.connect(_SQLITE_PATH)
        rows   = conn.execute(
            "SELECT status, COUNT(*) FROM claims GROUP BY status"
        ).fetchall()
        conn.close()
        counts = {r[0]: r[1] for r in rows}

    return {
        "total":    sum(counts.values()),
        "analyzed": counts.get("analyzed", 0),
        "fraud":    counts.get("fraud_flagged", 0),
        "approved": counts.get("approved", 0),
        "rejected": counts.get("rejected", 0),
        "pending":  counts.get("claim_submitted", 0) + counts.get("uploaded", 0),
    }


def db_mark_auditor(
    case_id: str,
    decision: str,
    note: str = "",
) -> None:
    """
    Record an auditor decision on a case and update its status.

    Parameters
    ----------
    case_id  : str  — target case
    decision : str  — one of:
                      "Confirm Fraud" | "Clear — Not Fraud"
                      "Approve Claim" | "Reject Claim"
    note     : str  — optional free-text reasoning for the audit trail
    """
    new_status = {
        "Confirm Fraud":     "rejected",
        "Clear — Not Fraud": "analyzed",
        "Approve Claim":     "approved",
        "Reject Claim":      "rejected",
    }.get(decision, "analyzed")

    db_upsert(
        case_id,
        status        = new_status,
        auditor_review = {
            "decision":    decision,
            "note":        note,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def db_backend_info() -> str:
    """Return a human-readable string identifying the active DB backend."""
    return "MongoDB Atlas" if _USE_MONGO else f"SQLite ({_SQLITE_PATH})"
