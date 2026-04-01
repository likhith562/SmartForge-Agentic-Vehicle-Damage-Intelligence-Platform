"""
src/db/mongo_client.py
======================
Hybrid persistence layer for SmartForge claims data.

Strategy
--------
1. **SQLite** — always written first (zero-latency, no network dependency).
   Guarantees real-time reads in the Auditor Dashboard even while MongoDB
   is down or over quota.
2. **MongoDB Atlas** — best-effort cloud sync after every SQLite write.
   Enables cross-session queries, auditor role filtering, and cloud backup.

If MONGO_URI is empty or the Atlas connection fails, the layer operates
entirely on SQLite with no loss of functionality.

Public API
----------
    db_upsert(case_id, **fields)           → None
    db_get(case_id)                        → dict
    db_find(filters, limit)                → list[dict]
    db_count(filters)                      → dict
    db_mark_auditor(case_id, decision, note) → None
    db_backend_info()                      → str
"""

from __future__ import annotations

import json as _json
import sqlite3 as _sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import settings

# ═══════════════════════════════════════════════════════════════════════════════
# Internal state
# ═══════════════════════════════════════════════════════════════════════════════

_USE_MONGO: bool = False
_mongo_col: Any  = None          # pymongo Collection — None until connected
_DESCENDING: Any = None          # pymongo.DESCENDING sentinel

# SQLite path — resolved from settings; parent dirs created on first use
_SQLITE_PATH = Path(settings.SQLITE_PATH)

# JSON-serialised fields that live as text in SQLite but as dicts in MongoDB
_JSON_FIELDS: frozenset[str] = frozenset({
    "user_data",
    "final_output",
    "checkpoint_dump",
    "fraud_report",
    "insurance",
    "agent_trace",
    "chat_history",
    "auditor_review",
})


# ═══════════════════════════════════════════════════════════════════════════════
# MongoDB connection attempt (module load time — non-blocking on failure)
# ═══════════════════════════════════════════════════════════════════════════════

def _connect_mongo() -> None:
    global _USE_MONGO, _mongo_col, _DESCENDING

    mongo_uri = settings.MONGO_URI
    if not mongo_uri or not mongo_uri.strip():
        print("⚠️  [DB] MONGO_URI not set — SQLite-only mode.")
        return

    try:
        from pymongo import MongoClient, DESCENDING as _DESC
        from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=4_000)
        client.admin.command("ping")           # fail fast if unreachable

        db = client["smartforge"]
        col = db["claims"]

        # Indexes for common query patterns
        col.create_index("vehicle_id")
        col.create_index("status")
        col.create_index([("created_at", _DESC)])

        _mongo_col  = col
        _DESCENDING = _DESC
        _USE_MONGO  = True

        safe_uri = mongo_uri[:40] + "…" if len(mongo_uri) > 40 else mongo_uri
        print(f"✅ [DB] MongoDB connected → {safe_uri}")

    except Exception as exc:
        print(f"⚠️  [DB] MongoDB unavailable ({exc}) — SQLite fallback active.")


_connect_mongo()


# ═══════════════════════════════════════════════════════════════════════════════
# SQLite bootstrap
# ═══════════════════════════════════════════════════════════════════════════════

_SCHEMA = """
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
    auditor_review  TEXT
)
"""


def _sqlite_init() -> None:
    """Idempotent — safe to call before every write."""
    _SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = _sqlite3.connect(_SQLITE_PATH)
    conn.execute(_SCHEMA)
    conn.commit()
    conn.close()


if not _USE_MONGO:
    _sqlite_init()
    print(f"✅ [DB] SQLite active → {_SQLITE_PATH}")


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: deserialise JSON text columns on SQLite read-back
# ═══════════════════════════════════════════════════════════════════════════════

def _deserialise_row(desc: list[str], row: tuple) -> dict:
    result: dict = {}
    for col, val in zip(desc, row):
        if col in _JSON_FIELDS and val:
            try:
                val = _json.loads(val)
            except Exception:
                pass
        result[col] = val
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def db_upsert(case_id: str, **fields) -> None:
    """
    Insert or update a claim document.

    Write strategy: SQLite first (always), then MongoDB Atlas (best-effort).
    This guarantees zero data loss even if the Atlas cluster is unavailable.

    Parameters
    ----------
    case_id : str
        Unique claim / session identifier.
    **fields :
        Any column/field values to set on the document.
        JSON-serialisable dicts and lists are handled automatically.
    """
    now = datetime.now(timezone.utc).isoformat()
    fields.setdefault("updated_at", now)

    # ── Step 1: SQLite (always, zero latency) ──────────────────────────────────
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
                f"UPDATE claims SET {col}=? WHERE case_id=?",   # noqa: S608
                (_val, case_id),
            )
        except Exception:
            pass   # silently skip columns not in schema

    conn.commit()
    conn.close()

    # ── Step 2: MongoDB Atlas sync (best-effort, never blocks on failure) ──────
    if _USE_MONGO and _mongo_col is not None:
        try:
            mongo_fields: dict = dict(fields)
            mongo_fields["case_id"] = case_id
            if "vehicle_id" in mongo_fields:
                mongo_fields["user_id"] = mongo_fields["vehicle_id"]

            # Deserialise JSON strings → dicts (avoid double-encoding in Atlas)
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
        except Exception as exc:
            # SQLite already written — no data loss
            print(f"⚠️ [DB] MongoDB sync failed (SQLite OK): {exc}")


def db_get(case_id: str) -> dict:
    """
    Fetch one claim document by its case_id.

    Returns an empty dict if the case is not found.
    """
    if _USE_MONGO and _mongo_col is not None:
        doc = _mongo_col.find_one({"case_id": case_id}, {"_id": 0})
        return doc or {}

    conn = _sqlite3.connect(_SQLITE_PATH)
    cur  = conn.execute("SELECT * FROM claims WHERE case_id=?", (case_id,))
    row  = cur.fetchone()
    conn.close()

    if not row:
        return {}

    desc = [d[0] for d in cur.description] if cur.description else []
    return _deserialise_row(desc, row)


def db_find(filters: dict | None = None, limit: int = 50) -> list[dict]:
    """
    Return a list of claim documents matching *filters*.

    Supported filter keys
    ---------------------
    vehicle_id : str   — case-insensitive partial match
    status     : str   — exact match; pass "" or "All" to skip
    is_fraud   : bool  — exact match
    date_from  : str   — ISO timestamp lower bound (inclusive)

    Parameters
    ----------
    filters : dict, optional
        Key-value pairs described above.  Omit or pass None for all records.
    limit : int
        Maximum number of documents to return (default 50).
    """
    filters = filters or {}

    if _USE_MONGO and _mongo_col is not None:
        query: dict = {}

        if filters.get("vehicle_id"):
            query["vehicle_id"] = {
                "$regex": filters["vehicle_id"], "$options": "i"
            }
        if filters.get("status") and filters["status"] not in ("", "All"):
            query["status"] = filters["status"]
        if filters.get("is_fraud") is not None:
            query["is_fraud"] = bool(filters["is_fraud"])
        if filters.get("date_from"):
            query["created_at"] = {"$gte": filters["date_from"]}

        cursor = (
            _mongo_col
            .find(query, {"_id": 0})
            .sort("created_at", _DESCENDING)
            .limit(limit)
        )
        return list(cursor)

    # ── SQLite path ────────────────────────────────────────────────────────────
    where:  list[str] = ["1=1"]
    params: list      = []

    if filters.get("vehicle_id"):
        where.append("vehicle_id LIKE ?")
        params.append(f"%{filters['vehicle_id']}%")
    if filters.get("status") and filters["status"] not in ("", "All"):
        where.append("status=?")
        params.append(filters["status"])
    if filters.get("is_fraud") is not None:
        where.append("is_fraud=?")
        params.append(int(filters["is_fraud"]))
    if filters.get("date_from"):
        where.append("created_at >= ?")
        params.append(filters["date_from"])

    sql = (
        f"SELECT * FROM claims "                     # noqa: S608
        f"WHERE {' AND '.join(where)} "
        f"ORDER BY created_at DESC "
        f"LIMIT {limit}"
    )

    conn = _sqlite3.connect(_SQLITE_PATH)
    cur  = conn.execute(sql, params)
    rows = cur.fetchall()
    desc = [d[0] for d in cur.description] if cur.description else []
    conn.close()

    return [_deserialise_row(desc, row) for row in rows]


def db_count(filters: dict | None = None) -> dict:
    """
    Return document counts grouped by status for dashboard stats.

    Returns
    -------
    dict with keys: total, analyzed, fraud, approved, rejected, pending
    """
    if _USE_MONGO and _mongo_col is not None:
        pipeline = [{"$group": {"_id": "$status", "n": {"$sum": 1}}}]
        rows   = list(_mongo_col.aggregate(pipeline))
        counts = {r["_id"]: r["n"] for r in rows if r.get("_id") is not None}
    else:
        conn   = _sqlite3.connect(_SQLITE_PATH)
        rows   = conn.execute(
            "SELECT status, COUNT(*) FROM claims GROUP BY status"
        ).fetchall()
        conn.close()
        counts = {r[0]: r[1] for r in rows}

    return {
        "total":    sum(counts.values()),
        "analyzed": counts.get("analyzed", 0),
        "fraud":    counts.get("fraud_checked", 0),
        "approved": counts.get("approved", 0),
        "rejected": counts.get("rejected", 0),
        "pending":  counts.get("claim_submitted", 0) + counts.get("uploaded", 0),
    }


def db_mark_auditor(case_id: str, decision: str, note: str = "") -> None:
    """
    Auditor manually overrides the status of a claim.

    Parameters
    ----------
    case_id  : str   — the claim to update
    decision : str   — one of the four button labels in the Auditor Dashboard
    note     : str   — free-text auditor note (optional)
    """
    _DECISION_TO_STATUS: dict[str, str] = {
        "Confirm Fraud":     "rejected",
        "Clear — Not Fraud": "analyzed",
        "Approve Claim":     "approved",
        "Reject Claim":      "rejected",
    }
    new_status = _DECISION_TO_STATUS.get(decision, "analyzed")

    db_upsert(
        case_id,
        status=new_status,
        auditor_review={
            "decision":    decision,
            "note":        note,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def db_backend_info() -> str:
    """Return a human-readable string describing the active backend."""
    if _USE_MONGO:
        return "MongoDB Atlas"
    return f"SQLite ({_SQLITE_PATH})"
