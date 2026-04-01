"""
src.db
------
Hybrid persistence layer: MongoDB Atlas (primary) with automatic SQLite
fallback when MONGO_URI is empty or the Atlas cluster is unreachable.

Public API — import from here:

    from src.db import db_upsert, db_get, db_find, db_count
    from src.db import db_mark_auditor, db_backend_info
"""

from src.db.mongo_client import (   # noqa: F401
    db_upsert,
    db_get,
    db_find,
    db_count,
    db_mark_auditor,
    db_backend_info,
)

__all__ = [
    "db_upsert",
    "db_get",
    "db_find",
    "db_count",
    "db_mark_auditor",
    "db_backend_info",
]
