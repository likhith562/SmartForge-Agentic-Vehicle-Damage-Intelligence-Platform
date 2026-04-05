"""
SmartForge db sub-package.
Exports the public DB API used by both dashboards.
"""

from .mongo_client import (
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
