"""
src.config
----------
Loads environment variables and exposes validated settings to the rest of
the application.  Import from here — never import settings.py directly.

Usage
-----
    from src.config import settings
    print(settings.GEMINI_MODEL)
"""

from src.config import settings  # noqa: F401  (re-export for convenience)

__all__ = ["settings"]
