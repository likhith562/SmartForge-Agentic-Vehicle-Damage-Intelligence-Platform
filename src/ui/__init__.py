"""
SmartForge ui sub-package.
Exports the two Gradio dashboard builder functions.
"""

from .user_dashboard    import build_user_demo
from .auditor_dashboard import build_auditor_demo

__all__ = [
    "build_user_demo",
    "build_auditor_demo",
]
