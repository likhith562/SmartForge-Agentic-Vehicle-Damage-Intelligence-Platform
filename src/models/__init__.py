"""
SmartForge models sub-package.
Exports the two AI API client functions used across pipeline nodes.
"""

from .gemini_client import call_gemini
from .groq_client import call_groq, generate_groq_narrative

__all__ = [
    "call_gemini",
    "call_groq",
    "generate_groq_narrative",
]
