"""
SmartForge — Agentic Vehicle Damage Intelligence Platform
Entry point: launches both Gradio dashboards simultaneously.

Usage:
    python main.py

Dashboards:
    User Dashboard    → http://localhost:7860  (+ public share link)
    Auditor Dashboard → http://localhost:7861  (+ public share link)
"""

import os
import sys


def _check_env() -> None:
    """Warn about missing optional secrets — never hard-blocks startup."""
    optional = [
        ("GEMINI_API_KEY",        "Gemini VLM enrichment disabled"),
        ("GROQ_API_KEY",          "Groq report generation disabled — rule-based fallback active"),
        ("SMARTFORGE_MONGO_URI",  "MongoDB not set — SQLite fallback active"),
    ]
    for key, msg in optional:
        if not os.environ.get(key):
            print(f"  ⚠️  {key} not set → {msg}")


def main() -> None:
    print("=" * 64)
    print("  🚗  SmartForge — Agentic Vehicle Damage Intelligence")
    print("=" * 64)

    # ── Environment check ────────────────────────────────────────────────────
    _check_env()

    # ── Lazy imports (heavy deps load after env check) ───────────────────────
    try:
        from src.ui.user_dashboard import build_user_demo
        from src.ui.auditor_dashboard import build_auditor_demo
    except ImportError as exc:
        print(f"\n❌ Import error: {exc}")
        print("   Run: pip install -r requirements.txt")
        sys.exit(1)

    # ── Build Gradio apps ────────────────────────────────────────────────────
    print("\n  Building User Dashboard…")
    user_demo = build_user_demo()

    print("  Building Auditor Dashboard…")
    auditor_demo = build_auditor_demo()

    # ── Launch User Dashboard (non-blocking) ─────────────────────────────────
    print("\n  Launching User Dashboard on port 7860…")
    user_demo.launch(
        server_name         = "0.0.0.0",
        server_port         = 7860,
        share               = True,
        show_api            = False,
        prevent_thread_lock = True,   # must be True so auditor can launch too
    )

    # ── Launch Auditor Dashboard (blocks main thread) ────────────────────────
    print("  Launching Auditor Dashboard on port 7861…")
    print("\n" + "─" * 64)
    print("  User Dashboard    → http://localhost:7860")
    print("  Auditor Dashboard → http://localhost:7861")
    print("─" * 64 + "\n")

    auditor_demo.launch(
        server_name         = "0.0.0.0",
        server_port         = 7861,
        share               = True,
        show_api            = False,
        prevent_thread_lock = False,  # blocks here — keeps process alive
    )


if __name__ == "__main__":
    main()
