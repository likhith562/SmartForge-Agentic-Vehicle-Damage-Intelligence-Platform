"""
src/config/settings.py
======================
Central configuration for SmartForge.

Priority order (highest → lowest):
  1. Real environment variables  (export SMARTFORGE_MONGO_URI=...)
  2. .env file in project root   (loaded via python-dotenv if installed)
  3. Hard-coded defaults below

Never commit real API keys.  Use .env (git-ignored) or your OS key-store.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Optional: load .env file ───────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass   # python-dotenv not installed — environment variables must be set manually


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MODEL WEIGHTS  (set these to your actual file paths)
# ═══════════════════════════════════════════════════════════════════════════════

DAMAGE_MODEL_PATH = os.getenv("DAMAGE_MODEL_PATH", "models/seg-best.pt")
PART_MODEL_PATH   = os.getenv("PART_MODEL_PATH",   "models/detect-best.pt")

# SAM (Segment Anything) — downloaded automatically on first run if missing
SAM_CHECKPOINT = os.getenv("SAM_CHECKPOINT", "models/sam_vit_b_01ec64.pth")
SAM_URL        = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. API KEYS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Gemini VLM ─────────────────────────────────────────────────────────────────
# Free key: https://aistudio.google.com/app/apikey
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL      = "gemini-2.5-flash"       # primary (multimodal, free tier)
GEMINI_FALLBACK   = "gemini-2.5-flash-lite"  # auto-fallback on HTTP 429
GEMINI_ENABLED    = bool(GEMINI_API_KEY)

# ── Groq (report generation — fast inference) ──────────────────────────────────
# Free key: https://console.groq.com
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"
GROQ_ENABLED = bool(GROQ_API_KEY)

# ── SerpAPI (reverse image search) ────────────────────────────────────────────
# 100 free searches/month: https://serpapi.com
SERPAPI_KEY     = os.getenv("SERPAPI_KEY", "")
SERPAPI_ENABLED = bool(SERPAPI_KEY)

# ── Winston AI (AI-generation detection) ──────────────────────────────────────
# 2 000 free credits/month: https://app.gowinston.ai
WINSTON_AI_KEY     = os.getenv("WINSTON_AI_KEY", "")
WINSTON_AI_ENABLED = bool(WINSTON_AI_KEY)
WINSTON_AI_THRESHOLD = 0.70   # AI-probability above this → flagged


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

# MongoDB Atlas URI — leave empty to fall back to local SQLite automatically.
# Atlas free tier: https://mongodb.com/atlas
# Format: "mongodb+srv://<user>:<password>@<cluster>.mongodb.net/"
MONGO_URI   = os.getenv("SMARTFORGE_MONGO_URI", "")
SQLITE_PATH = os.getenv("SQLITE_PATH", "data/smartforge_claims.db")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. COMPUTER VISION / SAHI
# ═══════════════════════════════════════════════════════════════════════════════

SAHI_CONFIDENCE = float(os.getenv("SAHI_CONFIDENCE", "0.3"))
SAHI_SLICE_SIZE = int(os.getenv("SAHI_SLICE_SIZE",   "640"))
SAHI_OVERLAP    = float(os.getenv("SAHI_OVERLAP",    "0.2"))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. AGENTIC THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

MAX_RETRIES              = int(float(os.getenv("MAX_RETRIES",              "2")))
ESCALATION_THRESHOLD     = int(float(os.getenv("ESCALATION_THRESHOLD",     "70")))
CONFIDENCE_RECHECK_LIMIT = float(os.getenv("CONFIDENCE_RECHECK_LIMIT",    "0.45"))
AUTO_APPROVE_THRESHOLD   = int(float(os.getenv("AUTO_APPROVE_THRESHOLD",   "85")))
HEALTH_SCORE_MIN         = float(os.getenv("HEALTH_SCORE_MIN",            "0.6"))


# ═══════════════════════════════════════════════════════════════════════════════
# 6. FRAUD DETECTION LAYER (Batch 1)
# ═══════════════════════════════════════════════════════════════════════════════

FRAUD_TRUST_THRESHOLD      = int(float(os.getenv("FRAUD_TRUST_THRESHOLD",     "40")))
FRAUD_GPS_MAX_DISTANCE_KM  = float(os.getenv("FRAUD_GPS_MAX_DISTANCE_KM",    "50.0"))
PHASH_HAMMING_THRESHOLD    = int(float(os.getenv("PHASH_HAMMING_THRESHOLD",   "8")))
FRAUD_HASH_DB_PATH         = os.getenv("FRAUD_HASH_DB_PATH", "data/fraud_hash_db.json")
MAX_FRAUD_RETRIES          = int(float(os.getenv("MAX_FRAUD_RETRIES",         "3")))

# Set BYPASS_FRAUD=true to skip all fraud checks in demo/dev mode
BYPASS_FRAUD = os.getenv("BYPASS_FRAUD", "false").lower() in ("1", "true", "yes")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. GOLDEN FRAME VERIFICATION (Batch 3)
# ═══════════════════════════════════════════════════════════════════════════════

GOLDEN_FRAME_CROP_MARGIN    = float(os.getenv("GOLDEN_FRAME_CROP_MARGIN",    "0.25"))
GOLDEN_FRAME_MIN_CROP_PX    = int(float(os.getenv("GOLDEN_FRAME_MIN_CROP_PX","128")))
GOLDEN_FRAME_CONFIDENCE_MIN = float(os.getenv("GOLDEN_FRAME_CONFIDENCE_MIN","0.55"))
GOLDEN_FRAME_CROP_DIR       = os.getenv("GOLDEN_FRAME_CROP_DIR", "data/golden_crops")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. FINANCIAL INTELLIGENCE (Batch 4)
# ═══════════════════════════════════════════════════════════════════════════════

# Simulates a Mitchell/Audatex industry-grade parts & labour database.
# Format: part_name → {replace: USD, paint: USD, labor_per_hour: USD}
REPAIR_DATABASE: dict[str, dict[str, float]] = {
    "Front Bumper":       {"replace": 450,  "paint": 200,  "labor_per_hour": 85},
    "Rear Bumper":        {"replace": 420,  "paint": 175,  "labor_per_hour": 85},
    "Side Bumper":        {"replace": 320,  "paint": 140,  "labor_per_hour": 80},
    "Door Panel":         {"replace": 1200, "paint": 350,  "labor_per_hour": 95},
    "Engine Hood":        {"replace": 900,  "paint": 280,  "labor_per_hour": 90},
    "Rear Boot":          {"replace": 850,  "paint": 260,  "labor_per_hour": 88},
    "Left Headlight":     {"replace": 800,  "paint": 0,    "labor_per_hour": 85},
    "Right Headlight":    {"replace": 800,  "paint": 0,    "labor_per_hour": 85},
    "Side Mirror":        {"replace": 350,  "paint": 80,   "labor_per_hour": 75},
    "Lower Front Bumper": {"replace": 280,  "paint": 120,  "labor_per_hour": 75},
    "Quarter Panel":      {"replace": 1100, "paint": 320,  "labor_per_hour": 95},
    "Roof Panel":         {"replace": 1800, "paint": 400,  "labor_per_hour": 100},
    "_default":           {"replace": 500,  "paint": 150,  "labor_per_hour": 80},
}

# Severity → recommended repair action
SEVERITY_TO_ACTION: dict[str, str] = {
    "Minor":    "REPAIR/PAINT",
    "Moderate": "REPAIR/PAINT",
    "Severe":   "REPLACE",
    "Critical": "REPLACE",
}

VEHICLE_VALUE        = float(os.getenv("VEHICLE_VALUE",        "15000"))  # USD default
TOTAL_LOSS_THRESHOLD = float(os.getenv("TOTAL_LOSS_THRESHOLD", "0.75"))   # >75% → TOTALED
USD_TO_INR           = float(os.getenv("USD_TO_INR",           "83"))     # display conversion


# ═══════════════════════════════════════════════════════════════════════════════
# 9. GRADIO / UI
# ═══════════════════════════════════════════════════════════════════════════════

GRADIO_SHARE       = os.getenv("GRADIO_SHARE", "true").lower() in ("1", "true", "yes")
GRADIO_DEBUG       = os.getenv("GRADIO_DEBUG", "false").lower() in ("1", "true", "yes")
GRADIO_THEME       = os.getenv("GRADIO_THEME", "dark")
GRADIO_VERSION_TAG = "v36"
USER_PORT          = int(os.getenv("USER_PORT",    "7860"))
AUDITOR_PORT       = int(os.getenv("AUDITOR_PORT", "7861"))


# ═══════════════════════════════════════════════════════════════════════════════
# 10. SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPTS: dict[str, str] = {

    # ── Gemini: vehicle classification + damage discovery (Task 1 & 4) ─────────
    "gemini_intake": (
        "You are a vehicle forensics AI embedded inside an autonomous insurance "
        "claims pipeline. Your job is to analyse a photograph of a vehicle and "
        "return a structured JSON response — no markdown, no prose, JSON only.\n\n"
        "Required fields:\n"
        "  vehicle_type        : 'car' | '2W' | '3W' | 'unknown'\n"
        "  vehicle_type_confidence : float 0.0–1.0\n"
        "  vehicle_make_estimate   : e.g. 'sedan-class', 'hatchback', 'SUV'\n"
        "  discovered_damages  : list of {type, location, severity, confidence}\n\n"
        "Damage types: Scratch | Dent | Cracked | Broken part | Missing part | "
        "Paint chip | Flaking | Corrosion\n"
        "Severity:     Minor | Moderate | Severe | Critical\n"
        "Confidence:   0.0–1.0"
    ),

    # ── Gemini: Golden Frame deep-look per crop (Batch 3) ──────────────────────
    "gemini_golden_frame": (
        "You are a precision vehicle damage verification AI. "
        "You are given a tightly cropped image of one specific damage region "
        "detected by a computer-vision model.\n\n"
        "Verify whether the damage is genuine and return JSON only:\n"
        "  confirmed      : true | false\n"
        "  damage_type    : string (from the standard taxonomy)\n"
        "  severity       : Minor | Moderate | Severe | Critical\n"
        "  confidence     : float 0.0–1.0\n"
        "  reasoning      : one sentence\n\n"
        "Be conservative — if the crop is ambiguous, set confirmed=false."
    ),

    # ── Groq: full claim report generation ─────────────────────────────────────
    "groq_report": (
        "You are SmartForge Report Writer, an expert insurance claims analyst AI. "
        "You receive a structured JSON object containing all pipeline outputs "
        "for a vehicle damage claim and must produce a professional, concise "
        "insurance adjuster report.\n\n"
        "Format rules:\n"
        "- Plain text only — no markdown, no bullets, no headers.\n"
        "- Three paragraphs maximum.\n"
        "- Para 1: vehicle & incident summary.\n"
        "- Para 2: damage findings, severity, and financial estimate.\n"
        "- Para 3: recommendation (Approve / Reject / Escalate) and justification."
    ),

    # ── Groq: conversational chat assistant (User Dashboard Tab 5) ─────────────
    "groq_chat": (
        "You are the SmartForge Claims Assistant, a helpful, professional AI "
        "embedded in an insurance claims portal. You have access to this user's "
        "claim analysis results (provided in the first message). "
        "Answer questions clearly and concisely. "
        "Never fabricate figures not present in the provided data. "
        "If unsure, say so and suggest the user contact their insurer."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 11. QUICK SANITY CHECK (run: python -m src.config.settings)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import textwrap

    print("\n" + "═" * 60)
    print("  SmartForge Settings — environment check")
    print("═" * 60)

    checks = [
        ("GEMINI_API_KEY",   GEMINI_ENABLED,    "gemini enabled"),
        ("GROQ_API_KEY",     GROQ_ENABLED,      "groq enabled"),
        ("SERPAPI_KEY",      SERPAPI_ENABLED,   "serpapi enabled"),
        ("WINSTON_AI_KEY",   WINSTON_AI_ENABLED,"winston ai enabled"),
        ("MONGO_URI",        bool(MONGO_URI),   "mongodb configured"),
        ("DAMAGE_MODEL",     True,              DAMAGE_MODEL_PATH),
        ("PART_MODEL",       True,              PART_MODEL_PATH),
    ]

    for label, flag, detail in checks:
        icon = "✅" if flag else "⚠️ "
        print(f"  {icon}  {label:<22} {detail}")

    print("═" * 60 + "\n")
