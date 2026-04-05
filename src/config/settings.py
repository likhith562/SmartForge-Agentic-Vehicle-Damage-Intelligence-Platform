"""
SmartForge — Configuration & Constants
=======================================
Single source of truth for every threshold, path, and API parameter used
across the pipeline.  All values can be overridden via environment variables
or a .env file (loaded automatically when python-dotenv is installed).

Usage:
    from src.config import Settings
    cfg = Settings()
    print(cfg.SAHI_CONFIDENCE)
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any


# ── Optional dotenv support ───────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — rely on os.environ directly


def _env(key: str, default):
    """Read an env-var and cast to the same type as *default*."""
    val = os.environ.get(key)
    if val is None:
        return default
    if isinstance(default, bool):
        return val.lower() in ("1", "true", "yes")
    if isinstance(default, float):
        return float(val)
    if isinstance(default, int):
        return int(val)
    return val


# ── Repair / Replace cost database ───────────────────────────────────────────
# Simulates a Mitchell / Audatex industry-grade parts & labour database.
# Format: part_name → {replace: USD, paint: USD, labor_per_hour: USD}
# Costs are in USD; converted to INR for display using USD_TO_INR.
REPAIR_DATABASE: Dict[str, Dict[str, float]] = {
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

# Severity → repair action mapping
# Minor / Moderate → cosmetic repair/repaint
# Severe / Critical → full part replacement
SEVERITY_TO_ACTION: Dict[str, str] = {
    "Minor":    "REPAIR/PAINT",
    "Moderate": "REPAIR/PAINT",
    "Severe":   "REPLACE",
    "Critical": "REPLACE",
}

# Part name normalisation map (YOLO class names → human-readable)
PART_NAME_MAP: Dict[str, str] = {
    "BUMPER--F":    "Front Bumper",
    "BUMPER--R":    "Rear Bumper",
    "BUMPER--S":    "Side Bumper",
    "DOOR":         "Door Panel",
    "HOOD":         "Engine Hood",
    "BOOT":         "Rear Boot",
    "HEADLIGHT--L": "Left Headlight",
    "HEADLIGHT--R": "Right Headlight",
    "MIRROR":       "Side Mirror",
}

# Spatial zone → natural language descriptions (used when part model misses)
ZONE_LANGUAGE_MAP: Dict[str, str] = {
    "Front Center Upper Section":  "at the top of the front fascia",
    "Front Center Main Body":      "across the front centre panel",
    "Front Center Lower Section":  "near the front lower grille area",
    "Front Left Upper Section":    "at the upper front-left corner",
    "Front Left Main Body":        "on the front-left body panel",
    "Front Left Lower Section":    "near the front-left wheel area",
    "Front Right Upper Section":   "at the upper front-right corner",
    "Front Right Main Body":       "on the front-right body panel",
    "Front Right Lower Section":   "near the front-right wheel area",
    "Middle Left Main Body":       "on the left side panel",
    "Middle Right Main Body":      "on the right side panel",
    "Rear Center Main Body":       "across the rear centre panel",
    "Rear Left Main Body":         "on the rear-left body panel",
    "Rear Right Main Body":        "on the rear-right body panel",
    "Rear Center Lower Section":   "near the rear lower bumper area",
}

# Severity → cost table (INR ranges, used for quick estimates pre-financial engine)
COST_TABLE: Dict[tuple, tuple] = {
    ("Scratch",      "Low"):    ("₹500–₹1,500",    "Polish & touch-up"),
    ("Scratch",      "Medium"): ("₹1,500–₹4,000",  "Repaint panel section"),
    ("Scratch",      "High"):   ("₹4,000–₹8,000",  "Full panel repaint"),
    ("Dent",         "Low"):    ("₹1,000–₹3,000",  "Paintless dent removal"),
    ("Dent",         "Medium"): ("₹3,000–₹7,000",  "Panel beating + repaint"),
    ("Dent",         "High"):   ("₹7,000–₹18,000", "Panel replacement"),
    ("Cracked",      "High"):   ("₹8,000–₹25,000", "Part replacement"),
    ("Broken part",  "High"):   ("₹5,000–₹40,000", "Component replacement"),
    ("Missing part", "High"):   ("₹5,000–₹40,000", "Component replacement"),
    ("Paint chip",   "Low"):    ("₹300–₹800",       "Spot touch-up"),
    ("Paint chip",   "Medium"): ("₹800–₹2,500",     "Spot repaint"),
    ("Flaking",      "Medium"): ("₹2,000–₹6,000",  "Strip & repaint"),
    ("Corrosion",    "Medium"): ("₹3,000–₹9,000",  "Rust treatment + repaint"),
    ("Corrosion",    "High"):   ("₹9,000–₹25,000", "Panel replacement"),
}
DEFAULT_COST: tuple = ("₹1,000–₹5,000", "Inspect and repair")


@dataclass
class Settings:
    """
    Fully-resolved configuration object.  Reads from environment variables
    first, then falls back to the defaults defined below.

    Instantiate once at app startup and pass around:
        cfg = Settings()
    """

    # ── Model paths ──────────────────────────────────────────────────────────
    DAMAGE_MODEL_PATH: str = field(
        default_factory=lambda: _env("DAMAGE_MODEL_PATH", "/content/seg-best.pt"))
    PART_MODEL_PATH: str = field(
        default_factory=lambda: _env("PART_MODEL_PATH", "/content/detect-best.pt"))
    SAM_CHECKPOINT: str = field(
        default_factory=lambda: _env("SAM_CHECKPOINT", "/content/sam_vit_b_01ec64.pth"))
    SAM_URL: str = (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    )

    # ── API keys ─────────────────────────────────────────────────────────────
    GEMINI_API_KEY: str = field(
        default_factory=lambda: _env("GEMINI_API_KEY", ""))
    GROQ_API_KEY: str = field(
        default_factory=lambda: _env("GROQ_API_KEY", ""))
    SERPAPI_KEY: str = field(
        default_factory=lambda: _env("SERPAPI_KEY", ""))
    WINSTON_AI_KEY: str = field(
        default_factory=lambda: _env("WINSTON_AI_KEY", ""))

    # ── Database ─────────────────────────────────────────────────────────────
    MONGO_URI: str = field(
        default_factory=lambda: _env("SMARTFORGE_MONGO_URI", ""))
    SQLITE_PATH: str = "/content/smartforge_claims.db"

    # ── Gemini / Groq models ──────────────────────────────────────────────────
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_FALLBACK_MODEL: str = "gemini-2.5-flash-lite"
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # ── SAHI / perception ────────────────────────────────────────────────────
    SAHI_CONFIDENCE: float = field(
        default_factory=lambda: _env("SAHI_CONFIDENCE", 0.3))
    SAHI_SLICE_SIZE: int = field(
        default_factory=lambda: _env("SAHI_SLICE_SIZE", 640))
    SAHI_OVERLAP: float = field(
        default_factory=lambda: _env("SAHI_OVERLAP", 0.2))
    CONFIDENCE_RECHECK_LIMIT: float = 0.45

    # ── Agentic pipeline thresholds ──────────────────────────────────────────
    MAX_RETRIES: int = field(
        default_factory=lambda: _env("MAX_RETRIES", 2))
    AUTO_APPROVE_THRESHOLD: int = field(
        default_factory=lambda: _env("AUTO_APPROVE_THRESHOLD", 85))
    ESCALATION_THRESHOLD: int = field(
        default_factory=lambda: _env("ESCALATION_THRESHOLD", 70))
    HEALTH_SCORE_MIN: float = 0.6

    # ── Fraud layer ───────────────────────────────────────────────────────────
    BYPASS_FRAUD: bool = field(
        default_factory=lambda: _env("BYPASS_FRAUD", True))
    MAX_FRAUD_RETRIES: int = field(
        default_factory=lambda: _env("MAX_FRAUD_RETRIES", 3))
    FRAUD_TRUST_THRESHOLD: int = field(
        default_factory=lambda: _env("FRAUD_TRUST_THRESHOLD", 40))
    FRAUD_GPS_MAX_DISTANCE_KM: float = field(
        default_factory=lambda: _env("FRAUD_GPS_MAX_DISTANCE_KM", 50.0))
    PHASH_HAMMING_THRESHOLD: int = field(
        default_factory=lambda: _env("PHASH_HAMMING_THRESHOLD", 8))
    WINSTON_AI_THRESHOLD: float = field(
        default_factory=lambda: _env("WINSTON_AI_THRESHOLD", 0.70))
    FRAUD_HASH_DB_PATH: str = field(
        default_factory=lambda: _env("FRAUD_HASH_DB_PATH", "/content/fraud_hash_db.json"))

    # ── Financial intelligence ────────────────────────────────────────────────
    VEHICLE_VALUE: float = field(
        default_factory=lambda: _env("VEHICLE_VALUE", 15000.0))
    TOTAL_LOSS_THRESHOLD: float = field(
        default_factory=lambda: _env("TOTAL_LOSS_THRESHOLD", 0.75))
    USD_TO_INR: float = field(
        default_factory=lambda: _env("USD_TO_INR", 83.0))

    # ── Golden Frame Verification (Batch 3) ───────────────────────────────────
    GOLDEN_FRAME_CROP_MARGIN: float = field(
        default_factory=lambda: _env("GOLDEN_FRAME_CROP_MARGIN", 0.25))
    GOLDEN_FRAME_MIN_CROP_PX: int = field(
        default_factory=lambda: _env("GOLDEN_FRAME_MIN_CROP_PX", 128))
    GOLDEN_FRAME_CONFIDENCE_MIN: float = field(
        default_factory=lambda: _env("GOLDEN_FRAME_CONFIDENCE_MIN", 0.55))
    GOLDEN_FRAME_CROP_DIR: str = field(
        default_factory=lambda: _env("GOLDEN_FRAME_CROP_DIR", "/content/golden_crops"))

    # ── False Positive Gate thresholds ────────────────────────────────────────
    NON_CAR_CONF_THRESHOLD: float = 0.60
    MIN_AREA_NON_CAR: float = 0.003
    MIN_DEFORMATION_GATE: float = 0.001
    FLAT_AREA_GATE: float = 0.005

    # ── Output paths ─────────────────────────────────────────────────────────
    AUDIT_LOG_PATH: str = "/content/audit_log.json"
    ESCALATION_PATH: str = "/content/escalation_record.json"
    CHECKPOINT_DUMP_PATH: str = "/content/checkpoint_dump.json"

    # ── Gradio UI ─────────────────────────────────────────────────────────────
    GRADIO_APP_TITLE: str = "🚗 SmartForge — Agentic Vehicle Damage Intelligence"
    GRADIO_APP_SUBTITLE: str = "Autonomous Insurance Claims · LangGraph + Gemini VLM + Groq"
    GRADIO_VERSION_TAG: str = "v1.0"
    GRADIO_THEME: str = "soft"
    GRADIO_SHARE: bool = True
    GRADIO_DEBUG: bool = False

    # ── Claim metadata (runtime defaults — overridden per-claim via UI) ───────
    CLAIM_ACCIDENT_DATE: str = ""
    CLAIM_LOSS_LOCATION_LAT: float = 0.0
    CLAIM_LOSS_LOCATION_LON: float = 0.0
    VEHICLE_ID: str = ""
    IMAGE_ID: str = ""
    POLICY_ID: str = ""

    # ── Expose shared constant dicts ──────────────────────────────────────────
    REPAIR_DATABASE: Dict[str, Any] = field(
        default_factory=lambda: REPAIR_DATABASE)
    SEVERITY_TO_ACTION: Dict[str, str] = field(
        default_factory=lambda: SEVERITY_TO_ACTION)
    PART_NAME_MAP: Dict[str, str] = field(
        default_factory=lambda: PART_NAME_MAP)
    ZONE_LANGUAGE_MAP: Dict[str, str] = field(
        default_factory=lambda: ZONE_LANGUAGE_MAP)
    COST_TABLE: Dict[tuple, tuple] = field(
        default_factory=lambda: COST_TABLE)
    DEFAULT_COST: tuple = field(
        default_factory=lambda: DEFAULT_COST)

    # ── Derived flags (computed from keys) ───────────────────────────────────
    @property
    def GEMINI_ENABLED(self) -> bool:
        return bool(self.GEMINI_API_KEY)

    @property
    def GROQ_ENABLED(self) -> bool:
        return bool(self.GROQ_API_KEY)

    @property
    def SERPAPI_ENABLED(self) -> bool:
        return bool(self.SERPAPI_KEY)

    @property
    def WINSTON_AI_ENABLED(self) -> bool:
        return bool(self.WINSTON_AI_KEY)

    def summary(self) -> str:
        """Print a human-readable startup summary."""
        lines = [
            "SmartForge Configuration",
            f"  Gemini    : {'✅ ' + self.GEMINI_MODEL if self.GEMINI_ENABLED else '⚠️  disabled'}",
            f"  Groq      : {'✅ ' + self.GROQ_MODEL if self.GROQ_ENABLED else '⚠️  disabled'}",
            f"  MongoDB   : {'✅ configured' if self.MONGO_URI else '⚠️  SQLite fallback'}",
            f"  SerpAPI   : {'✅ enabled' if self.SERPAPI_ENABLED else '—  disabled'}",
            f"  Winston AI: {'✅ enabled' if self.WINSTON_AI_ENABLED else '—  disabled (ELA fallback)'}",
            f"  Bypass    : {'BYPASS_FRAUD=True (demo mode)' if self.BYPASS_FRAUD else 'Full fraud layer active'}",
        ]
        return "\n".join(lines)


# ── Module-level singleton (import-time) ──────────────────────────────────────
# Use this in most places:  from src.config.settings import cfg
cfg = Settings()
