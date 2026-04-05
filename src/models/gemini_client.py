"""
SmartForge — Gemini VLM Client
================================
Thin wrapper around the Google GenAI SDK (google-genai ≥ 0.8).
Provides a single call_gemini() function used by every pipeline node
that needs multimodal intelligence.

Features
--------
- Uses gemini-2.5-flash as primary model (free tier, multimodal)
- Automatically falls back to gemini-2.5-flash-lite on 429 / rate-limit
- Strips markdown fences from response text before JSON parsing
- Returns {"_error": "..."} on any failure so callers degrade gracefully
- Prints concise [DEBUG] lines so Colab output stays readable

Usage
-----
    from src.models.gemini_client import call_gemini

    result = call_gemini(
        prompt     = "Is this vehicle damaged? Return JSON only.",
        image_path = "/content/car.jpg",
        schema_hint = '{"damaged": true_or_false}',
    )
    if "_error" not in result:
        print(result["damaged"])
"""

import json
from typing import Any, Dict

from src.config.settings import cfg


def call_gemini(
    prompt: str,
    image_path: str,
    schema_hint: str = "",
) -> Dict[str, Any]:
    """
    Send *image_path* + *prompt* to Gemini and return a parsed JSON dict.

    Parameters
    ----------
    prompt      : str  — task description for the model
    image_path  : str  — path to a JPEG/PNG file on disk
    schema_hint : str  — optional JSON schema snippet embedded in the prompt
                         to guide structured output

    Returns
    -------
    dict  — parsed JSON response from Gemini, or {"_error": "..."} on failure

    Notes
    -----
    - The model is instructed to return ONLY valid JSON — no markdown, no prose.
    - If the response is wrapped in ```json … ``` fences they are stripped.
    - On a 429 / rate-limit the function retries once with GEMINI_FALLBACK_MODEL.
    """
    if not cfg.GEMINI_ENABLED:
        return {"_error": "Gemini disabled — GEMINI_API_KEY not set"}

    try:
        from google import genai as _genai
        from google.genai import types as _types
    except ImportError:
        return {"_error": "google-genai not installed — run: pip install google-genai"}

    # ── Build the full prompt ─────────────────────────────────────────────────
    full_prompt = (
        f"{prompt}\n\n"
        "IMPORTANT: Respond ONLY with valid JSON. "
        "No markdown fences, no preamble, no explanation.\n"
        f"Expected schema: {schema_hint}"
    )

    client = _genai.Client(api_key=cfg.GEMINI_API_KEY)

    with open(image_path, "rb") as fh:
        image_bytes = fh.read()

    def _try(model_name: str) -> Dict[str, Any]:
        print(f"[DEBUG] 🚀 Gemini call → {model_name}")
        response = client.models.generate_content(
            model    = model_name,
            contents = [
                _types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                full_prompt,
            ],
        )
        text = response.text.strip()
        print(f"[DEBUG] 🟢 Response ({len(text)} chars):\n{text[:400]}\n")

        # Strip markdown fences if present
        if text.startswith("```"):
            parts = text.split("```")
            text  = parts[1] if len(parts) > 1 else text
            if text.startswith("json"):
                text = text[4:]

        return json.loads(text.strip())

    # ── Primary model attempt ─────────────────────────────────────────────────
    try:
        return _try(cfg.GEMINI_MODEL)

    except Exception as exc:
        err_str = str(exc).lower()
        is_rate_limit = any(
            k in err_str for k in ("429", "rate", "quota", "resource_exhausted")
        )

        if is_rate_limit:
            print(
                f"[DEBUG] ⚠️  Rate limit on {cfg.GEMINI_MODEL} "
                f"— retrying with {cfg.GEMINI_FALLBACK_MODEL}"
            )
            # ── Fallback model attempt ────────────────────────────────────────
            try:
                return _try(cfg.GEMINI_FALLBACK_MODEL)
            except Exception as exc2:
                print(f"[DEBUG] 🔴 Fallback {cfg.GEMINI_FALLBACK_MODEL} failed: {exc2}")
                return {"_error": str(exc2)}

        print(f"[DEBUG] 🔴 Gemini FAILED: {exc}")
        return {"_error": str(exc)}
