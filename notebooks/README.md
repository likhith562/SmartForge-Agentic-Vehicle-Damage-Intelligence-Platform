# 📓 SmartForge Notebook — Gradio Live Demo Guide

> **Open this notebook in Google Colab to run the full agentic pipeline and launch both interactive Gradio dashboards with public share links — no local setup required.**
>
> `Vehicle_Damage_Agentic_AI_v36_gradio.ipynb` · LangGraph + Gemini VLM + Groq · Autonomous Insurance Claims Processing

[![Open In Colab — Gradio Live Demo](https://img.shields.io/badge/Open%20In%20Colab%20%7C%20Gradio%20Live%20Demo-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/your-username/smartforge-agentic-ai/blob/main/notebooks/Vehicle_Damage_Agentic_AI_v36_gradio.ipynb)

---

## What the Notebook Launches

Running all cells produces **two public Gradio share links** (valid for 1 week each), printed at the bottom of Cell G4:

| App | Port | URL format | Audience |
|-----|------|-----------|----------|
| **User Dashboard** | 7860 | `https://xxxxxxxx.gradio.live` | Vehicle owner / claimant |
| **Auditor Dashboard** | 7861 | `https://yyyyyyyy.gradio.live` | Insurance adjuster / auditor |

Share the respective link with each audience — no authentication is required in the demo setup.

---

## Overview

SmartForge is a fully agentic, end-to-end insurance claims processing pipeline built on **LangGraph**. It accepts one or more vehicle damage photos and autonomously:

1. Detects and segments damage using a custom YOLO model + SAM
2. Estimates depth deformation with MiDaS
3. Runs a **5-check fraud detection** layer (EXIF temporal, GPS consistency, software integrity, perceptual hash, AI/screen-detection)
4. Enriches detections with **Gemini VLM** (vehicle type, part location, low-confidence verification)
5. Verifies damage with **Golden Frame forensic crops** sent back to Gemini at full resolution
6. Classifies severity, generates line-item cost estimates, and issues an insurance ruling
7. Produces AI-written damage narratives via **Groq**
8. Surfaces results in **two Gradio dashboards** — a claimant-facing portal and a full auditor/admin console

---

## Architecture

```
intake ──► fraud ──┬──► map_images ──► cv_worker(×N) ──► fusion   ──►┐
(single-image)     │   (multi-image Send fan-out)                    │
                   │                                                 ▼
                   └──────────────────────────────────────► perception
                                                                     │
                                                              gemini_agent
                                                                     │
                                                        false_positive_gate
                                                                     │
                                                           health_monitor
                                                           ┌─────────┘
                                                    (retry?│circuit breaker)
                                                           ▼
                                                  verification_v2_node  ←  Golden Frame
                                                           │
                                                     reasoning_node     ← cost estimation
                                                           │
                                                    [INTERRUPT]
                                                     decision_node      ← claims ruling
                                                           │
                                                      report_node       ← Groq narratives
                                                           │
                                                          END
```

**Key design patterns:**

| Pattern | Where used |
|---------|-----------|
| LangGraph cyclic graph with circuit breaker | `health_monitor` → `perception_retry` loop |
| LangGraph `Send` API for parallel fan-out | `map_images_node` — one worker per uploaded photo |
| Human-in-the-Loop interrupt | `interrupt_before=["decision"]` on high-value claims |
| NetworkX in-memory graph DB | `fusion_node` — cross-image damage deduplication |
| MemorySaver checkpointing | Full state dump at every super-step |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Runtime** | Google Colab with **GPU** (T4 or better) — `Runtime → Change runtime type → T4 GPU` |
| **Python** | 3.10+ (Colab default) |
| **Disk space** | ~3 GB for model weights + dependencies |
| **API access** | Gemini API key, Groq API key (see API Keys section below) |

---

## API Keys & Secrets

SmartForge uses **Google Colab Secrets** to avoid hardcoding credentials. All keys are read at runtime via `google.colab.userdata`.

### Required Keys

| Secret Name | Where to get it | Used for |
|---|---|---|
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com/app/apikey) | Gemini VLM — vehicle classification, damage verification, Golden Frame forensics |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com/keys) | Groq — fast AI-generated damage narrative text, auditor bot, user chat |

### Optional Keys

| Secret Name | Purpose | Fallback behaviour |
|---|---|---|
| `SERPAPI_KEY` | Reverse image search in fraud layer (Google Lens) | Fraud check skipped silently with warning |
| `WINSTON_AI_KEY` | AI-generated image detection Stage 1 (cloud, highest accuracy) | Falls back to ELA forensics (Stage 2) → Laplacian variance (Stage 3) |
| `SMARTFORGE_MONGO_URI` | MongoDB Atlas connection string | Auto-falls back to local SQLite at `/content/smartforge_claims.db` |

### How to Add Secrets in Google Colab

> **Colab Secrets** are the correct way to store credentials — never paste API keys directly into notebook cells.

1. Open your notebook in Google Colab
2. Click the **🔑 Key icon** in the left sidebar (or go to `Tools → Secrets`)
3. Click **+ Add new secret**
4. Enter the **Name** exactly as shown in the table above (e.g. `GEMINI_API_KEY`)
5. Paste the key value into the **Value** field
6. Toggle **Notebook access** to **ON** for each secret
7. Repeat for each key

The notebook reads them automatically:

```python
from google.colab import userdata
GEMINI_API_KEY = userdata.get("GEMINI_API_KEY")
GROQ_API_KEY   = userdata.get("GROQ_API_KEY")
```

> ⚠️ Secrets are **per-account and per-notebook**. If you share a copy of the notebook, the recipient must add their own secrets.

---

## Model Files

| File | Source | What it does |
|---|---|---|
| `seg-best.pt` | ✅ Included in repo — `notebooks/models/seg-best.pt` | Custom YOLO damage segmentation model |
| `detect-best.pt` | ✅ Included in repo — `notebooks/models/detect-best.pt` | Custom YOLO vehicle part detection model |
| `sam_vit_b_01ec64.pth` | ⬇️ Auto-downloaded from Meta at runtime | Meta's Segment Anything Model (SAM ViT-B) — ~375 MB |

### How models are loaded

**`seg-best.pt` and `detect-best.pt`** ship with this repository inside the `notebooks/models/` folder. Copy them to `/content/` before running Cell 1:

```python
# Option A — direct copy if models were uploaded via Colab Files panel
!cp notebooks/models/seg-best.pt    /content/seg-best.pt
!cp notebooks/models/detect-best.pt /content/detect-best.pt
```

Or if you mounted Google Drive and cloned there:

```python
import shutil
shutil.copy("/content/drive/MyDrive/smartforge/models/seg-best.pt",    "/content/seg-best.pt")
shutil.copy("/content/drive/MyDrive/smartforge/models/detect-best.pt", "/content/detect-best.pt")
```

**`sam_vit_b_01ec64.pth`** is **not** stored in the repo (it is 375 MB). The notebook downloads it automatically from Meta's servers during Cell 7:

```python
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
```

No action needed — if the file is missing from `/content/`, the pipeline fetches it before the CV stack starts. Subsequent runs in the same Colab session reuse the cached file.

---

## Quick Start (Google Colab)

```
1.  Clone the repo into your Google Drive (or open directly in Colab via GitHub)
2.  Runtime → Change runtime type → T4 GPU
3.  Add Colab Secrets:  GEMINI_API_KEY  and  GROQ_API_KEY  (see above)
4.  Copy model weights:
        !cp notebooks/models/seg-best.pt    /content/seg-best.pt
        !cp notebooks/models/detect-best.pt /content/detect-best.pt
5.  Runtime → Run all
        ↳ SAM weights (~375 MB) are downloaded automatically — no action needed
        ↳ Cell 2 installs all packages — ~3–5 minutes on first run
6.  Wait for Cell G4 to print public share URLs (≈ 5–8 minutes total first run)
7.  Open the User Dashboard URL in any browser
    Share the Auditor Dashboard URL with the adjuster/admin
```

On subsequent runs within the same Colab session (packages cached), skip step 4 and go straight to `Runtime → Run all` — takes ~2–3 minutes.

---

## Cell-by-Cell Guide

### Setup Cells

| Cell | Name | What it does | Must edit? |
|------|------|-------------|-----------|
| **Cell 0** | Drive Mount | Mounts Google Drive for model/log persistence across sessions | Optional |
| **Cell 1** | Configuration | All runtime parameters — model paths, thresholds, API key names | ✏️ **Yes** (verify model paths) |
| **Cell 2** | Install Dependencies | `pip install` for all packages — run once per session (~3–5 min) | No |
| **Cell 3** | Imports & Env Check | Validates GPU availability, imports all libraries | No |

> **Cell 1 is the only cell you need to edit.** All pipeline and dashboard cells (4–G4) run without modification.

### Pipeline Cells

| Cell | Name | What it does |
|------|------|-------------|
| **Cell 4** | State Schema | Defines `SmartForgeState` TypedDict — single source of truth for all agents; `make_initial_state()` factory |
| **Cell 5** | Helpers & Cost Engine | `PART_NAME_MAP`, `ZONE_LANGUAGE_MAP`, `COST_TABLE`, `REPAIR_DATABASE`; `estimate_cost()`, `compute_severity()`, `compute_iou()`, `_log_msg()` |
| **Cell 6** | `intake` node | File validation, resolution correction (upscale < 224px, cap > 4096px), `_analyse_image_conditions()` — V-channel variance → adaptive SAHI confidence + skip_downsampling flag |
| **Cell 6b** | `fraud` node | 5-check fraud & integrity layer: EXIF temporal, GPS Haversine, software source, pHash duplicate, screen/AI detection; `human_audit_node`, `fraud_router` |
| **Cell 6c** | Batch 2 Map-Reduce | `map_images_node` (Send API fan-out), `cv_worker_node` (lightweight per-image SAHI), `fusion_node` (NetworkX DiGraph build + part-based deduplication + recycling detection) |
| **Cell 6d** | `verification_v2` | Batch 3 Golden Frame: `get_high_res_crop()`, `_save_crop_to_tmp()`, `_call_gemini_with_crop()` with VERIFICATION_SCHEMA; multi-angle secondary crop logic |
| **Cell 7** | `perception` node | SAHI → SAM (SamPredictor.predict) → MiDaS (depth + deformation index) → YOLO part detection → `raw_detections` assembly; reads `adaptive_sahi_conf` and `retry_count` from state; frees `image_bgr` after use |
| **Cell 7b** | `gemini_agent` | Gemini VLM enrichment (3 batched API calls): Call A — vehicle type; Call B — location enrichment + low-conf verification (1 call, not N+M); Call C — full-image missed-damage scan |
| **Cell 7c** | `false_positive_gate` | 4-gate false-positive filter: Gate 0 (gemini_discovery bypass), Gates 1–4 (conf floor, area floor, depth flatness, Gemini veto); Gemini positive override for FLAT_SURFACE |
| **Cell 8** | `health_monitor` | 2 validation checks: (1) area_ratio + deformation bounds, (2) CV-only confidence variance ≤ 0.08; `pipeline_stability_flag`; `health_monitor_router` conditional edge |
| **Cell 9** | `perception_retry` | Circuit-breaker retry wrapper — increments `retry_count`, delegates to `perception_node` |
| **Cell 10** | `reasoning` node | Severity classification, Batch 4 financial intelligence: `_get_repair_data()`, REPAIR/REPLACE logic, total_loss_flag, confirmed vs conservative score |
| **Cell 11** | `decision` node | Fraud flag check, CLM_PENDING / CLM_WORKSHOP / CLM_MANUAL ruling, HITL `interrupt_before=["decision"]` |
| **Cell 12** | `report` node | `generate_groq_narrative()` — 3-section report (Executive Summary + Forensic Integrity + Detailed Estimate); CONFIRMED_STATUSES matching; score normalization; saves `/content/final_output.json` |
| **Cell 13** | Build LangGraph | Assembles all nodes, fixed edges, conditional edges; compiles with `MemorySaver` checkpointer; `interrupt_before=["decision"]` |

### Dashboard Cells

| Cell | Name | What it does |
|------|------|-------------|
| **Cell G1** | Dashboard Config | `GRADIO_APP_TITLE`, `GRADIO_THEME`, `GRADIO_SHARE`, `MONGO_URI`; reads `SMARTFORGE_MONGO_URI` env var |
| **Cell G2** | Database Layer | MongoDB Atlas connection attempt (4s timeout) → SQLite fallback; defines `db_upsert` (hybrid write), `db_get`, `db_find`, `db_count`, `db_mark_auditor`, `db_backend_info`; SQLite schema with all JSON fields |
| **Cell G3** | User Dashboard | Builds `user_demo` gr.Blocks — 5 tabs (Intake, Insurance Preference, Analysis, Summary, AI Assistant); Leaflet iframe, HTML5 date picker, 3-strike fraud enforcement, deferred `.then()` wiring; **does NOT launch yet** |
| **Cell G4** | Auditor Dashboard + Launch | Builds `auditor_demo` — 5 tabs + gr.Sidebar AI bot; auto-load handlers; **launches both apps** → two public share links printed |

---

## Configuration Reference

All user-editable settings live in **Cell 1**. Edit only this cell.

```python
# ── Model paths ─────────────────────────────────────────────────────────────
DAMAGE_MODEL_PATH        = "/content/seg-best.pt"
PART_MODEL_PATH          = "/content/detect-best.pt"
SAM_CHECKPOINT           = "/content/sam_vit_b_01ec64.pth"
SAM_URL                  = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

# ── SAHI inference settings ──────────────────────────────────────────────────
SAHI_CONFIDENCE          = 0.3    # Base confidence (auto-adjusted per image)
SAHI_SLICE_SIZE          = 640    # Tile size for sliced inference
SAHI_OVERLAP             = 0.2    # Tile overlap ratio

# ── Claim metadata (set per job or leave blank) ──────────────────────────────
VEHICLE_ID               = ""
IMAGE_ID                 = ""
POLICY_ID                = ""

# ── Agentic thresholds ────────────────────────────────────────────────────────
MAX_RETRIES              = 2      # Max perception retries before circuit-breaker
ESCALATION_THRESHOLD     = 70     # Vehicle health score below this → CLM_WORKSHOP
CONFIDENCE_RECHECK_LIMIT = 0.45   # YOLO confidence below this → Gemini re-verification
AUTO_APPROVE_THRESHOLD   = 85     # Retained for reference; AI never auto-approves
HEALTH_SCORE_MIN         = 0.6

# ── Gemini & Groq models ──────────────────────────────────────────────────────
GEMINI_MODEL             = "gemini-2.5-flash"
GEMINI_FALLBACK_MODEL    = "gemini-2.5-flash-lite"   # auto-used on 429 rate limit
GROQ_MODEL               = "llama-3.3-70b-versatile"

# ── Fraud detection ────────────────────────────────────────────────────────────
FRAUD_TRUST_THRESHOLD    = 40     # Below → SUSPICIOUS_HIGH_RISK
FRAUD_GPS_MAX_DISTANCE_KM= 50.0   # Max GPS drift (km)
BYPASS_FRAUD             = True   # Set False for production
MAX_FRAUD_RETRIES        = 3      # 3-strike tolerance
WINSTON_AI_THRESHOLD     = 0.70   # AI probability above this → flagged
PHASH_HAMMING_THRESHOLD  = 8      # Hamming distance ≤ 8 → near-duplicate
FRAUD_HASH_DB_PATH       = "/content/fraud_hash_db.json"

# ── Golden Frame (Batch 3) ─────────────────────────────────────────────────────
GOLDEN_FRAME_CROP_MARGIN      = 0.25   # 25% context padding around bbox
GOLDEN_FRAME_MIN_CROP_PX      = 128    # Minimum crop side length (pixels)
GOLDEN_FRAME_CONFIDENCE_MIN   = 0.55   # Gemini confidence below this → reject
GOLDEN_FRAME_CROP_DIR         = "/content/golden_crops"

# ── Financial Intelligence (Batch 4) ──────────────────────────────────────────
VEHICLE_VALUE            = 15000  # USD default
TOTAL_LOSS_THRESHOLD     = 0.75   # Repair > 75% vehicle value → TOTALED
USD_TO_INR               = 83     # Display conversion rate
```

---

## Dashboards

After Cell G4 runs, two public Gradio URLs are printed. Share these links for demos.

### User Dashboard (Port 7860)

| Tab | Purpose |
|---|---|
| 📥 1 · Vehicle Intake | Enter Vehicle ID, owner name, incident date + location on Leaflet map, upload damage photos |
| 🛡️ 2 · Insurance Preference | Choose "Yes – file a claim" (activates fraud checks) or "No – assessment only"; enter policy details |
| 🔬 3 · Damage Analysis | Trigger full AI pipeline; see status stepper, agent timeline, detection table |
| 📊 4 · Executive Summary | Vehicle health score, line-item cost estimate, claims ruling badge, fraud badge, forensic report |
| 💬 5 · AI Assistant | Groq-powered chat scoped to the current session; full DB context injected per turn |

### Auditor Dashboard (Port 7861)

| Tab | Purpose |
|---|---|
| 🗂️ 1 · Case Explorer | Search all cases by ID, status, date, fraud flag; click row for full detail + JSON |
| 📋 2 · Insurance Claims | All filed claims with cost breakdowns; approve or reject directly |
| 🚨 3 · Fraud Review | Suspicious cases with forensic detail; mark as fraud, clear, approve, or reject |
| 👤 4 · User Management | Per-vehicle aggregated stats; click user row for full claim history |
| 📊 5 · Audit Logs | MemorySaver checkpoint timeline, agent trace JSON, all agent decisions table |

**AI Auditor Sidebar** — Groq-powered bot available from every tab; auto-injects live DB context (recent cases, system counts, fraud flags) on every message.

> **Share toggle:** Set `GRADIO_SHARE = True` in Cell G1 (default) to generate public ngrok share links valid for 1 week. Set to `False` for local-only access.

---

## Database Setup

SmartForge automatically selects a database backend based on `SMARTFORGE_MONGO_URI`:

### Option A — MongoDB Atlas (Recommended for Production)

1. Create a free cluster at [mongodb.com/atlas](https://www.mongodb.com/atlas)
2. Get your connection string: `mongodb+srv://<user>:<password>@<cluster>.mongodb.net/`
3. Whitelist `0.0.0.0/0` in **Network Access** settings (required for Colab's dynamic IPs)
4. Add it as a Colab Secret named `SMARTFORGE_MONGO_URI`, **or** paste directly into Cell G1:

```python
MONGO_URI = "mongodb+srv://user:password@cluster.mongodb.net/"
```

### Option B — SQLite (Zero-config Fallback)

Leave `MONGO_URI = ""` in Cell G1. SmartForge automatically creates `/content/smartforge_claims.db`. All dashboard functions work identically — data is lost when the Colab session ends.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No GPU detected` warning | `Runtime → Change runtime type → T4 GPU`, then `Runtime → Restart and run all` |
| `FileNotFoundError: seg-best.pt` | Run `!cp notebooks/models/seg-best.pt /content/ && !cp notebooks/models/detect-best.pt /content/` — models are in the `notebooks/models/` folder of this repo |
| `KeyError: GEMINI_API_KEY` or `SecretNotFoundError` | Add the secret in Colab Secrets (🔑 icon) and enable **Notebook access** to ON |
| `429 quota exhausted` on Gemini | Pipeline auto-falls back to `gemini-2.5-flash-lite`; add billing to your Google AI project for higher quota |
| Gradio share link not generated | `GRADIO_SHARE = True` must be set in Cell G1; Colab must have internet access |
| `ModuleNotFoundError` on any package | Re-run Cell 2 (dependency install); some packages need a kernel restart after first install |
| Claim always routes to `CLM_MANUAL` | All unconfirmed or rejected detections force manual review — expected behaviour for non-car vehicles or low-confidence images with no Gemini API key |
| MongoDB connection timeout | Whitelist `0.0.0.0/0` in Atlas Network Access settings, or leave `MONGO_URI` empty to use SQLite fallback |
| Gradio link expired | Gradio share links expire after 1 week — re-run Cell G4 to get fresh links |
| `CUDA out of memory` | Restart runtime, re-run from Cell 2; SAM + MiDaS require ~4 GB VRAM |

---

## Notes for Contributors

- **Cell 1 is the only user-editable cell** in the pipeline section. All other pipeline cells (0–13) are functional and should not be modified unless you are extending the architecture.
- The LangGraph state (`SmartForgeState`) is the **single source of truth** — all node outputs are returned as partial state dicts and merged by the graph engine. No node mutates state directly.
- Fraud checks degrade gracefully — missing API keys cause individual checks to be skipped without breaking the pipeline.
- `GRADIO_DEBUG = True` in Cell G1 shows full tracebacks in the browser for development.
- `BYPASS_FRAUD = True` (Cell 1 default) skips all fraud checks for quick demos. The Gradio UI dynamically overrides this to `False` when a user selects "Yes – I want to file a claim" in Tab 2.
- Both Gradio share links expire after **1 week**. Re-run Cell G4 to get fresh links without rerunning the full pipeline.

---

*SmartForge v36 · LangGraph + Gemini 2.5 Flash + Groq LLaMA 3.3 70B · Google Colab (T4 GPU)*
