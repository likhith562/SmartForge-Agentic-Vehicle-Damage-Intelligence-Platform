# 🚗 SmartForge — Agentic Vehicle Damage Intelligence Platform

> **Autonomous Insurance Claims Processing · LangGraph + Gemini VLM + Groq**  
> `Vehicle_Damage_Agentic_AI_v36_fixed.ipynb`

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [API Keys & Secrets](#api-keys--secrets)
- [Model Files](#model-files)
- [Quick Start (Google Colab)](#quick-start-google-colab)
- [Cell-by-Cell Guide](#cell-by-cell-guide)
- [Configuration Reference](#configuration-reference)
- [Dashboards](#dashboards)
- [Database Setup](#database-setup)
- [Troubleshooting](#troubleshooting)

---

## Overview

SmartForge is a fully agentic, end-to-end insurance claims processing pipeline built on **LangGraph**. It accepts one or more vehicle damage photos and autonomously:

1. Detects and segments damage using a custom YOLO model + SAM
2. Estimates depth deformation with MiDaS
3. Runs a **5-check fraud detection** layer (EXIF temporal, GPS consistency, software integrity, perceptual hash, AI screen-detection)
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
|---|---|
| LangGraph cyclic graph with circuit breaker | `health_monitor` → `perception_retry` loop |
| LangGraph `Send` API for parallel fan-out | `map_images_node` — one worker per uploaded photo |
| Human-in-the-Loop interrupt | `interrupt_before=["decision"]` on high-value claims |
| NetworkX in-memory graph DB | `fusion_node` — cross-image damage deduplication |
| MemorySaver checkpointing | Full state dump at every super-step |

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Runtime** | Google Colab with **GPU** (T4 or better). `Runtime → Change runtime type → T4 GPU` |
| **Python** | 3.10+ (Colab default) |
| **Disk space** | ~3 GB for model weights + dependencies |
| **API access** | Gemini API, Groq API (see below) |

---

## API Keys & Secrets

SmartForge uses **Google Colab Secrets** to avoid hardcoding credentials. All keys are read at runtime via `google.colab.userdata`.

### Required Keys

| Secret Name | Where to get it | Used for |
|---|---|---|
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com/app/apikey) | Gemini VLM — vehicle classification, damage verification, Golden Frame forensics |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com/keys) | Groq — fast AI-generated damage narrative text |

### Optional Keys

| Secret Name | Purpose | Fallback behaviour |
|---|---|---|
| `SERPAPI_KEY` | Reverse image search in fraud layer | Fraud check skipped silently |
| `WINSTON_AI_KEY` | AI-generated image detection (fraud layer) | Check skipped silently |
| `SMARTFORGE_MONGO_URI` | MongoDB Atlas connection string | Auto-falls back to local SQLite |

---

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
| `seg-best.pt` | ✅ Included in repo — `models/seg-best.pt` | Custom YOLO damage segmentation model |
| `detect-best.pt` | ✅ Included in repo — `models/detect-best.pt` | Custom YOLO vehicle part detection model |
| `sam_vit_b_01ec64.pth` | ⬇️ Auto-downloaded from Meta at runtime | Meta's Segment Anything Model (SAM ViT-B) |

### How models are loaded

**`seg-best.pt` and `detect-best.pt`** ship with this repository inside the `models/` folder. When you clone the repo and open the notebook in Colab, copy them to `/content/` with:

```python
# Run this once after cloning — or add it as a Colab cell
!cp models/seg-best.pt     /content/seg-best.pt
!cp models/detect-best.pt  /content/detect-best.pt
```

Or if you mounted Google Drive and cloned there:

```python
import shutil
shutil.copy("/content/drive/MyDrive/smartforge/models/seg-best.pt",    "/content/seg-best.pt")
shutil.copy("/content/drive/MyDrive/smartforge/models/detect-best.pt", "/content/detect-best.pt")
```

**`sam_vit_b_01ec64.pth`** is **not** stored in the repo (it is 375 MB). The notebook downloads it automatically from Meta's servers on first run:

```python
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
```

No action needed — if the file is missing from `/content/`, the pipeline fetches it before the CV stack starts. Subsequent runs in the same Colab session reuse the cached file.

---

## Quick Start (Google Colab)

```
1.  Clone the repo into your Google Drive (or open directly in Colab via GitHub)
2.  Runtime → Change runtime type → T4 GPU
3.  Add Colab Secrets:  GEMINI_API_KEY  and  GROQ_API_KEY
4.  Copy model weights:  !cp models/seg-best.pt /content/  &&  !cp models/detect-best.pt /content/
5.  Runtime → Run all
        ↳ SAM weights (~375 MB) are downloaded automatically — no action needed
6.  Wait for Cell G4 to print public share URLs (≈ 5–8 minutes first run)
7.  Open the User Dashboard URL in any browser
```

On subsequent runs within the same Colab session, skip steps 1–3 and go straight to `Runtime → Run all`.

---

## Cell-by-Cell Guide

| Cell | Name | What it does | Must edit? |
|---|---|---|---|
| **Cell 0** | Drive Mount | Mounts Google Drive (optional) | No |
| **Cell 1** | Configuration | All runtime parameters — model paths, thresholds, API keys | ✏️ **Yes** (model paths) |
| **Cell 2** | Install Dependencies | `pip install` for all packages — run once per session | No |
| **Cell 3** | Imports & Env Check | Validates GPU, imports libraries | No |
| **Cell 4** | State Schema | LangGraph `TypedDict` state definition | No |
| **Cell 5** | Helpers & Cost Engine | Part name maps, repair database, cost estimation | No |
| **Cell 6** | `intake` node | Image validation, resolution correction, condition analysis | No |
| **Cell 6b** | `fraud` node | 5-check fraud & integrity layer | No |
| **Cell 6c** | `map_images` / `fusion` | Batch 2 multi-image parallel fan-out + NetworkX graph DB | No |
| **Cell 6d** | `verification_v2` | Batch 3 Golden Frame: crops damage bbox → Gemini forensic analysis | No |
| **Cell 7** | `perception` node | SAHI → SAM → MiDaS → Part Detection | No |
| **Cell 7b** | `gemini_agent` | Gemini VLM enrichment (vehicle type, part location, verification) | No |
| **Cell 7c** | `false_positive_gate` | 4-layer false-positive filter for non-car vehicles | No |
| **Cell 8** | `health_monitor` | Validates perception output; routes retry or reasoning | No |
| **Cell 9** | `perception_retry` | Circuit-breaker retry wrapper | No |
| **Cell 10** | `reasoning` node | Severity classification + cost estimation | No |
| **Cell 11** | `decision` node | Insurance ruling (CLM_AUTO_APPROVE / CLM_MANUAL / CLM_REJECT) | No |
| **Cell 12** | `report` node | Groq-generated AI damage narrative | No |
| **Cell 13** | Build LangGraph | Assembles graph with all nodes, edges, checkpointer | No |
| **Cell G1** | Dashboard Config | Gradio title, theme, share toggle, MongoDB URI | ✏️ Optional |
| **Cell G2** | Database Layer | MongoDB Atlas / SQLite auto-fallback setup | No |
| **Cell G3** | User Dashboard | Builds 5-tab claimant UI (does not launch) | No |
| **Cell G4** | Auditor Dashboard + Launch | Builds auditor UI and **launches both apps** | No |

---

## Configuration Reference

All user-editable settings live in **Cell 1**. Edit only this cell.

```python
# ── Model paths ────────────────────────────────────────────────────────────────
DAMAGE_MODEL_PATH        = "/content/seg-best.pt"        # YOLO damage segmentation
PART_MODEL_PATH          = "/content/detect-best.pt"     # YOLO part detection
SAM_CHECKPOINT           = "/content/sam_vit_b_01ec64.pth"
SAM_URL                  = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

# ── SAHI inference settings ────────────────────────────────────────────────────
SAHI_CONFIDENCE          = 0.3    # Base detection confidence (auto-adjusted per image)
SAHI_SLICE_SIZE          = 640    # Tile size for sliced inference
SAHI_OVERLAP             = 0.2    # Tile overlap ratio

# ── Claim metadata (set per job or leave blank) ────────────────────────────────
VEHICLE_ID               = ""
IMAGE_ID                 = ""
POLICY_ID                = ""

# ── Agentic thresholds ─────────────────────────────────────────────────────────
MAX_RETRIES              = 2      # Max perception retries before circuit-breaker fires
ESCALATION_THRESHOLD     = 70     # Vehicle health score below this → escalate
CONFIDENCE_RECHECK_LIMIT = 0.45   # YOLO confidence below this → Gemini re-verification
AUTO_APPROVE_THRESHOLD   = 85     # Health score above this → auto-approve claim

# ── Gemini & Groq models ───────────────────────────────────────────────────────
GEMINI_MODEL             = "gemini-2.5-flash"
GEMINI_MODEL_FALLBACK    = "gemini-2.0-flash"
GROQ_MODEL               = "llama-3.3-70b-versatile"
```

---

## Dashboards

After Cell G4 runs, two public Gradio URLs are printed. Share these links for demos.

### User Dashboard (Port 7860)

| Tab | Purpose |
|---|---|
| 📥 Vehicle Intake | Enter Vehicle ID, Policy ID, upload photos |
| 🔬 Damage Analysis | Trigger full AI pipeline; see annotated damage maps |
| 🛡️ Insurance Claim | File claim; activates 5-check fraud layer |
| 📊 Executive Summary | Vehicle health score, cost estimate, claims ruling, fraud badge |
| 💬 AI Assistant | Groq-powered chat scoped to the current session |

### Auditor Dashboard (Port 7861)

| Tab | Purpose |
|---|---|
| 🗂️ Case Explorer | Search all cases by ID, status, date, fraud flag |
| 📋 Insurance Claims | All filed claims with cost breakdowns |
| 🚨 Fraud Review | Suspicious cases — mark as fraud, clear, or approve |
| 👤 User Management | Per-vehicle claim history and statistics |
| 📋 Audit Logs | Full agent trace, MemorySaver checkpoint dumps, pipeline timeline |

> **Share toggle:** Set `GRADIO_SHARE = True` in Cell G1 (default) to generate public ngrok links valid for 72 hours. Set to `False` for local-only access.

---

## Database Setup

SmartForge automatically selects a database backend:

### Option A — MongoDB Atlas (Recommended for Production)

1. Create a free cluster at [mongodb.com/atlas](https://www.mongodb.com/atlas)
2. Get your connection string: `mongodb+srv://<user>:<password>@<cluster>.mongodb.net/`
3. Add it as a Colab Secret named `SMARTFORGE_MONGO_URI`, **or** paste it into Cell G1:

```python
MONGO_URI = "mongodb+srv://user:password@cluster.mongodb.net/"
```

### Option B — SQLite (Zero-config Fallback)

Leave `MONGO_URI = ""` in Cell G1. SmartForge automatically creates a local SQLite file at `/content/smartforge.db`. All dashboard functions work identically — data is lost when the Colab session ends.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No GPU detected` warning | `Runtime → Change runtime type → T4 GPU`, then `Runtime → Restart and run all` |
| `FileNotFoundError: seg-best.pt` | Run `!cp models/seg-best.pt /content/ && !cp models/detect-best.pt /content/` — the files are in the `models/` folder of this repo |
| `KeyError: GEMINI_API_KEY` | Add the secret in Colab Secrets (🔑 icon) and enable Notebook access |
| `429 quota exhausted` on Gemini | Pipeline auto-falls back to `gemini-2.0-flash`; add billing to your Google AI project for higher quota |
| Gradio share link not generated | `GRADIO_SHARE = True` must be set in Cell G1; Colab must have internet access |
| `ModuleNotFoundError` on any package | Re-run Cell 2 (dependency install); some packages need a kernel restart after first install |
| Claim always routes to `CLM_MANUAL` | All unconfirmed or rejected detections force manual review — this is expected behaviour for non-car vehicles or low-confidence images |
| MongoDB connection timeout | Whitelist `0.0.0.0/0` in your Atlas Network Access settings, or switch to SQLite fallback |

---

## Notes for Contributors

- **Cell 1 is the only user-editable cell** in the pipeline section. All other pipeline cells (0–13) are functional and should not be modified unless you are extending the architecture.
- The LangGraph state (`SmartForgeState`) is the **single source of truth** — all node outputs are returned as partial state dicts and merged by the graph engine.
- Fraud checks degrade gracefully — missing API keys cause individual checks to be skipped without breaking the pipeline.
- `GRADIO_DEBUG = True` in Cell G1 shows full tracebacks in the browser for development.

---

*SmartForge v36 · LangGraph + Gemini 2.5 Flash + Groq LLaMA 3.3 70B · Google Colab (T4 GPU)*
