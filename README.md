<div align="center">

# 🚗 Agentic Vehicle Damage Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Directed%20Cyclic%20Graph-FF6B35?style=for-the-badge)](https://github.com/langchain-ai/langgraph)
[![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash%20VLM-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070b-F55036?style=for-the-badge)](https://groq.com)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://mongodb.com)
[![Gradio](https://img.shields.io/badge/Gradio-5%20Tab%20Dashboard-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![Open In Colab](https://img.shields.io/badge/Open%20In-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/your-username/smartforge-agentic-ai/blob/main/notebooks/Vehicle_Damage_Agentic_AI_v36_gradio.ipynb)

<br/>

**An end-to-end autonomous insurance claims processing system** powered by a LangGraph Directed Cyclic Graph orchestrating 13 specialized AI agents — from YOLO/SAM/MiDaS computer vision through Gemini multimodal verification, a 5-check forensic fraud layer, and Groq-generated structured reports — with dual Gradio dashboards and MongoDB persistence.

<br/>

[**📓 Open Notebook + Gradio Live Demo**](#quick-start) · [**🏗️ Architecture**](#architecture) · [**📊 Dashboards**](#dashboards) · [**🛡️ Fraud Layer**](#fraud-detection-layer) · [**🚀 Setup**](#installation)

</div>

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
  - [System Overview](#system-overview)
  - [Computer Vision Pipeline](#computer-vision-pipeline)
  - [Fraud Detection Layer](#fraud-detection-layer)
  - [Agentic Decision & Reasoning Flow](#agentic-decision--reasoning-flow)
  - [Multi-Image Map-Reduce](#multi-image-map-reduce)
  - [LangGraph State Schema](#langgraph-state-schema)
  - [Graph Topology](#graph-topology)
- [Feature Batches](#feature-batches)
- [Dashboards](#dashboards)
  - [User Dashboard (5-Tab)](#user-dashboard-5-tab)
  - [Auditor Dashboard (5-Tab)](#auditor-dashboard-5-tab)
- [Database Schema](#database-schema)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Tech Stack](#tech-stack)
- [Roadmap](#roadmap)

---

## Project Overview

SmartForge is a **production-grade autonomous insurance claims processing platform** built on a LangGraph Directed Cyclic Graph (DCG). The system processes vehicle damage images end-to-end — from fraud-gated intake through multi-model computer vision, Gemini VLM multimodal verification, severity classification, financial estimation, and Groq-generated structured reports — all persisted to MongoDB and surfaced through two role-separated Gradio applications.

### Why SmartForge?

The global motor insurance industry processes over **300 million claims annually**, with an estimated **10–15% involving fraudulent submissions**. Traditional assessment workflows take 3–5 business days and require manual adjuster review for every claim. SmartForge reduces this to:

| Metric | Traditional | SmartForge |
|--------|-------------|------------|
| Processing time | 3–5 days | ~5 minutes |
| Fraud detection | Manual review | Automated 5-check layer |
| Accuracy | Adjuster-dependent | YOLO + SAM + Gemini verification |
| Cost estimation | Workshop estimate | Mitchell/Audatex-style DB |
| Auditability | Paper trail | Full MemorySaver checkpoint log |

---

## Architecture

SmartForge is built around four interconnected subsystems. All four are visualised in the diagrams below.

### System Overview

The top-level architecture orchestrates all agents through a LangGraph state machine. Every node communicates exclusively through `SmartForgeState` — a typed dictionary that flows through the graph and accumulates results without mutation conflicts.

<div align="center">
<img src="assets/diagrams/system_architecture.png" alt="SmartForge System Architecture" width="100%">
<br/>
<em>SmartForge System Architecture — Full agentic pipeline from user upload to final claim decision</em>
</div>

<br/>

The pipeline follows two execution paths depending on fraud detection results:

```
User Upload
    │
    ▼
Intake Agent ──► Fraud Detection Layer (5-check) ──► Trust Score < 40 ──► Human Audit System
                                                  │
                                                  |  Trust Score ≥ 70 (PASS)
                                                  |
                                                  ▼
                        ┌──────────────────────────────────────────────────┐
                        │              Agentic AI Core                     │
                        │  Perception Engine (YOLO + SAHI + SAM + MiDaS)  │
                        │  Gemini Agent (multimodal validation)            │
                        │  False Positive Gate                             │
                        │  Health Monitor (retry loop)                     │
                        │  Verification Engine (Golden Frame crops)        │
                        └──────────────────────────────────────────────────┘
                                                  │
                                                  ▼
                                          Reasoning Engine ──► Decision Engine ──► Report Generator
                                          (severity + cost)    (score + ruling)    (Groq narrative)
                                                  │
                                                  ▼
                                  Final Output: Claim Decision + Cost Estimate
```

---

### Computer Vision Pipeline

<div align="center">
<img src="assets/diagrams/cv_pipeline.png" alt="Computer Vision & Damage Analysis Pipeline" width="100%">
<br/>
<em>Computer Vision & Damage Analysis Pipeline — YOLO → SAHI → SAM → MiDaS → Feature Extraction</em>
</div>

<br/>

The perception engine runs four stacked models in sequence:

| Stage | Model | Purpose |
|-------|-------|---------|
| **Detection** | YOLOv8 (custom-trained) | Detect vehicle parts and damage regions |
| **Slicing** | SAHI (640px slices, 0.2 overlap) | Recover small damage missed at full resolution |
| **Segmentation** | SAM (Segment Anything) | Precise binary masks around each detection |
| **Depth** | MiDaS | 2D→3D reasoning, compute deformation index |
| **Feature extraction** | Custom | Bounding boxes, area ratios, relative deformation index |

**Adaptive SAHI confidence** is computed at intake time based on specular highlight variance (high-gloss vehicles raise confidence to avoid false positives on reflections).

---

### Fraud Detection Layer

<div align="center">
<img src="assets/diagrams/fraud_layer.png" alt="SmartForge Fraud Detection Layer Flow" width="100%">
<br/>
<em>Fraud Detection Layer — 5-check parallel pipeline computing a 0–100 Trust Score</em>
</div>

<br/>

The fraud layer runs **5 independent checks** in parallel and aggregates into a Trust Score. A score below 40 routes to the Human Audit System; ≥ 40 passes to the perception pipeline.

| Check | Method | Score Impact |
|-------|--------|-------------|
| **① Temporal Consistency** | EXIF `DateTimeOriginal` vs `claim_date` | +20 pts pass / −25 pts fail |
| **② GPS Location Consistency** | EXIF GPS vs claim location (Haversine distance) | +20 pts pass / −30 pts fail |
| **③ Software / Source Integrity** | EXIF `ImageSoftware` — Adobe/Canva flags editing | +10 pts pass / −30 pts fail |
| **④ pHash Duplicate Detection** | Perceptual hash vs local fraud DB + SerpAPI (optional) | +20 pts pass / −40 pts duplicate |
| **⑤ AI-Generation & Screen Forensics** | ELA analysis + FFT Moiré pattern detection | +10 pts pass / −35 pts fail |

```
Trust Score = base_100 + Σ(check_adjustments)
Trust Score < 40  →  SUSPICIOUS_HIGH_RISK  →  Human Audit
Trust Score ≥ 40  →  VERIFIED              →  Perception Pipeline
```

> **Note:** `BYPASS_FRAUD = True` skips all checks for demo runs. When a user files an insurance claim in the UI, `BYPASS_FRAUD` is dynamically set to `False` and the full 5-check layer activates.

---

### Agentic Decision & Reasoning Flow

<div align="center">
<img src="assets/diagrams/decision_flow.png" alt="SmartForge Agentic Decision & Reasoning Flow" width="100%">
<br/>
<em>Agentic Decision & Reasoning Flow — From CV candidate output to final claim ruling</em>
</div>

<br/>

After CV detections pass the fraud gate, the agentic reasoning stack takes over:

1. **Gemini Verification Agent** — multimodal validation of each detection bounding box against the full image; confirms physical vs cosmetic damage
2. **Golden Frame Processing** — crops high-resolution regions around each detection (25% context padding) and sends to Gemini for deep forensic analysis
3. **Verification Output** — produces `is_physical_damage`, `severity_index`, `confidence_score`, `repair_recommendation` per detection
4. **Damage Classification** — refines type (Scratch, Crack, Dent, Deformation) and severity (Low/Moderate/High)
5. **Structural Analysis** — determines `part_structurally_compromised` flag and Cosmetic vs Functional categorisation
6. **Repair Recommendation Engine** — Repair/Replace decision via `SEVERITY_TO_ACTION` mapping
7. **Cost Estimation Engine** — Mitchell/Audatex-style `REPAIR_DATABASE` with replace, paint, and labour costs (INR + USD)
8. **Aggregation Engine** — combines all damages, computes overall health score (0–100)
9. **Final Decision Logic** — All AI decisions route to one of three outcomes: `CLM_PENDING` (score ≥ 70, clean claim, awaiting auditor sign-off); `CLM_WORKSHOP` (score < 70 or High-severity damage, workshop inspection required); `CLM_MANUAL` (fraud flags or unconfirmed detections, immediate manual review). **The AI never auto-approves** — final `CLM_APPROVED` status can only be set by a human auditor via the Auditor Dashboard

---

### Multi-Image Map-Reduce

<div align="center">
<img src="assets/diagrams/multi_image_architecture.png" alt="SmartForge Multi-Image Map-Reduce Architecture" width="100%">
<br/>
<em>Batch 2: Multi-Image Map-Reduce — Parallel processing of N images with NetworkX Graph DB fusion</em>
</div>

<br/>

For claims with multiple vehicle photos (different angles), LangGraph's `Send` API fans out to `N` parallel `cv_worker` nodes simultaneously:

```
map_images_node ──[Send API]──► cv_worker_0 (img_0) ──┐
                              ► cv_worker_1 (img_1) ──┤──► fusion_node ──► gemini_agent ──► ...
                              ► cv_worker_N (img_N) ──┘
```

**Fusion via NetworkX DiGraph:**
```
Nodes: IMAGE | PART | DETECTION
Edges: contains (IMAGE→DETECTION) | located_on (DETECTION→PART)

Deduplication:
  - Spatial overlap matching (IoU / geometry)
  - Highest-confidence view selected per damage
  - Cross-image redundancy removed
```

---

### LangGraph State Schema

The `SmartForgeState` TypedDict is the single source of truth flowing through every node. Key fields:

```python
class SmartForgeState(TypedDict):
    # Message History (append reducer — never overwrites)
    messages:               Annotated[List[dict], operator.add]

    # Workflow Context
    image_path:             str
    image_bgr:              Optional[object]          # numpy BGR array
    raw_detections:         List[dict]                # PerceptionAgent output
    depth_map:              Optional[object]          # MiDaS depth map
    damages_output:         List[dict]                # ReasoningAgent output
    final_output:           Optional[dict]            # Complete result

    # Gemini VLM Agent
    vehicle_type:           str                       # "car"|"2W"|"3W"|"unknown"
    vehicle_make_estimate:  str                       # "sedan-class","hatchback"
    adaptive_sahi_conf:     float                     # Adaptive confidence from intake

    # Validation Metrics (HealthMonitor)
    health_score:           float
    validation_passed:      bool
    retry_count:            int

    # Batch 3: Golden Frame
    verified_damages:       Annotated[List[dict], operator.add]
    golden_crops:           List[dict]

    # Batch 2: Multi-Image Map-Reduce
    image_paths:            List[str]
    all_raw_detections:     Annotated[List[dict], operator.add]
    fused_detections:       Annotated[List[dict], operator.add]

    # Batch 1: Fraud Layer
    fraud_report:           Optional[dict]
    claim_date:             str
    claim_lat:              float
    claim_lon:              float

    # Batch 4: Financial Intelligence
    total_loss_flag:        bool
    financial_estimate:     Optional[dict]

    # Audit
    job_id:                 str
    vehicle_id:             str
    pipeline_trace:         dict
```

---

### Graph Topology

```
intake ──► fraud ──┬──► map_images ──► cv_worker(×N) ──► fusion  ──►┐
                   │   [Batch 2 Send fan-out]                       │
                   │                                                ▼
                   └──────────────────────────────────────►    perception
                                                                    │
                                                 human_audit ◄─SUSPICIOUS
                                                                    │
                                                              gemini_agent
                                                                    │
                                                        false_positive_gate
                                                                    │
                                                          health_monitor ──► perception_retry
                                                                    │              ↑
                                                                    └──────────────┘
                                                                    │ (max 2 retries)
                                                              verification_v2
                                                                    │
                                                               reasoning
                                                                    │
                                                         decision [HITL interrupt]
                                                                    │
                                                                 report ──► END
```

**Conditional edges:**

| Source | Condition | Target |
|--------|-----------|--------|
| `fraud` | `trust_score < 40` | `human_audit → END` |
| `fraud` | `len(image_paths) > 1` | `map_images` |
| `fraud` | verified, single image | `perception` |
| `health_monitor` | `validation_passed = True` | `verification_v2` |
| `health_monitor` | `validation_passed = False` | `perception_retry` |

---

## Feature Batches

SmartForge was developed incrementally in 4 production-ready feature batches:

<details>
<summary><b>Batch 1 — 5-Check Fraud & Integrity Layer</b></summary>

- EXIF temporal consistency check
- GPS Haversine distance validation
- EXIF software source integrity
- Perceptual hash (pHash) duplicate detection with local fraud DB
- FFT Moiré pattern detection + ELA AI-generation forensics (Winston AI → ELA → Laplacian 3-stage fallback)
- `BYPASS_FRAUD` toggle for demo mode; dynamically switched `False` when user selects "Yes – I want to file a claim" in Tab 2 (Insurance Preference)
- **3-Strike photo retry limit** — fraud-flagged uploads can be retried up to `MAX_FRAUD_RETRIES` (default 3) times; on the third failure the case is permanently closed with a `FRAUD_MAX_RETRIES_EXCEEDED` flag

</details>

<details>
<summary><b>Batch 2 — Multi-Image Map-Reduce + NetworkX Graph DB</b></summary>

- LangGraph `Send` API fan-out to N parallel `cv_worker` nodes
- Per-image perception + Gemini verification in parallel
- NetworkX DiGraph: IMAGE → DETECTION → PART nodes and edges
- IoU-based spatial overlap matching for deduplication
- Confidence scoring selects best-angle detection per damage
- Graph queryable after run: `claims_graph.predecessors(part_node)`

</details>

<details>
<summary><b>Batch 3 — Golden Frame Verification</b></summary>

- Full-resolution bounding box crops (25% padding for context)
- Minimum crop size: 128px — prevents Gemini from analysing a 5px region
- Per-crop Gemini Deep Look: `is_physical_damage`, `severity_index`, `confidence_score`, `repair_recommendation`
- Multi-angle cross-verification for damages seen in multiple images
- Golden crops stored selectively (fraud/critical cases only) to avoid storage overhead

</details>

<details>
<summary><b>Batch 4 — Financial Intelligence Engine</b></summary>

- Mitchell/Audatex-style `REPAIR_DATABASE` with replace, paint, and labour costs
- Severity-to-action mapping: Minor/Moderate → `REPAIR/PAINT`; Severe/Critical → `REPLACE`
- Fuzzy part name matching for database lookups
- Total Loss threshold: repair cost > 75% vehicle value → `TOTALED`
- USD and INR outputs (configurable exchange rate)
- Line-item `financial_estimate` dictionary passed through to all reports

</details>

---

## Dashboards

SmartForge ships two completely separate Gradio applications running on different ports. **Roles are never mixed in a single UI** — a deliberate security and UX decision.

### User Dashboard (5-Tab)

> Port `7860` · Audience: Vehicle owner / claimant

```
Tab 1 — 📥 1 · Vehicle Intake
  • Vehicle ID (mandatory, validated — e.g. VH001 or TN-09-AB-1234)
  • Owner / Claimant Name
  • Vehicle Type dropdown: Auto-Detect (Gemini VLM) | Car/Sedan/SUV |
    2-Wheeler | 3-Wheeler
  • Incident Date — native HTML5 date picker (capped at today, auto-synced
    to a hidden Gradio Textbox before submission)
  • Incident Location — interactive Leaflet map (OpenStreetMap tiles):
      - Address / city search bar with Nominatim autocomplete (300 ms debounce,
        keyboard navigation ↑↓ Enter Esc, outside-click dismiss)
      - GPS button — calls browser geolocation API, centres map instantly
      - Draggable pin — click anywhere or drag to reposition
      - Live coordinate bar updates as you drag (lat, lon to 6 d.p.)
      - "Confirm Location" button writes lat/lon to the Gradio state
        fields used by the fraud GPS consistency check
  • Photo upload — drag & drop, multi-file (JPG/PNG); multiple images
    activate Batch 2 Multi-Image Map-Reduce automatically
  • "Save & Proceed to Insurance Preference" saves session to DB and
    advances to Tab 2
  • Intake Status textbox shows success/error summary

Tab 2 — 🛡️ 2 · Insurance Preference
  • Determines whether fraud checks activate during Damage Analysis
  • Info box explains the 3-attempt fraud retry limit upfront
  • Radio: "Yes – I want to file a claim" | "No – damage assessment only"
  • YES → reveals insurance claim form:
      - Policy Number (mandatory)
      - Accident Date (auto-filled from Tab 1 DB record)
      - Claim Reason (mandatory)
      - Additional Notes (optional — FIR number, witness info, etc.)
  • NO → assessment-only mode saved; fraud checks bypassed in Tab 3
  • "Save Preference & Proceed" writes preference to DB and advances to Tab 3

Tab 3 — 🔬 3 · Damage Analysis
  • "Run Full Analysis" triggers the complete LangGraph pipeline
  • Pipeline Status textbox — scrolling log of each agent's progress
  • Status Stepper (HTML) — visual progress indicator across stages
  • Primary Vehicle Photo viewer and Pipeline Timeline node tiles
  • Detection Records table — ID | Type | Location | Severity | Conf | Status

Tab 4 — 📊 4 · Executive Summary
  • Executive Summary textbox — Groq-generated plain-English narrative
  • Health Score badge — colour-coded: green ≥ 80 / amber ≥ 60 / red < 60
  • Claim Ruling badge (CLM_PENDING / CLM_WORKSHOP / CLM_MANUAL)
  • Line-Item Repair Estimate table with grand total row
  • Fraud Detection badge and Forensic Integrity report

Tab 5 — 💬 5 · AI Assistant
  • gr.ChatInterface powered by Groq Llama-3.3-70b
  • Scope: current session only — cannot access other users' data
  • Chat history persisted to MongoDB after each turn
```

---

### Auditor Dashboard (5-Tab)

> Port `7861` · Audience: Insurance adjuster / compliance auditor
> Role: **AUDITOR** — no `vehicle_id` filter applied → full case visibility

```
Tab 1 — 🗂️ 1 · Case Explorer
  • Search: Vehicle ID, Status dropdown, Fraud Only checkbox
  • Stats cards: Total Cases | Analyzed | Fraud Flagged | Approved | Rejected | Pending
  • Results table — 10 columns with click-to-load case detail

Tab 2 — 📋 2 · Insurance Claims
  • All filed claims, status filter, approve/reject actions
  • Click row to auto-fill Case ID, then Approve or Reject

Tab 3 — 🚨 3 · Fraud Review
  • All fraud-flagged cases with full forensic detail on row click
  • Auditor Decision: Confirm Fraud | Clear | Approve | Reject

Tab 4 — 👤 4 · User Management
  • Per-vehicle aggregated stats and full claim history on row click

Tab 5 — 📊 5 · Audit Logs
  • MemorySaver checkpoint timeline, agent trace JSON, decision table
```

---

## Database Schema

SmartForge uses **MongoDB Atlas** as primary storage with automatic SQLite fallback. One document per case:

```json
{
  "case_id":         "VH001-abc123f",
  "user_id":         "VH001",
  "vehicle_id":      "VH001",
  "status":          "approved",
  "created_at":      "2026-03-29T04:15:00.000Z",
  "user_data": {
    "owner_name":    "Rajesh Kumar",
    "incident_date": "2026-03-29",
    "incident_lat":  13.0827,
    "incident_lon":  80.2707
  },
  "final_output": {
    "claim_ruling_code":         "CLM_PENDING",
    "overall_assessment_score":  70,
    "confirmed_damage_count":    3,
    "financial_estimate": {
      "total_repair_usd":  1210.00,
      "total_repair_inr_fmt": "₹100,430",
      "total_loss_flag":   false,
      "disposition":       "REPAIRABLE"
    }
  },
  "fraud_report": {
    "trust_score":  85,
    "status":       "VERIFIED",
    "flags":        []
  },
  "is_fraud":        false,
  "auditor_review":  { "decision": "Approve Claim", "reviewed_at": "2026-03-29T05:00Z" }
}
```

### Status Pipeline

```
uploaded → pref_saved → analyzed → claim_submitted → fraud_checked → approved / rejected
```

> **Note:** `approved` and `rejected` are set exclusively by the human auditor. The AI pipeline never sets `approved` directly.

---

## Repository Structure

```
smartforge-agentic-ai/
│
├── .env.example                    # API key template
├── .gitignore
├── README.md                       # This file
├── requirements.txt
├── main.py                         # Entry point — launches both Gradio apps
│
├── assets/
│   ├── diagrams/                   # Architecture PNG diagrams
│   └── demo_screenshots/           # UI screenshots
│
├── data/
│   ├── sample_images/              # Test images
│   └── fraud_hash_db.json          # Local pHash fraud database
│
├── notebooks/
│   ├── README.md                   # Notebook quick-start guide + Gradio live demo info
│   └── Vehicle_Damage_Agentic_AI_v36_gradio.ipynb
│
└── src/
    ├── config/
    │   └── settings.py             # Env vars, thresholds, REPAIR_DATABASE
    ├── db/
    │   └── mongo_client.py         # db_upsert, db_get, db_find, db_count
    ├── models/
    │   ├── gemini_client.py        # Gemini 2.5 Flash VLM calls
    │   └── groq_client.py          # Groq Llama-3.3-70b calls
    ├── cv/
    │   ├── fraud_checks.py         # EXIF, GPS, pHash, ELA, FFT Moiré
    │   ├── perception.py           # SAHI + YOLO + SAM + MiDaS
    │   ├── depth.py                # MiDaS depth + deformation index
    │   └── fusion.py               # NetworkX DiGraph multi-image fusion
    ├── graph/
    │   ├── state.py                # SmartForgeState TypedDict
    │   ├── workflow.py             # StateGraph builder + compile
    │   └── nodes/                  # One file per LangGraph node
    └── ui/
        ├── theme.py                # Badge helpers, stat cards, pipeline timeline
        ├── helpers.py              # _status_stepper, _pipeline_timeline
        ├── user_dashboard.py       # 5-tab user Gradio app (port 7860)
        └── auditor_dashboard.py    # 5-tab auditor Gradio app (port 7861)
```

---

## Installation

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | |
| CUDA GPU | T4 or better | Required for YOLOv8 + SAM + MiDaS |
| Google Colab | — | Recommended runtime |
| MongoDB Atlas | Free tier | Optional — SQLite fallback available |

### Clone & Install

```bash
git clone https://github.com/your-username/smartforge-agentic-ai.git
cd smartforge-agentic-ai
pip install -r requirements.txt
```

### Model Files

| File | Source | Purpose |
|---|---|---|
| `seg-best.pt` | ✅ Included in `notebooks/models/` | Custom YOLO damage segmentation |
| `detect-best.pt` | ✅ Included in `notebooks/models/` | Custom YOLO part detection |
| `sam_vit_b_01ec64.pth` | ⬇️ Auto-downloaded at runtime (~375 MB) | Meta SAM ViT-B |

```bash
# Copy models to Colab working directory
!cp notebooks/models/seg-best.pt    /content/seg-best.pt
!cp notebooks/models/detect-best.pt /content/detect-best.pt
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```env
# AI API Keys
GEMINI_API_KEY=AIza...          # https://aistudio.google.com/app/apikey (free tier)
GROQ_API_KEY=gsk_...            # https://console.groq.com (free tier)

# Database
SMARTFORGE_MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/
# Leave empty → automatic SQLite fallback

# Optional Fraud Extensions
SERPAPI_KEY=...                 # https://serpapi.com (100/month free)
WINSTON_AI_KEY=...              # https://app.gowinston.ai (2000 credits/month)

# Pipeline Thresholds
SAHI_CONFIDENCE=0.3
FRAUD_TRUST_THRESHOLD=40
TOTAL_LOSS_THRESHOLD=0.75
VEHICLE_VALUE=15000
MAX_FRAUD_RETRIES=3
```

| Variable | Default | Effect |
|----------|---------|--------|
| `FRAUD_TRUST_THRESHOLD` | 40 | Below → `SUSPICIOUS_HIGH_RISK` |
| `ESCALATION_THRESHOLD` | 70 | Below → `CLM_WORKSHOP` |
| `TOTAL_LOSS_THRESHOLD` | 0.75 | Repair > 75% vehicle value → TOTALED |
| `MAX_RETRIES` | 2 | HealthMonitor retry limit |
| `MAX_FRAUD_RETRIES` | 3 | Photo re-upload attempts before case closure |

---

## Quick Start

### Option A — Google Colab with Gradio Live Demo (Recommended)

[![Open In Colab](https://img.shields.io/badge/Open%20In-Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/your-username/smartforge-agentic-ai/blob/main/notebooks/Vehicle_Damage_Agentic_AI_v36_gradio.ipynb)

1. Click the badge above or open `notebooks/Vehicle_Damage_Agentic_AI_v36_gradio.ipynb` in Colab
2. Set **Runtime → Change runtime type → T4 GPU**
3. Add secrets via the **🔑 Key icon** in the left sidebar:

   | Secret Name | Where to get it |
   |---|---|
   | `GEMINI_API_KEY` | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) |
   | `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) |
   | `SMARTFORGE_MONGO_URI` | MongoDB Atlas URI *(optional — SQLite fallback if omitted)* |

4. Run all cells top-to-bottom (Cell 0 → Cell G4)
5. **Two public Gradio share links appear in Cell G4** — one for users (port 7860), one for auditors (port 7861)

> See [`notebooks/README.md`](notebooks/README.md) for the full cell-by-cell walkthrough and live demo instructions.

### Option B — Local Launch

```bash
python main.py
# User Dashboard:    http://localhost:7860
# Auditor Dashboard: http://localhost:7861
```

### Option C — Direct Pipeline (no UI)

```python
from src.graph.workflow import graph
from src.graph.state import make_initial_state

state  = make_initial_state("/path/to/car_image.jpg")
thread = {"configurable": {"thread_id": state["job_id"]}}

for event in graph.stream(state, thread, stream_mode="values"):
    partial = event

# Resume after HITL interrupt at decision node
for event in graph.stream(None, thread, stream_mode="values"):
    final = event

print(final["final_output"]["executive_summary"])
```

---

## Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **Orchestration** | LangGraph (StateGraph + MemorySaver) | DCG pipeline, HITL, retries, fan-out |
| **Object Detection** | YOLOv8 (Ultralytics, custom-trained) | Damage and part detection |
| **Small Object** | SAHI | High-recall detection on large images |
| **Segmentation** | SAM (Segment Anything Model ViT-B) | Precise damage masks |
| **Depth Estimation** | MiDaS | 3D reasoning, deformation index |
| **Multimodal VLM** | Gemini 2.5 Flash | Vehicle type, location enrichment, Golden Frame |
| **Text Generation** | Groq Llama-3.3-70b-versatile | Executive summary, forensic report |
| **Fraud — Forensics** | ELA, FFT, Laplacian variance | AI-generation and screen-capture detection |
| **Fraud — Duplicates** | imagehash (pHash) | Cross-claim image recycling detection |
| **Fraud — Metadata** | exifread | EXIF temporal + GPS + software checks |
| **Graph DB** | NetworkX DiGraph | Multi-image detection fusion |
| **Persistence** | MongoDB Atlas / SQLite | Case storage, chat history, audit logs |
| **UI — User** | Gradio Blocks | 5-tab claimant dashboard (port 7860) |
| **UI — Auditor** | Gradio Blocks + gr.Sidebar | 5-tab admin dashboard (port 7861) |
| **Env** | Google Colab T4 GPU | Primary compute runtime |

---

## Roadmap

- [ ] **Modular Refactor** — migrate monolithic notebook to `src/` package structure
- [ ] **FastAPI backend** — expose pipeline as REST API (`POST /analyze`, `GET /case/{id}`)
- [ ] **WebSocket streaming** — real-time pipeline progress updates
- [ ] **YOLOv9 upgrade** — retrain damage and part detection models
- [ ] **Multi-language reports** — Groq prompt templates for Tamil, Hindi, Telugu
- [ ] **Mobile PWA** — camera-first intake for field adjusters
- [ ] **Docker** — containerised deployment with `docker-compose.yml`
- [ ] **Unit test coverage** — complete `tests/` suite with CI/CD via GitHub Actions

---

## Acknowledgements

- [LangGraph](https://github.com/langchain-ai/langgraph) — Directed Cyclic Graph orchestration
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — Object detection backbone
- [SAHI](https://github.com/obss/sahi) — Small object detection framework
- [Segment Anything (Meta AI)](https://github.com/facebookresearch/segment-anything) — Image segmentation
- [MiDaS (isl-org)](https://github.com/isl-org/MiDaS) — Monocular depth estimation
- [Google Gemini](https://ai.google.dev) — Multimodal VLM for verification
- [Groq](https://groq.com) — Ultra-fast LLM inference
- [Gradio](https://gradio.app) — Dashboard UI framework
- [MongoDB Atlas](https://mongodb.com/atlas) — Cloud database

---

<div align="center">

**Built with 🔬 by the SmartForge team**

*SmartForge v36 · LangGraph DCG · SAHI + SAM + MiDaS · Gemini 2.5 Flash · Groq Llama-3.3-70b · 5-Check Fraud Layer · 3-Strike Fraud Retry · Golden Frame Verification · NetworkX Graph DB · Human-Auditor-Only Approval*

</div>
