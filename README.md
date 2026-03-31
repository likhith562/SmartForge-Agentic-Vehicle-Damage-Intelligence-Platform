<div align="center">

# 🚗 Agentic Vehicle Damage Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Directed%20Cyclic%20Graph-FF6B35?style=for-the-badge)](https://github.com/langchain-ai/langgraph)
[![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash%20VLM-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070b-F55036?style=for-the-badge)](https://groq.com)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://mongodb.com)
[![Gradio](https://img.shields.io/badge/Gradio-5%20Tab%20Dashboard-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Open In Colab](https://img.shields.io/badge/Open%20In-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

<br/>

**An end-to-end autonomous insurance claims processing system** powered by a LangGraph Directed Cyclic Graph orchestrating 13 specialized AI agents — from YOLO/SAM/MiDaS computer vision through Gemini multimodal verification, a 5-check forensic fraud layer, and Groq-generated structured reports — with dual Gradio dashboards and MongoDB persistence.

<br/>

[**📓 Open in Colab**](#quick-start) · [**🏗️ Architecture**](#architecture) · [**📊 Dashboards**](#dashboards) · [**🛡️ Fraud Layer**](#fraud-detection-layer) · [**🚀 Setup**](#installation)

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
  - [System Overview](#system-overview)
  - [Computer Vision Pipeline](#computer-vision-pipeline)
  - [Fraud Detection Layer](#fraud-detection-layer)
  - [Agentic Decision & Reasoning Flow](#agentic-decision--reasoning-flow)
  - [Multi-Image Map-Reduce](#multi-image-map-reduce)
  - [LangGraph State Schema](#langgraph-state-schema)
  - [Graph Topology](#graph-topology)
- [Feature Batches](#-feature-batches)
- [Dashboards](#-dashboards)
  - [User Dashboard (5-Tab)](#user-dashboard-5-tab)
  - [Auditor Dashboard (5-Tab)](#auditor-dashboard-5-tab)
- [Database Schema](#-database-schema)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Configuration](#%EF%B8%8F-configuration)
- [Quickstart](#-quick-start)
- [Tech Stack](#-tech-stack)
- [Roadmap](#-roadmap)

---

## 🔭 Project Overview

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

## 🏗️ Architecture

SmartForge is built around four interconnected subsystems. All four are visualised in the diagrams below.

### System Overview

The top-level architecture orchestrates all agents through a LangGraph state machine. Every node communicates exclusively through `SmartForgeState` — a typed dictionary that flows through the graph and accumulates results without mutation conflicts.

<div align="center">
<img src="assets/diagrams/system_architecture.png" alt="SmartForge System Architecture" width="380">
<br/>
<em>SmartForge System Architecture — Full agentic pipeline from user upload to final claim decision</em>
</div>

<br/>

The pipeline follows two execution paths depending on fraud detection results:

```
User Upload
    │
    ▼
Intake Agent ──► Fraud Detection Layer (5-check) ──► Trust Score < 70 ──► Human Audit System
                                                  │
                                                  ▼ Trust Score ≥ 70 (PASS)
                        ┌──────────────────────────────────────────────────┐
                        │              Agentic AI Core                     │
                        │  Perception Engine (YOLO + SAHI + SAM + MiDaS) │
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
<img src="assets/diagrams/cv_pipeline.png" alt="Computer Vision & Damage Analysis Pipeline" width="420">
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
<img src="assets/diagrams/fraud_layer.png" alt="SmartForge Fraud Detection Layer Flow" width="580">
<br/>
<em>Fraud Detection Layer — 5-check parallel pipeline computing a 0–100 Trust Score</em>
</div>

<br/>

The fraud layer runs **5 independent checks** in parallel and aggregates into a Trust Score. A score below 70 routes to the Human Audit System; ≥ 70 passes to the perception pipeline.

| Check | Method | Score Impact |
|-------|--------|-------------|
| **① Temporal Consistency** | EXIF `DateTimeOriginal` vs `claim_date` | +20 pts pass / −25 pts fail |
| **② GPS Location Consistency** | EXIF GPS vs claim location (Haversine distance) | +20 pts pass / −30 pts fail |
| **③ Software / Source Integrity** | EXIF `ImageSoftware` — Adobe/Canva flags editing | +10 pts pass / −30 pts fail |
| **④ pHash Duplicate Detection** | Perceptual hash vs local fraud DB + SerpAPI (optional) | +20 pts pass / −40 pts duplicate |
| **⑤ AI-Generation & Screen Forensics** | ELA analysis + FFT Moiré pattern detection | +10 pts pass / −35 pts fail |

```
Trust Score = base_100 + Σ(check_adjustments)
Trust Score < 70  →  SUSPICIOUS_HIGH_RISK  →  Human Audit
Trust Score ≥ 70  →  VERIFIED              →  Perception Pipeline
```

> **Note:** `BYPASS_FRAUD = True` skips all checks for demo runs. When a user files an insurance claim in the UI, `BYPASS_FRAUD` is dynamically set to `False` and the full 5-check layer activates.

---

### Agentic Decision & Reasoning Flow

<div align="center">
<img src="assets/diagrams/decision_flow.png" alt="SmartForge Agentic Decision & Reasoning Flow" width="380">
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
9. **Final Decision Logic** — Score ≥ 85 → Auto-Approved; 70–85 → Workshop; < 70 → Manual Review

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
intake ──► fraud ──┬──► map_images ──► cv_worker(×N) ──► fusion ──►┐
                   │   [Batch 2 Send fan-out]                        │
                   │                                                  ▼
                   └────────────────────────────────────► perception
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
| `fraud` | `trust_score < 70` | `human_audit → END` |
| `fraud` | `len(image_paths) > 1` | `map_images` |
| `fraud` | verified, single image | `perception` |
| `health_monitor` | `validation_passed = True` | `verification_v2` |
| `health_monitor` | `validation_passed = False` | `perception_retry` |

---

## 🧩 Feature Batches

SmartForge was developed incrementally in 4 production-ready feature batches:

<details>
<summary><b>Batch 1 — 5-Check Fraud & Integrity Layer</b></summary>

- EXIF temporal consistency check
- GPS Haversine distance validation
- EXIF software source integrity
- Perceptual hash (pHash) duplicate detection with local fraud DB
- FFT Moiré pattern detection + ELA AI-generation forensics (Winston AI / Laplacian fallback)
- `BYPASS_FRAUD` toggle for demo mode; dynamically switched `False` when insurance claim is filed

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

## 📊 Dashboards

SmartForge ships two completely separate Gradio applications running on different ports. **Roles are never mixed in a single UI** — a deliberate security and UX decision.

### User Dashboard (5-Tab)

> Port `7860` · Audience: Vehicle owner / claimant

```
Tab 1 — 📥 Vehicle Intake
  • Vehicle ID (mandatory field, validated)
  • Owner name, vehicle type dropdown
  • Multi-image file upload (activates Batch 2 automatically)
  • Incident date + GPS latitude/longitude
  • Saves session to MongoDB, navigates to Tab 2

Tab 2 — 🔬 Damage Analysis
  • "Run Full Analysis" triggers complete LangGraph pipeline
  • BYPASS_FRAUD=True (fraud gate runs separately in Tab 3)
  • Shows: primary vehicle photo, detection records table
    (✅ Confirmed / 🚩 Rejected / ❓ Pending), pipeline timeline
  • Case status updated: uploaded → analyzed

Tab 3 — 🛡️ Insurance Claim (conditional)
  • Radio: "Yes – file a claim" | "No – assessment only"
  • YES → insurance form reveals (policy number, claim reason, date)
  • YES → BYPASS_FRAUD=False → full 5-check fraud layer re-runs
  • Fraud badge shows trust score + active flags
  • Case status: claim_submitted → fraud_checked → approved/rejected

Tab 4 — 📊 Executive Summary
  • Total Loss banner (red gradient) or Repairable banner (green)
  • Colour-coded health score badge (green ≥80 / amber ≥60 / red <60)
  • Groq-generated executive summary addressed to claimant
  • Financial line-item table with grand total (USD + INR)
  • Claim ruling badge: CLM_APPROVED | CLM_WORKSHOP | CLM_MANUAL | CLM_PENDING
  • Fraud detection result + forensic integrity text

Tab 5 — 💬 AI Chat Assistant
  • gr.ChatInterface powered by Groq Llama-3.3-70b
  • RAG-style: injects current session JSON as system context
  • STRICT RULE: answers only about this vehicle/claim
  • Chat history persisted to MongoDB after each turn
  • Pre-loaded example questions
```

---

### Auditor Dashboard (5-Tab)

> Port `7861` · Audience: Insurance adjuster / compliance auditor  
> Role: **AUDITOR** — no `vehicle_id` filter applied → full case visibility

```
Tab 1 — 🗂️ Case Explorer
  • Search by vehicle_id (partial match), status, fraud-only toggle
  • Live stats cards: Total / Analyzed / Fraud / Approved / Rejected / Pending
  • Click any row → loads full detail panel:
    - Case summary text, vehicle photo, auditor review history
    - final_output JSON viewer (Code component)

Tab 2 — 📋 Insurance Claims
  • All cases where filing_claim=True
  • Columns: Policy No, Filed At, Claim Reason, Cost USD/INR, Ruling, Status, Fraud

Tab 3 — 🚨 Fraud Review
  • All is_fraud=True records
  • Full forensic detail on row select:
    - pHash Hamming distance, matched claim path
    - ELA score, AI probability, method
    - FFT Moiré signals, screen detection confidence
  • Auditor Decision: Confirm Fraud | Clear – Not Fraud | Approve Claim | Reject Claim
  • Decision + note written to auditor_review field, status updated in DB

Tab 4 — 👤 User Management
  • Per-vehicle aggregation: cases, claims filed, fraud flags, total cost, approved/rejected
  • Click user row → full claim history table

Tab 5 — 📋 Audit Logs (Compliance Backbone)
  • MemorySaver checkpoint dump rendered as formatted table:
    Step | Node | Timestamp | Retries | Health Score | Detections | Messages
  • Full agent_trace JSON (Code component)
  • All agent decisions across all cases in sortable table
```

---

## 🗄️ Database Schema

SmartForge uses **MongoDB Atlas** as primary storage with automatic SQLite fallback. One document per case:

```json
{
  "case_id":         "VH001-abc123f",
  "user_id":         "VH001",
  "vehicle_id":      "VH001",
  "images":          ["/content/VH001-abc123f_img0.jpg"],
  "status":          "approved",
  "created_at":      "2026-03-29T04:15:00.000Z",
  "updated_at":      "2026-03-29T05:00:00.000Z",

  "user_data": {
    "owner_name":    "Rajesh Kumar",
    "vehicle_type":  "Car / Sedan / SUV",
    "incident_date": "2026-03-29",
    "incident_lat":  13.0827,
    "incident_lon":  80.2707
  },

  "final_output": {
    "claim_id":                  "CLM-VH001-...",
    "overall_assessment_score":  70,
    "claim_ruling_code":         "CLM_PENDING",
    "confirmed_damage_count":    3,
    "financial_estimate": {
      "total_repair_usd":        1210.00,
      "total_repair_inr_fmt":    "₹100,430",
      "total_loss_flag":         false,
      "disposition":             "REPAIRABLE",
      "line_items":              [...]
    },
    "executive_summary":         "...",
    "forensic_report":           "...",
    "pipeline_trace":            {...}
  },

  "checkpoint_dump": [
    { "step": 10, "node": "loop", "retry_count": 0,
      "health_score": 1.0, "validation_passed": true,
      "n_detections": 9, "n_messages": 11 }
  ],

  "fraud_report": {
    "trust_score":  0,
    "status":       "SUSPICIOUS_HIGH_RISK",
    "flags":        ["RECYCLED_IMAGE: pHash match (Hamming=0)", "SCREEN_CAPTURE"],
    "details": {
      "phash_check":          { "status": "DUPLICATE_DETECTED", "hamming_distance": 0 },
      "ai_generation_check":  { "ela_score": 1.252, "is_ai_generated": false },
      "screen_detection":     { "is_screen": true, "confidence": 0.5 }
    }
  },

  "fraud_hash":      "c8d12bbdb496b268",
  "insurance":       { "filing_claim": true, "policy_number": "POL-2024-001" },
  "agent_trace":     { "intake_agent": {...}, "fraud_agent": {...}, "report_agent": {...} },
  "chat_history":    [["What is my repair cost?", "Your estimated total repair cost is..."]],
  "is_fraud":        false,
  "auditor_review":  { "decision": "Approve Claim", "reviewed_at": "2026-03-29T05:00Z" }
}
```

### Storage Policy

| Data | Stored | Rationale |
|------|--------|-----------|
| `final_output` | ✅ Always | Core claim result |
| `checkpoint_dump` | ✅ Always | Compliance flight recorder |
| `fraud_report` | ✅ Always | Mandatory audit trail |
| `agent_trace` | ✅ Always | Decision transparency |
| `chat_history` | ✅ Per session | AI assistant memory |
| `golden_crops` | ⚡ Selective | Stored only for fraud/critical cases (large overhead) |

### Role-Based Access

```python
# User — sees only their own cases
db_find({"vehicle_id": current_user_id})

# Auditor — sees ALL cases, no filter
db_find({})
```

### Status Pipeline

```
uploaded → analyzed → claim_submitted → fraud_checked → approved / rejected
```

---

## 📁 Repository Structure

```
smartforge-agentic-ai/
│
├── .env.example                    # API key template (GROQ, GEMINI, MONGO_URI)
├── .gitignore                      # Excludes __pycache__, .env, uploaded images, *.pt models
├── LICENSE                         # MIT License
├── README.md                       # This file
├── requirements.txt                # All Python dependencies with pinned versions
├── main.py                         # Entry point — launches both Gradio apps simultaneously
│
├── assets/
│   ├── diagrams/
│   │   ├── system_architecture.png      # Full system architecture overview
│   │   ├── cv_pipeline.png              # Computer Vision & Damage Analysis Pipeline
│   │   ├── fraud_layer.png              # 5-Check Fraud Detection Layer Flow
│   │   ├── decision_flow.png            # Agentic Decision & Reasoning Flow
│   │   └── multi_image_architecture.png # Multi-Image Map-Reduce Architecture
│   └── demo_screenshots/
│       ├── user_tab1_intake.png
│       ├── user_tab2_analysis.png
│       ├── user_tab3_insurance.png
│       ├── user_tab4_summary.png
│       ├── auditor_tab1_cases.png
│       └── auditor_tab3_fraud.png
│
├── data/
│   ├── sample_images/
│   │   ├── car_clean.jpg                # No-damage test image
│   │   ├── car_damaged_front.jpg        # Front bumper damage
│   │   └── car_fraud_screenshot.jpg     # Screen-capture fraud test
│   ├── fraud_hash_db.json               # Local pHash fraud database (seed entries)
│   └── mock_checkpoints.json            # For auditor timeline testing without DB
│
├── notebooks/
│   ├── Vehicle_Damage_Agentic_AI_v31.ipynb  # Latest monolithic Colab notebook
│   └── experiments/
│       ├── yolo_training_experiments.ipynb
│       └── sahi_confidence_tuning.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py              # Loads .env, threshold constants, REPAIR_DATABASE
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   └── mongo_client.py          # db_upsert, db_get, db_find, db_count, db_mark_auditor
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gemini_client.py         # Gemini 2.5 Flash VLM — vehicle type, enrichment, Golden Frame
│   │   └── groq_client.py           # Groq Llama-3.3-70b — executive summary, forensic report
│   │
│   ├── cv/
│   │   ├── __init__.py
│   │   ├── fraud_checks.py          # EXIF, GPS Haversine, pHash, ELA, FFT Moiré
│   │   ├── perception.py            # SAHI slicing, YOLOv8 inference, SAM segmentation
│   │   ├── depth.py                 # MiDaS depth estimation, deformation index
│   │   └── fusion.py                # NetworkX DiGraph — Batch 2 multi-image fusion
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py                 # SmartForgeState TypedDict definition
│   │   ├── workflow.py              # StateGraph builder — nodes, edges, compile
│   │   └── nodes/
│   │       ├── __init__.py
│   │       ├── intake.py            # Image validation, adaptive SAHI confidence
│   │       ├── fraud.py             # 5-check fraud layer + fraud_router
│   │       ├── perception.py        # SAHI → YOLO → SAM → MiDaS pipeline
│   │       ├── gemini_agent.py      # Vehicle type, location enrichment, full-image scan
│   │       ├── false_positive_gate.py  # Non-car domain shift corrections
│   │       ├── health_monitor.py    # Validation + conditional retry routing
│   │       ├── verification_v2.py   # Golden Frame high-res crop verification
│   │       ├── reasoning.py         # Severity classification + financial estimation
│   │       ├── decision.py          # Claim ruling + HITL interrupt
│   │       ├── report.py            # Three-section narrative (Groq)
│   │       ├── map_reduce.py        # map_images, cv_worker, fusion nodes (Batch 2)
│   │       └── human_audit.py       # Terminal fraud node
│   │
│   └── ui/
│       ├── __init__.py
│       ├── theme.py                 # _get_theme, _score_badge, _ruling_badge, _fraud_badge
│       ├── helpers.py               # _pipeline_timeline, _status_stepper, _stat_card
│       ├── user_dashboard.py        # 5-tab user Gradio app (Blocks)
│       └── auditor_dashboard.py     # 5-tab auditor Gradio app (Blocks)
│
└── tests/
    ├── __init__.py
    ├── test_fraud_layer.py          # ELA/pHash/FFT assertions
    ├── test_graph_state.py          # LangGraph routing to human_audit on fraud
    ├── test_financial_engine.py     # Cost estimation assertions
    └── test_db_layer.py             # MongoDB/SQLite upsert/get/find tests
```

---

## 🚀 Installation

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

### Model Downloads

SmartForge requires three pre-trained model files. Download and place them in the project root or set paths in `.env`:

```bash
# YOLOv8 damage segmentation model (custom-trained)
# Place at: data/models/seg-best.pt

# YOLOv8 part detection model (custom-trained)
# Place at: data/models/detect-best.pt

# SAM ViT-B checkpoint (~375 MB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
     -O data/models/sam_vit_b_01ec64.pth
```

> **Note:** The custom YOLO models (`seg-best.pt`, `detect-best.pt`) are not included in this repository. Contact the project maintainers for model access.

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```env
# ── AI API Keys ──────────────────────────────────────────────────────────────
GEMINI_API_KEY=AIza...                # https://aistudio.google.com/app/apikey (free tier)
GROQ_API_KEY=gsk_...                  # https://console.groq.com (free tier)

# ── Database ─────────────────────────────────────────────────────────────────
SMARTFORGE_MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/
# Leave empty → automatic SQLite fallback at /content/smartforge_claims.db

# ── Optional Fraud Extensions ─────────────────────────────────────────────────
SERPAPI_KEY=...                       # https://serpapi.com — reverse image search (100/month free)
WINSTON_AI_KEY=...                    # https://app.gowinston.ai — AI image detection (2000 credits/month)

# ── Model Paths ──────────────────────────────────────────────────────────────
DAMAGE_MODEL_PATH=data/models/seg-best.pt
PART_MODEL_PATH=data/models/detect-best.pt
SAM_CHECKPOINT=data/models/sam_vit_b_01ec64.pth

# ── Pipeline Thresholds ───────────────────────────────────────────────────────
SAHI_CONFIDENCE=0.3
SAHI_SLICE_SIZE=640
SAHI_OVERLAP=0.2
MAX_RETRIES=2
FRAUD_TRUST_THRESHOLD=40
AUTO_APPROVE_THRESHOLD=85
ESCALATION_THRESHOLD=70
TOTAL_LOSS_THRESHOLD=0.75
VEHICLE_VALUE=15000
```

### Key Thresholds

| Variable | Default | Effect |
|----------|---------|--------|
| `FRAUD_TRUST_THRESHOLD` | 40 | Trust scores below this → `SUSPICIOUS_HIGH_RISK` |
| `AUTO_APPROVE_THRESHOLD` | 85 | Health scores at or above → auto-approved |
| `ESCALATION_THRESHOLD` | 70 | Health scores below → workshop inspection |
| `TOTAL_LOSS_THRESHOLD` | 0.75 | Repair > 75% vehicle value → TOTALED |
| `MAX_RETRIES` | 2 | HealthMonitor retry limit before circuit break |
| `SAHI_CONFIDENCE` | 0.3 | Base YOLO confidence (auto-raised for high-gloss vehicles) |

---

## ⚡ Quick Start

### Option A — Google Colab (Recommended)

1. Open `notebooks/Vehicle_Damage_Agentic_AI_v31.ipynb` in Google Colab
2. Set Runtime → T4 GPU
3. Add secrets in the Colab sidebar: `Gemini_API_Key`, `GROQ_API_KEY`, `SMARTFORGE_MONGO_URI`
4. Run cells in order:
   ```
   Cell 0   → Drive mount (optional)
   Cell 1   → Configuration (edit API keys + thresholds)
   Cell 2   → Install dependencies (~3–5 minutes)
   Cell 3   → Imports & environment check
   Cells 4–13 → Pipeline nodes (intake, fraud, perception, agents, graph)
   Cell G1  → Dashboard config (MONGO_URI, theme, share)
   Cell G2  → Database layer (MongoDB/SQLite auto-select)
   Cell G3  → User Dashboard (builds user_demo)
   Cell G4  → Auditor Dashboard + launches both apps
   ```
5. Two public share links appear — one for users, one for auditors

### Option B — Local Launch

```bash
python main.py
# User Dashboard:    http://localhost:7860  (+ public share link)
# Auditor Dashboard: http://localhost:7861  (+ public share link)
```

### Option C — Direct Pipeline Run (no UI)

```python
from src.graph.workflow import graph
from src.graph.state import make_initial_state

state  = make_initial_state("/path/to/car_image.jpg")
thread = {"configurable": {"thread_id": state["job_id"]}}

# Phase 1 — run to HITL interrupt before decision node
for event in graph.stream(state, thread, stream_mode="values"):
    partial = event

# Phase 2 — auto-approve and complete report
for event in graph.stream(None, thread, stream_mode="values"):
    final = event

print(final["final_output"]["executive_summary"])
```

---

## 🔬 Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **Orchestration** | LangGraph (StateGraph + MemorySaver) | DCG pipeline, HITL, retries, fan-out |
| **Object Detection** | YOLOv8 (Ultralytics, custom-trained) | Damage and part detection |
| **Small Object** | SAHI (Slicing Aided Hyper Inference) | High-recall detection on large images |
| **Segmentation** | SAM (Segment Anything Model ViT-B) | Precise damage masks |
| **Depth Estimation** | MiDaS | 3D reasoning, deformation index |
| **Multimodal VLM** | Gemini 2.5 Flash | Vehicle type, location enrichment, Golden Frame |
| **Text Generation** | Groq Llama-3.3-70b-versatile | Executive summary, forensic report |
| **Fraud — Forensics** | ELA, FFT, Laplacian variance | AI-generation and screen-capture detection |
| **Fraud — Duplicates** | imagehash (pHash) | Cross-claim image recycling detection |
| **Fraud — Metadata** | exifread | EXIF temporal + GPS + software checks |
| **Fraud — Reverse Search** | SerpAPI (optional) | Internet duplicate search |
| **Graph DB** | NetworkX DiGraph | Multi-image detection fusion |
| **Persistence** | MongoDB Atlas / SQLite | Case storage, chat history, audit logs |
| **UI — User** | Gradio Blocks | 5-tab claimant dashboard (port 7860) |
| **UI — Auditor** | Gradio Blocks | 5-tab admin dashboard (port 7861) |
| **Env** | Google Colab T4 GPU | Primary compute runtime |

### Python Dependencies

```txt
langgraph>=0.2
langchain-core>=0.3
ultralytics>=8.0
sahi>=0.11
timm
torch>=2.0
torchvision
opencv-python
Pillow
numpy
matplotlib
groq
google-genai
google-generativeai
exifread
imagehash
networkx
requests
gradio>=4.44.0,<5.0
pandas
pymongo
dnspython
```

---

## 🗺️ Roadmap

- [ ] **Modular Refactor** — migrate monolithic notebook to `src/` package structure per repository layout
- [ ] **FastAPI backend** — expose pipeline as REST API (`POST /analyze`, `GET /case/{id}`)
- [ ] **WebSocket streaming** — real-time pipeline progress updates in dashboard
- [ ] **MongoDB full migration** — replace all SQLite fallback paths; add Atlas Search for text queries
- [ ] **YOLOv9 upgrade** — retrain damage and part detection models
- [ ] **Multi-language reports** — Groq prompt templates for Tamil, Hindi, Telugu
- [ ] **Mobile PWA** — camera-first intake for field adjusters
- [ ] **Webhook notifications** — Slack/email alerts on fraud detection or claim approval
- [ ] **Unit test coverage** — complete `tests/` suite with CI/CD via GitHub Actions
- [ ] **Docker** — containerised deployment with `docker-compose.yml`


## 🙏 Acknowledgements

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

*SmartForge v31 · LangGraph DCG · SAHI + SAM + MiDaS · Gemini 2.5 Flash · Groq Llama-3.3-70b · 5-Check Fraud Layer · Golden Frame Verification · NetworkX Graph DB*

</div>
