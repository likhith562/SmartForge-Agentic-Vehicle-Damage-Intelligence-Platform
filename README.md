<div align="center">

# 🚗 Agentic Vehicle Damage Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Directed%20Cyclic%20Graph-FF6B35?style=for-the-badge)](https://github.com/langchain-ai/langgraph)
[![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash%20VLM-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070b-F55036?style=for-the-badge)](https://groq.com)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://mongodb.com)
[![Gradio](https://img.shields.io/badge/Gradio-5%20Tab%20Dashboard-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![Open In Colab](https://img.shields.io/badge/Open%20Notebook%20%7C%20Gradio%20Live%20Demo-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/your-username/smartforge-agentic-ai/blob/main/notebooks/Vehicle_Damage_Agentic_AI_v36_gradio.ipynb)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
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
  - [Agentic Decision and Reasoning Flow](#agentic-decision-and-reasoning-flow)
  - [Multi-Image Map-Reduce](#multi-image-map-reduce)
  - [LangGraph State Schema](#langgraph-state-schema)
  - [Graph Topology](#graph-topology)
- [Feature Batches](#feature-batches)
- [Dashboards](#dashboards)
  - [User Dashboard 5-Tab](#user-dashboard-5-tab)
  - [Auditor Dashboard 5-Tab](#auditor-dashboard-5-tab)
- [Database Schema](#database-schema)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Tech Stack](#tech-stack)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)
- [License](#license)

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
                                                  |  Trust Score ≥ 40 (PASS)
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

**Key design patterns:**

| Pattern | Where used |
|---------|-----------|
| LangGraph cyclic graph with circuit breaker | `health_monitor` → `perception_retry` loop |
| LangGraph `Send` API for parallel fan-out | `map_images_node` — one worker per uploaded photo |
| Human-in-the-Loop interrupt | `interrupt_before=["decision"]` on high-value claims |
| NetworkX in-memory graph DB | `fusion_node` — cross-image damage deduplication |
| MemorySaver checkpointing | Full state dump at every super-step |

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

**Adaptive SAHI confidence** is computed at intake time by `_analyse_image_conditions()`:

- V-channel variance > 3000 → `high_reflection` → confidence raised to **0.45** (shiny/reflective surfaces generate specular false positives)
- V-channel variance < 1000 → `low_contrast` → confidence lowered to **0.25** (dark/matte surfaces hide subtle damage)
- V-channel variance 1000–3000 → `normal` → use default **0.30**

On perception retry, confidence is further reduced by 20% per attempt: `conf = base_conf * (0.8 ** retry)`.

An **adaptive downsampling check** also runs: if scaling a large image to the 4096px cap would reduce the short side below 2× the SAHI tile size, downsampling is skipped to preserve scratch-level detail (`skip_downsampling = True`).

---

### Fraud Detection Layer

<div align="center">
<img src="assets/diagrams/fraud_layer.png" alt="SmartForge Fraud Detection Layer Flow" width="100%">
<br/>
<em>Fraud Detection Layer — 5-check pipeline computing a 0–100 Trust Score</em>
</div>

<br/>

The fraud layer runs **5 independent checks** in sequence and aggregates a Trust Score. A score below 40 routes to the Human Audit System; ≥ 40 passes to the perception pipeline.

| Check | Method | Score Impact |
|-------|--------|-------------|
| **① Temporal Consistency** | EXIF `DateTimeOriginal` vs `claim_date` — photos predating the claimed accident date | +20 pts pass / −25 pts fail |
| **② GPS Location Consistency** | EXIF GPS coordinates vs claim loss location (Haversine great-circle distance); max drift 50 km | +20 pts pass / −30 pts fail |
| **③ Software / Source Integrity** | EXIF `ImageSoftware` tag — Adobe/Photoshop/Lightroom/Canva/GIMP → editing flag; mobile camera hints pass | +10 pts pass / −30 pts fail |
| **④ pHash Duplicate Detection** | Perceptual hash (pHash) vs local `fraud_hash_db.json` using Hamming distance ≤ 8; optional SerpAPI Google Lens reverse-image search | +20 pts pass / −40 pts duplicate |
| **⑤ AI-Generation and Screen Forensics** | 3-stage unbreakable stack: Winston AI API → ELA (Error Level Analysis) → Laplacian variance; FFT Moiré (mid-freq energy ratio > 0.38) + colour-banding histogram (> 30 empty R-bins) for screen-capture detection | +10 pts pass / −35 pts fail |

```
Trust Score = 0 + Σ(check_adjustments)    (clamped 0–100)
Trust Score <  40  →  SUSPICIOUS_HIGH_RISK  →  human_audit → END
Trust Score ≥  40  →  VERIFIED              →  perception pipeline
```

**3-Strike photo retry limit:** Fraud-flagged uploads can be retried up to `MAX_FRAUD_RETRIES` (default 3) times. On the third failure the case is permanently closed. All attempt timestamps are persisted to the DB.

> **`BYPASS_FRAUD = True`** (Cell 1 default) skips all 5 checks instantly for demos or EXIF-less images. When a user selects "Yes – I want to file a claim" in Tab 2 of the Gradio UI, `BYPASS_FRAUD` is dynamically set to `False` and the full layer activates.

---

### Agentic Decision and Reasoning Flow

<div align="center">
<img src="assets/diagrams/decision_flow.png" alt="SmartForge Agentic Decision & Reasoning Flow" width="100%">
<br/>
<em>Agentic Decision & Reasoning Flow — From CV candidate output to final claim ruling</em>
</div>

<br/>

After CV detections pass the fraud gate, the agentic reasoning stack takes over:

1. **Gemini Verification Agent** — 3 batched API calls (Task 1.2 improvement: Tasks 2+3 fused into 1 call):
   - **Call A** — vehicle type classification: `vehicle_type` (car/2W/3W/truck/unknown), `vehicle_make_estimate`, `confidence`
   - **Call B** — batch location enrichment + low-confidence verification in a single call (replaces N+M individual calls)
   - **Call C** — full-image scan for missed damages with strict rules (body panels only, ≥ 65% confidence, no re-reporting existing detections)

2. **False Positive Gate** — 4-layer rejection stack targeting domain-shift errors (model trained on cars misfires on bikes/3W):
   - Gate 1: Vehicle-type confidence floor — non-car detections require conf ≥ 0.60
   - Gate 2: Minimum area gate — `area_ratio < 0.003` on non-car → noise
   - Gate 3: Depth flatness gate — `deformation < 0.001 AND area < 0.005` → normal surface; 2D types (Scratch/Paint chip/Flaking/Corrosion) are **exempt** since MiDaS shows near-zero deformation for surface-only damage
   - Gate 4: Gemini veto — `gemini_verified == False` → always reject
   - **Gemini positive override:** `gemini_verified == True` clears Gate 3 rejections (Senior Adjuster Rule)

3. **Golden Frame Verification (Batch 3)** — crops high-resolution bounding box regions (25% context padding, minimum 128px) from the original full-resolution source image and sends to Gemini with a structured forensic schema:
   - `is_physical_damage`, `confidence_score`, `damage_type_refined`, `severity_index`, `part_structurally_compromised`, `repair_recommendation`, `technical_reasoning`
   - Confidence gate: Gemini confidence < 0.55 → `rejected_low_confidence`
   - `is_physical_damage = False` → `rejected_false_positive` (shadows/reflections/dirt removed)
   - Multi-angle cross-verification: if `visibility_count > 1`, secondary crop from a different angle sent alongside for 3D depth consistency

4. **Damage Classification** — `compute_severity()` uses deformation index + area ratio + damage type priors to assign Low/Medium/High severity and Cosmetic/Functional/Moderate category

5. **Financial Intelligence (Batch 4)** — Mitchell/Audatex-style `REPAIR_DATABASE` matrix:
   - Minor/Moderate → `REPAIR/PAINT` (paint cost + 2h labour)
   - Severe/Critical → `REPLACE` (replace cost + 4h labour)
   - Fuzzy part name matching (e.g. "Left Headlight" → "Headlight" entry)
   - Total Loss check: `grand_total_usd > VEHICLE_VALUE × TOTAL_LOSS_THRESHOLD` → `TOTALED`

6. **Final Decision Logic** — claim ruling is always one of:
   - `CLM_PENDING` — AI assessment complete, awaiting auditor sign-off
   - `CLM_WORKSHOP` — score < `ESCALATION_THRESHOLD` (70) or High-severity damage, workshop inspection required
   - `CLM_MANUAL` — fraud flags or unconfirmed detections present, immediate manual forensic review

**The AI never auto-approves.** Final `CLM_APPROVED` status can only be set by a human auditor via the Auditor Dashboard.

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
  - All DETECTION nodes sharing the same PART node = same physical damage
  - Golden Record = highest-confidence detection (best-angle view)
  - visibility_count = how many images confirmed this damage
  - seen_in_indices = exact image indices for full audit trail
  - primary_image_idx = which image gives the best (highest-conf) view

Fraud Loop Detection:
  - Identical bounding boxes across DETECTION nodes linked to DIFFERENT IMAGE nodes
    → Image Recycling Fraud loop flagged in fraud_report
```

Query the `claims_graph` after a run:

```python
import networkx as nx

# List all image nodes
images = [(n, d) for n, d in claims_graph.nodes(data=True) if d.get("node_type") == "image"]
print("Images:", images)

# Find all detections on a specific part
part_node = "PART_Front Bumper"
detections = list(claims_graph.predecessors(part_node))
print("Detections on Front Bumper:", detections)

# Trace a specific detection back to its source image
det_id = "W0-001"
images_containing = list(claims_graph.predecessors(det_id))
print(f"{det_id} came from:", images_containing)

# Check for recycling loops
for n, d in claims_graph.nodes(data=True):
    if d.get("fraud_recycling_flag"):
        print(f"Recycling detected on: {n}")
```

---

### LangGraph State Schema

The `SmartForgeState` TypedDict is the single source of truth flowing through every node. All node outputs are returned as **partial state dicts** and merged by LangGraph — no node mutates state directly. The `Annotated[List, operator.add]` reducer on list fields enables safe parallel fan-out without overwrite conflicts.

```python
class SmartForgeState(TypedDict):
    # ── Message History (reducer: append, never overwrite) ────────────────────
    messages:                Annotated[List[dict], operator.add]

    # ── Workflow Context ──────────────────────────────────────────────────────
    image_path:              str
    image_bgr:               Optional[object]   # numpy BGR array (freed after perception)
    image_rgb:               Optional[object]   # numpy RGB array
    raw_detections:          List[dict]         # PerceptionAgent output
    depth_map:               Optional[object]   # MiDaS output
    damages_output:          List[dict]         # ReasoningAgent output
    final_output:            Optional[dict]     # Complete result

    # ── Gemini VLM Agent ─────────────────────────────────────────────────────
    vehicle_type:            str                # "car" | "2W" | "3W" | "unknown"
    vehicle_type_confidence: float             # 0.0–1.0
    vehicle_make_estimate:   str               # e.g. "sedan-class", "hatchback"
    gemini_agent_ran:        bool
    gemini_discovered_count: int               # damages found by Task 4 full-image scan

    # ── Adaptive Intake Analysis ──────────────────────────────────────────────
    adaptive_sahi_conf:      float             # computed from V-channel variance
    scene_type:              str               # "high_reflection" | "normal" | "low_contrast" | "unknown"

    # ── Validation Metrics ────────────────────────────────────────────────────
    health_score:            float
    validation_passed:       bool
    validation_errors:       List[str]
    retry_count:             int               # circuit breaker counter

    # ── Batch 3: Golden Frame ─────────────────────────────────────────────────
    verified_damages:        Annotated[List[dict], operator.add]
    golden_crops:            List[dict]        # crop metadata for audit trail

    # ── Batch 2: Multi-Image Map-Reduce ──────────────────────────────────────
    image_paths:             List[str]
    all_raw_detections:      Annotated[List[dict], operator.add]   # fan-out accumulator
    fused_detections:        Annotated[List[dict], operator.add]   # post-fusion records

    # ── Batch 1: Fraud Layer ──────────────────────────────────────────────────
    is_fraud:                bool
    fraud_attempts:          int               # 3-strike counter
    fraud_report:            Optional[dict]
    claim_date:              str
    claim_lat:               float
    claim_lon:               float

    # ── Batch 4: Financial Intelligence ──────────────────────────────────────
    pipeline_stability_flag: str              # "Stable" | "Unstable" | "CircuitBreaker"
    total_loss_flag:         bool
    financial_estimate:      Optional[dict]

    # ── Audit Metadata ────────────────────────────────────────────────────────
    job_id:                  str
    vehicle_id:              str
    policy_id:               str
    pipeline_trace:          dict
    started_at:              str
```

---

### Graph Topology

```
                    ┌─────────────────────────────┐
                    │      LANGGRAPH STATE        │
                    │  TypedDict — single source  │
                    │  of truth for all agents    │
                    └──────────────┬──────────────┘
                                   │
intake ──► fraud ──┬──► map_images ──► cv_worker(×N) ──► fusion  ──►┐
                   │   [Batch 2 Send fan-out]                       │
                   │                                                ▼
                   └──────────────────────────────────────►    perception
                                                                    │
                                                 human_audit ◄─SUSPICIOUS
                                                      │
                                                     END
                                                                    │ VERIFIED
                                                              gemini_agent
                                                                    │
                                                        false_positive_gate
                                                                    │
                                                          health_monitor ──► perception_retry ──┐
                                                                    │                           │
                                                                    │◄──────────────────────────┘
                                                                    │ (max 2 retries, then circuit break)
                                                              verification_v2
                                                              [Batch 3 Golden Frame]
                                                                    │
                                                               reasoning
                                                               [Batch 4 Financial]
                                                                    │
                                                         decision [HITL interrupt]
                                                                    │
                                                                 report ──► END
                                   │
                    ┌──────────────▼──────────────┐
                    │    MEMORYSAVER CHECKPOINTER │
                    │  State persisted at every   │
                    │  super-step — full audit    │
                    │  trail dumpable at any time │
                    └─────────────────────────────┘
```

**Conditional edges:**

| Source | Condition | Target |
|--------|-----------|--------|
| `fraud` | `trust_score < 40` | `human_audit → END` |
| `fraud` | `len(image_paths) > 1` | `map_images` |
| `fraud` | verified, single image | `perception` |
| `health_monitor` | `validation_passed = True` | `verification_v2` |
| `health_monitor` | `validation_passed = False AND retry < MAX_RETRIES` | `perception_retry` |
| `health_monitor` | `validation_passed = False AND retry >= MAX_RETRIES` | `verification_v2` (circuit break) |

---

## Feature Batches

SmartForge was developed incrementally in 4 production-ready feature batches:

<details>
<summary><b>Batch 1 — 5-Check Fraud & Integrity Layer</b></summary>

- EXIF temporal consistency check — `_parse_exif_datetime()` compares `DateTimeOriginal` to claim date; delta_days > 1 → TEMPORAL_MISMATCH
- GPS Haversine distance validation — `_haversine_km()` great-circle formula; flags drift > `FRAUD_GPS_MAX_DISTANCE_KM` (50 km default)
- EXIF software source integrity — known editing tools (Adobe/Photoshop/Lightroom/Canva/GIMP/Snapseed/PicsArt) → −30 pts; mobile camera hints → +10 pts; unknown software → warn flag
- Perceptual hash (pHash) duplicate detection — `imagehash.phash()` + Hamming distance ≤ `PHASH_HAMMING_THRESHOLD` (8); auto-enrols new images into `fraud_hash_db.json` for future cross-claim detection
- Optional SerpAPI Google Lens reverse-image check — finds images recycled from the internet; handles 401/429 gracefully
- 3-stage unbreakable AI-generation detection: Winston AI API (Stage 1) → ELA forensics `perform_ela_check()` (Stage 2) → Laplacian variance smoothness check (Stage 3); never crashes — each stage has independent try/catch
- Screen-capture detection: FFT Moiré mid-frequency energy ratio > 0.38 + colour-banding histogram comb (> 30 empty R-bins in non-tail region)
- `BYPASS_FRAUD = True` for demo mode; dynamically switched to `False` when user selects "Yes – I want to file a claim"
- **3-Strike photo retry limit** — `MAX_FRAUD_RETRIES = 3`; permanently closes case on third failure with `FRAUD_MAX_RETRIES_EXCEEDED`

</details>

<details>
<summary><b>Batch 2 — Multi-Image Map-Reduce + NetworkX Graph DB</b></summary>

- LangGraph `Send` API fan-out to N parallel `cv_worker` nodes (one per uploaded image)
- Each worker runs lightweight intake + SAHI detection (skips SAM/MiDaS/Gemini — those run after fusion on the best frame)
- Stamps every detection with `source_image_index` and `source_image_path` for complete traceability
- `fusion_node` builds `nx.DiGraph` with IMAGE, PART, and DETECTION node types and contains/located_on edges
- Part-based deduplication: all detections on the same PART node → single Golden Record (highest confidence)
- Multi-image metadata: `visibility_count`, `seen_in_indices`, `primary_image_idx`, `fused_from_count`, `is_fused`
- Image Recycling Fraud loop detection: identical bounding boxes across multiple IMAGE nodes → `fraud_recycling_flag`
- `claims_graph` exposed globally for post-run audit queries (see Graph Topology section)
- Fraud recycling flags bubble up to `fraud_report.flags` list and can trigger `SUSPICIOUS_HIGH_RISK`

</details>

<details>
<summary><b>Batch 3 — Golden Frame Verification</b></summary>

- `get_high_res_crop()` extracts padded bounding box from the full-resolution original source image (not the resized version sent to SAHI)
- 25% context margin enforced; minimum 128px side length enforced by padding in all four directions
- `_call_gemini_with_crop()` sends crop with a structured forensic prompt — forces strict JSON schema response: `is_physical_damage`, `confidence_score`, `damage_type_refined`, `severity_index`, `part_structurally_compromised`, `repair_recommendation`, `technical_reasoning`
- Confidence gate: Gemini confidence < `GOLDEN_FRAME_CONFIDENCE_MIN` (0.55) → `rejected_low_confidence`
- `is_physical_damage = False` → `rejected_false_positive` (shadows/reflections/dirt removed from claim)
- Multi-angle cross-verification: if `visibility_count > 1`, picks a different `seen_in_indices` entry, generates secondary crop, passes alongside primary in prompt for 3D depth consistency note
- Rejected detections are NOT deleted — marked `is_verified = False`, retained for full audit trail
- Crops saved to `/content/golden_crops/` directory; metadata stored in `golden_crops` state field
- Batch 3 intercepts the flow between `health_monitor` and `reasoning` for both single-image and multi-image paths
- Cost efficiency: Batch 2 fusion de-duplicates first, so Gemini runs once per physical damage (not once per photo)

</details>

<details>
<summary><b>Batch 4 — Financial Intelligence Engine</b></summary>

- `REPAIR_DATABASE` dict with 12 named vehicle parts + `_default` fallback; each entry has `replace` (USD), `paint` (USD), `labor_per_hour` (USD)
- `SEVERITY_TO_ACTION`: Minor/Moderate → `REPAIR/PAINT`; Severe/Critical → `REPLACE`
- `_get_repair_data()` tries exact match first, then fuzzy substring match, then falls back to `_default`
- Cost formula: REPAIR = `paint + (labor_per_hour × 2h)`; REPLACE = `replace + (labor_per_hour × 4h)`
- Severity preference: uses `severity_gemini` from Golden Frame verification when available, falls back to CV-derived `severity`
- Total Loss check: `grand_total_usd > VEHICLE_VALUE × TOTAL_LOSS_THRESHOLD` → `total_loss_flag = True` → `disposition = TOTALED`
- All costs output in USD and INR (configurable `USD_TO_INR` exchange rate, default 83)
- `line_items` list included in `financial_estimate` for line-item table display in dashboard
- Groq three-section report: Section 1 (Executive Summary for claimant, non-technical); Section 2 (Forensic Integrity report for legal, factual); Section 3 (Detailed Estimate — always rule-based, LLM not used for numbers to prevent hallucination)
- Score normalization: 0 confirmed damages after all gates → `overall_assessment_score = 100`, `inspection_recommendation = "No Repair Required"`

</details>

---

## Dashboards

SmartForge ships two completely separate Gradio applications running on different ports. **Roles are never mixed in a single UI** — a deliberate security and UX decision.

### User Dashboard 5-Tab

> Port `7860` · Audience: Vehicle owner / claimant

```
Tab 1 — 📥 1 · Vehicle Intake
  • Vehicle ID (mandatory — validated; e.g. VH001 or TN-09-AB-1234)
  • Owner / Claimant Name
  • Vehicle Type dropdown:
      Auto-Detect (Gemini VLM) | Car / Sedan / SUV |
      2-Wheeler (Bike/Scooter) | 3-Wheeler (Auto)
  • Incident Date — native HTML5 <input type="date"> (capped at today;
    value is synced to a hidden Gradio Textbox via JS preprocessing
    on the button click so the server always receives the correct date)
  • Incident Location — interactive Leaflet map (OpenStreetMap tiles):
      - Address / city search bar with Nominatim autocomplete:
          · 300 ms debounce to avoid hammering the Nominatim API
          · Keyboard navigation: ↑↓ arrows, Enter to select, Esc to dismiss
          · Outside-click dismisses the dropdown
          · Top 5 results displayed with name + detail lines
      - 🌐 GPS button — calls browser Geolocation API, centres map on
        the user's current position at zoom 16
      - Draggable marker — click anywhere on the map or drag the existing
        pin; coordinate bar updates live to 6 decimal places
      - "✅ Confirm Location" button — reads lat/lon from the iframe via
        window.parent.claimLat / window.postMessage (with same-origin
        fallback), writes values to the Gradio Textbox fields that are
        passed to the fraud GPS consistency check
      - Map rendered in a sandboxed iframe with Leaflet 1.9.4 and a
        toolbar layout: search bar | GPS button | map canvas | coord bar
  • Photo upload — multi-file drag-and-drop (JPG/PNG); multiple images
    automatically activate Batch 2 Multi-Image Map-Reduce
  • "→ Save & Proceed to Insurance Preference" saves all data to DB
    (status = uploaded) and navigates to Tab 2
  • Intake Status textbox shows success summary or validation errors

Tab 2 — 🛡️ 2 · Insurance Preference
  • Determines whether fraud checks activate during Damage Analysis —
    saved to DB BEFORE the pipeline runs so the pipeline reads it
  • Info banner explains the 3-attempt fraud retry limit upfront
  • Radio:
      "Yes – I want to file a claim"   → BYPASS_FRAUD=False in Tab 3
      "No – damage assessment only"    → BYPASS_FRAUD=True in Tab 3
  • YES → reveals insurance claim form:
      - Policy Number * (mandatory)
      - Accident Date (auto-filled from DB record; editable)
      - Claim Reason * (mandatory)
      - Additional Notes (optional — FIR number, witness contacts, etc.)
      - Fraud warning banner explaining fraud checks and retry limit
  • NO → assessment-only mode; BYPASS_FRAUD=True in Tab 3
  • "✅ Save Preference & Proceed" writes insurance data + preference
    to DB (status = pref_saved) and navigates to Tab 3
  • Accident date auto-populates from the Tab 1 DB record when switching
    to this tab (via u_tabs.change handler)

Tab 3 — 🔬 3 · Damage Analysis
  • "🔍 Run Full Analysis" triggers the complete LangGraph pipeline:
      intake → fraud → perception → gemini_agent → false_positive_gate
      → health_monitor → verification_v2 → reasoning → decision → report
  • Fraud logic:
      wants_insurance=True AND BYPASS_FRAUD=False → full 5-check layer
      wants_insurance=False OR BYPASS_FRAUD=True  → fraud skipped
  • 3-Strike enforcement:
      - fraud_attempts counter read from DB before each analysis
      - If fraud_attempts >= MAX_FRAUD_RETRIES → "CASE PERMANENTLY
        CLOSED" message returned immediately; no pipeline run
      - On fraud flag: increment counter, show retries_left to user,
        route to fraud_flagged status
      - On 3rd failure: permanent closure message, status = rejected
  • Status Stepper (HTML progress bar across top):
      uploaded → pref_saved → analyzed → claim_submitted →
      fraud_flagged → approved → rejected
      Active step highlighted in brand colour; done steps show ✓
  • Pipeline Status textbox — scrolling agent-by-agent log
  • Primary Vehicle Photo viewer (left column, 260px height)
  • ⚡ Pipeline Timeline (right column) — colour-coded node tiles:
      Done nodes: green background with ✓ overlay badge
      Idle nodes: grey background
      Hover: slight elevation (translateY + shadow)
      Nodes shown: intake | fraud | perception | gemini_agent |
        false_positive_gate | health_monitor | verification_v2 |
        reasoning | decision | report
      HealthMonitor retries: amber note badge below the timeline
        (e.g. "🔄 1 HealthMonitor retry(s)")
  • 🔎 Detection Records table:
      Columns: ID | Type | Location | Severity | Conf | Status
      Status values:
        ✅ Confirmed  — verification_status in confirmed set, not rejected
        🚩 Rejected   — rejected=True from any gate
        ❓ <status>   — pending/other
  • After completion: auto-navigates to Tab 4 (Executive Summary) via
    deferred .then() chained onto the analysis button click event
  • Navigation hint box at bottom of tab

Tab 4 — 📊 4 · Executive Summary
  • "🔄 Refresh Report" button — re-loads from DB for current session
    (use this if Tab 4 opened before analysis finished)
  • Status Stepper refreshed on load
  • Pipeline Timeline auto-refreshed via deferred .then() from Tab 3
  • Executive Summary textbox:
      - Groq Llama-3.3-70b generated, 2–3 sentences, non-technical
      - Prefixed "Dear <owner_name>," when owner name was entered
      - Copyable (show_copy_button=True), scrollable, max 25 lines
  • Health Score badge (right column):
      ≥ 80 → green badge (sf-ok palette)
      ≥ 60 → amber badge (sf-warn palette)
      < 60  → red badge (sf-err palette)
      Format: large score value + "/100" suffix
  • Claim Ruling badge:
      CLM_PENDING   ⏳  Score OK, no unconfirmed detections — AI
                        assessment complete, awaiting auditor approval
      CLM_WORKSHOP  🔧  score < ESCALATION_THRESHOLD or High severity —
                        workshop inspection required before settlement
      CLM_MANUAL    👁️  fraud flags OR unconfirmed detections present —
                        immediate manual forensic audit required
      Badge shows: icon + code | processing_status sub-line |
        claim_ruling text explanation
  • Total Loss / Repairable banner (displayed above estimate table):
      TOTALED: gradient red banner with USD/INR amounts and percentage note
      REPAIRABLE: gradient green banner with disposition and totals
  • 💰 Line-Item Repair Estimate table:
      Columns: Part | Action | Severity | Cost (USD) | Cost (INR)
      Grand total row appended at bottom
      Click-to-sort on all headers (▲ asc / ▼ desc / ⇅ unsorted)
  • 🛡️ Fraud Detection badge:
      VERIFIED: green badge, trust score, no flag list
      SUSPICIOUS_HIGH_RISK: red badge, trust score, bulleted flag list
  • 🔬 Forensic Integrity textbox:
      Groq-generated legal-grade paragraph: ELA score, Laplacian
      variance, SerpAPI result, AI detection method, checks_run count,
      trust verdict; copyable; scrollable

Tab 5 — 💬 5 · AI Assistant
  • gr.ChatInterface powered by Groq Llama-3.3-70b-versatile
  • System context injected from DB on every message turn:
      vehicle_id, vehicle_type, health_score, ruling, ruling_code,
      damages (first 5), financial_estimate, fraud_status, trust_score,
      fraud_attempts, insurance details, owner_name, case_status
  • Strict scope rule: "Answer ONLY about this vehicle and claim.
    Never reveal other users' data."
  • Sends 6-turn rolling history to Groq on each message for coherent
    multi-turn conversation
  • Chat history persisted to DB after each reply
  • Pre-loaded example questions:
      "What damages were found on my vehicle?"
      "What is my total repair cost in INR?"
      "Should I file an insurance claim?"
      "Explain the fraud detection result."
      "How many fraud attempts have I used?"
      "What is my vehicle health score?"
```

---

### Auditor Dashboard 5-Tab

> Port `7861` · Audience: Insurance adjuster / compliance auditor
> Role: **AUDITOR** — no `vehicle_id` filter applied → full case visibility across all users

**AI Auditor Sidebar** — a `gr.Sidebar` panel positioned on the right, collapsed by default. Contains a full `gr.Chatbot` (Groq Llama-3.3-70b) that auto-injects live DB context on every message:

- 15 most recent cases: vehicle ID, status, score, ruling, cost, fraud status, trust score, flag count, policy number
- System-wide counts: total / analyzed / fraud / approved / rejected / pending
- 5 most recent fraud-flagged cases with their active flag descriptions
- Supports multi-turn conversation with 6-turn rolling history window
- Available from every tab without leaving the current view

```
Tab 1 — 🗂️ 1 · Case Explorer
  • Search controls:
      - Vehicle ID textbox (partial match, case-insensitive regex in MongoDB)
      - Status dropdown: All | uploaded | analyzed | claim_submitted |
        fraud_checked | approved | rejected
      - 🚨 Fraud Only checkbox
      - 🔍 Search button
  • Stats cards row (auto-loads on page open; refreshes whenever Tab 1
    is selected via aud_tabs.select handler):
      Total Cases | Analyzed | Fraud Flagged | Approved | Rejected | Pending Review
      Each card is colour-coded: brand / info / error / ok / warn
  • Results table — 10 columns (sortable headers, sticky header row):
      Case ID | Vehicle ID | Status | Score | Damages |
      Cost (USD) | Fraud Status | Trust | Ruling | Created
      Click any row to populate the Case Detail panel below
  • Case Detail panel (populated on row click via .select event):
      - Summary textbox (scrollable, copyable):
          Case ID, Vehicle ID, Owner, Status, Created
          Analysis: health score, ruling code, confirmed damages,
            total cost USD, total cost INR
          Fraud: status, trust score, flags count, each flag bulleted
          Insurance: filing_claim, policy number, claim reason
          Damage Detections list: icon + ID + type + location +
            severity + confidence for every detection
      - Vehicle Photo viewer (primary image, 220px height)
      - Auditor Review textbox (prior decision, note, timestamp)
      - final_output JSON viewer (Code component with JSON syntax
        highlighting, truncated to 3000 chars for performance)

Tab 2 — 📋 2 · Insurance Claims
  • Status filter dropdown: All | claim_submitted | approved | rejected
    — table refreshes immediately on dropdown change (no button needed)
  • "🔄 Load Claims" button for manual refresh
  • Summary markdown: total filed claims, approved count, rejected count, pending count
  • Claims table — 10 columns (sortable, sticky header):
      Case ID | Vehicle | Policy No | Filed At | Claim Reason |
      Cost (USD) | Cost (INR) | Ruling | Status | Fraud
      Click any row → auto-fills Case ID in the process panel below
  • ⚖️ Process a Claim panel:
      - Case ID to Process textbox (auto-filled on row click, or manual
        entry for cases not visible in current filter view)
      - ✅ Approve Claim button:
          Guard: if rec.is_fraud=True → "Cannot approve fraud-flagged
          case. Clear fraud flag in Fraud Review tab first."
          Otherwise: sets status = approved + writes auditor_review to DB
      - ❌ Reject Claim button:
          Sets status = rejected + writes auditor_review to DB
      - Action Result textbox: success confirmation or specific error

Tab 3 — 🚨 3 · Fraud Review
  • "🔄 Load Fraud Cases" button — loads all is_fraud=True records plus
    status=rejected records; deduplicates by case_id
  • Fraud table — 10 columns (sortable, sticky header):
      Case ID | Vehicle | Trust Score | Fraud Status | Flags |
      pHash Match | ELA Score | Screen Detect | Auditor | Created
      Click any row to load forensic detail and set _selected_case_id
  • Full Fraud Analysis textbox (scrollable, copyable, 18 lines):
      Case ID, trust score, status, checks_run, checked_at
      ── FLAGS ──
        Bulleted list of all active fraud flags
      ── pHASH CHECK ──
        Status (UNIQUE / DUPLICATE_DETECTED / HASH_ERROR)
        pHash hex string, matched_claim path, Hamming distance
      ── AI-GENERATION (ELA) ──
        is_ai_generated, ai_probability, ela_score
        Method (winston_ai_api / ela_forensics / laplacian_variance)
        Reasoning string from the detection method
      ── SCREEN DETECTION ──
        is_screen (True/False), confidence score, active signals list
      ── AUDITOR REVIEW ──
        Prior decision JSON dump (or "None yet.")
  • ⚖️ Auditor Decision panel:
      - Radio: Confirm Fraud | Clear — Not Fraud |
                Approve Claim | Reject Claim
      - Note textbox (optional — persisted to auditor_review.note
        for permanent audit trail)
      - "💾 Save Auditor Decision" button:
          db_mark_auditor(case_id, decision, note) called
          Status mapping: Confirm Fraud → rejected;
            Clear — Not Fraud → analyzed;
            Approve Claim → approved;
            Reject Claim → rejected
          Refreshes fraud table after save

Tab 4 — 👤 4 · User Management
  • "🔄 Load Users" button — aggregates all DB records by vehicle_id
  • Summary markdown: unique vehicle / user count
  • Users table — 8 columns (sortable):
      Vehicle / User | Cases | Claims Filed | Fraud Flags |
      Total Cost USD | Approved | Rejected | Last Activity
      Total Cost USD = sum of all financial_estimate amounts for vehicle
      Click any row to load claim history for that vehicle
  • Claim History panel (populated on row click):
      - Header: "Claim history for <vehicle_id> — N case(s):"
      - History table — 7 columns:
          Case ID | Status | Score | Ruling | Cost (USD) | Fraud | Created

Tab 5 — 📊 5 · Audit Logs
  • Vehicle ID filter textbox (blank = latest 50 cases across all users)
  • "🔄 Load Logs" button
  • 📌 MemorySaver Checkpoint Timeline textbox (copyable, 12 lines):
      Fixed-width table: Step | Node | Timestamp | Retries |
        Health | Dets | Msgs
      One row per checkpoint entry — shows exact timing, retries,
      health score, detection count, message count at each super-step
      This is the compliance flight recorder for every claim
  • 🧠 Agent Trace Code viewer (JSON syntax highlighting):
      Full pipeline_trace dict: reasoning, decision, and details
      per node — complete explainability layer for compliance/legal
  • 🗂️ All Agent Decisions table — 5 columns:
      Case ID | Agent | Decision | Timestamp | Reasoning
      Flattens all agents from all matching cases into one table
```

---

## Database Schema

SmartForge uses **MongoDB Atlas** as primary storage with automatic SQLite fallback. `db_upsert()` always writes SQLite first (zero-latency, no network) then syncs to MongoDB Atlas best-effort so the dashboards are always responsive even during Atlas outages.

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
    "owner_name":      "Rajesh Kumar",
    "vehicle_type":    "Car / Sedan / SUV",
    "image_paths":     ["/content/VH001-abc123f_img0.jpg"],
    "incident_date":   "2026-03-29",
    "incident_lat":    13.0827,
    "incident_lon":    80.2707,
    "wants_insurance": true,
    "timestamp":       "2026-03-29T04:15:00Z"
  },

  "final_output": {
    "job_id":                        "VH001--20260329T041500",
    "claim_id":                      "CLM-VH001--20260329T041500",
    "vehicle_id":                    "VH001",
    "vehicle_type":                  "car",
    "vehicle_make_estimate":         "sedan-class",
    "gemini_agent_ran":              true,
    "claim_ruling_code":             "CLM_PENDING",
    "claim_ruling":                  "AI assessment complete (score 70/100). Awaiting auditor verification for final approval.",
    "processing_status":             "claim_submitted",
    "auto_approved":                 false,
    "overall_assessment_score":      70,
    "confirmed_damage_count":        3,
    "inspection_recommendation":     "Repair Required",
    "total_estimated_repair_cost":   "₹21,000–₹48,500",
    "settlement_estimate":           "₹21,000–₹48,500",
    "ruling_timestamp":              "2026-03-29T05:00:00Z",
    "damage_detected":               true,
    "damages": [
      {
        "detection_id":                    "D001",
        "type":                            "Dent",
        "location":                        "Front Bumper",
        "location_type":                   "detected",
        "bounding_box":                    [120, 340, 480, 560],
        "confidence":                      0.812,
        "low_confidence_flag":             false,
        "verification_status":             "gemini_golden_frame_confirmed",
        "relative_deformation_index":      0.021,
        "area_ratio":                      0.00724,
        "severity":                        "High",
        "damage_category":                 "Functional",
        "repair_type":                     "Panel replacement",
        "estimated_repair_cost":           "₹7,000–₹18,000",
        "severity_gemini":                 "Severe",
        "is_verified":                     true,
        "golden_frame_path":               "/content/golden_crops/D001_Dent.jpg",
        "golden_frame_size":               [512, 384],
        "golden_frame_primary_idx":        0,
        "multi_angle_verified":            false,
        "repair_recommendation":           "Panel Replacement",
        "structurally_compromised":        true,
        "gemini_reasoning":                "Deep deformation visible across lower fascia; structural compromise confirmed.",
        "rejected":                        false,
        "rejection_reason":                null,
        "source":                          "cv_model",
        "gemini_location":                 null,
        "gemini_location_source":          "cv_primary",
        "gemini_verified":                 null
      }
    ],
    "financial_estimate": {
      "line_items": [
        {
          "part":             "Front Bumper",
          "action":           "REPLACE",
          "severity":         "Severe",
          "cost_usd":         790.00,
          "cost_inr":         65570,
          "cost_inr_fmt":     "₹65,570",
          "gemini_reasoning": "Deep deformation visible across lower fascia; structural compromise confirmed."
        }
      ],
      "total_repair_usd":      790.00,
      "total_repair_inr":      65570,
      "total_repair_inr_fmt":  "₹65,570",
      "vehicle_value_usd":     15000,
      "total_loss_threshold":  "75%",
      "total_loss_flag":       false,
      "disposition":           "REPAIRABLE",
      "currency_note":         "USD costs; INR display at x83"
    },
    "executive_summary":       "Dear Rajesh Kumar, your vehicle has sustained significant front-end damage ...",
    "forensic_report":         "All 5 integrity checks were completed. Trust score: 85/100. ELA score: 1.25 ...",
    "detailed_estimate_text":  "Line items:\n  | Front Bumper | REPLACE | $790.00 | ₹65,570 |",
    "ai_narrative_summary":    "=== SECTION 1: EXECUTIVE SUMMARY ===\nDear Rajesh Kumar ...",
    "pipeline_trace": {
      "intake_agent":        { "timestamp": "...", "reasoning": "...", "decision": "...", "details": {} },
      "fraud_agent":         { "timestamp": "...", "reasoning": "...", "decision": "...", "details": {} },
      "perception_agent":    { "timestamp": "...", "reasoning": "...", "decision": "...", "details": {} },
      "gemini_agent":        { "timestamp": "...", "reasoning": "...", "decision": "...", "details": {} },
      "false_positive_gate": { "timestamp": "...", "reasoning": "...", "decision": "...", "details": {} },
      "health_monitor":      { "timestamp": "...", "reasoning": "...", "decision": "...", "details": {} },
      "verification_v2":     { "timestamp": "...", "reasoning": "...", "decision": "...", "details": {} },
      "reasoning_agent":     { "timestamp": "...", "reasoning": "...", "decision": "...", "details": {} },
      "decision_agent":      { "timestamp": "...", "reasoning": "...", "decision": "...", "details": {} },
      "report_agent":        { "timestamp": "...", "reasoning": "...", "decision": "...", "details": {} }
    },
    "job_summary": {
      "job_id":          "VH001--20260329T041500",
      "started_at":      "2026-03-29T04:15:00Z",
      "completed_at":    "2026-03-29T05:00:00Z",
      "elapsed_seconds": 287.4,
      "agents_run":      ["intake","perception","gemini_agent","false_positive_gate",
                          "health_monitor","reasoning","decision","report"],
      "retry_count":     0,
      "total_messages":  18,
      "drive_backup":    false,
      "status":          "completed"
    }
  },

  "checkpoint_dump": [
    {
      "step":              -1,
      "node":              "gradio_partial",
      "timestamp":         "2026-03-29T05:00:00Z",
      "retry_count":       0,
      "health_score":      1.0,
      "validation_passed": true,
      "n_detections":      9,
      "n_messages":        18
    }
  ],

  "fraud_report": {
    "trust_score":  85,
    "status":       "VERIFIED",
    "flags":        [],
    "details": {
      "photo_datetime":       "2026-03-29T08:22:11",
      "photo_gps":            { "lat": 13.08271, "lon": 80.27068 },
      "gps_distance_km":      0.04,
      "exif_software":        "None",
      "source_type":          "original_mobile_capture",
      "phash_check":          { "status": "UNIQUE", "phash": "c8d12bbdb496b268", "hamming_distance": null },
      "ai_generation_check":  { "is_ai_generated": false, "ai_probability": 0.062,
                                "ela_score": 1.252, "method": "ela_forensics",
                                "reasoning": "ELA score 1.25 ≤ 5.0 — consistent compression" },
      "screen_detection":     { "is_screen": false, "confidence": 0.0, "signals": [] }
    },
    "next_node":   "perception",
    "checked_at":  "2026-03-29T04:15:10Z",
    "checks_run":  5
  },

  "fraud_hash":    "c8d12bbdb496b268",
  "insurance": {
    "filing_claim":    true,
    "policy_number":   "POL-2024-001",
    "claim_reason":    "Rear-end collision at NH-44",
    "incident_date":   "2026-03-29",
    "notes":           "FIR #2024/KUR/0221",
    "submitted_at":    "2026-03-29T04:30:00Z",
    "preference_saved_at": "2026-03-29T04:20:00Z"
  },
  "agent_trace": {
    "intake_agent": { "timestamp": "...", "reasoning": "...", "decision": "...", "details": {} }
  },
  "chat_history":  [["What is my repair cost?", "Your estimated total repair cost is ₹65,570 ($790 USD)..."]],
  "is_fraud":       false,
  "fraud_attempts": 0,
  "auditor_review": {
    "decision":    "Approve Claim",
    "note":        "Verified against workshop photo and police report",
    "reviewed_at": "2026-03-29T05:00:00Z"
  }
}
```

### Storage Policy

| Data | Stored | Rationale |
|------|--------|-----------|
| `final_output` | ✅ Always | Core claim result |
| `checkpoint_dump` | ✅ Always | Compliance flight recorder |
| `fraud_report` | ✅ Always | Mandatory audit trail |
| `agent_trace` | ✅ Always | Decision transparency / explainability |
| `chat_history` | ✅ Per session | AI assistant multi-turn memory |
| `golden_crops` | ⚡ Selective | Stored only for fraud/critical cases (large storage overhead) |

### Status Pipeline

```
uploaded → pref_saved → analyzed → claim_submitted → fraud_checked → approved / rejected
```

> **Note:** `approved` and `rejected` are set exclusively by the human auditor via the Auditor Dashboard. The AI pipeline never sets `approved` directly — all AI-completed claims land in `claim_submitted` (→ `CLM_PENDING`), `pending_workshop_inspection` (→ `CLM_WORKSHOP`), or `manual_review_required` (→ `CLM_MANUAL`).

### Role-Based Access

```python
# User — sees only their own cases (enforced in handle_* functions)
db_find({"vehicle_id": current_user_id})

# Auditor — sees ALL cases, no filter
db_find({})
```

### MongoDB Indexes

Created automatically on first connection for fast query performance:

```python
_mongo_col.create_index("vehicle_id")
_mongo_col.create_index("status")
_mongo_col.create_index([("created_at", DESCENDING)])
```

---

## Repository Structure

```
smartforge-agentic-ai/
│
├── .env.example                    # API key template (GROQ, GEMINI, MONGO_URI, all thresholds)
├── .gitignore                      # Excludes __pycache__, .env, uploaded images, *.pt models
├── LICENSE
├── README.md                       # This file
├── requirements.txt                # All Python dependencies with pinned versions
├── main.py                         # Entry point — launches both Gradio apps simultaneously
│                                   # user_demo on port 7860, auditor_demo on port 7861
│
├── assets/
│   ├── diagrams/
│   │   ├── system_architecture.png       # Full system architecture overview
│   │   ├── cv_pipeline.png               # Computer Vision & Damage Analysis Pipeline
│   │   ├── fraud_layer.png               # 5-Check Fraud Detection Layer Flow
│   │   ├── decision_flow.png             # Agentic Decision & Reasoning Flow
│   │   └── multi_image_architecture.png  # Multi-Image Map-Reduce Architecture
│   └── demo_screenshots/
│       ├── user_tab1_intake.png
│       ├── user_tab2_insurance_pref.png
│       ├── user_tab3_analysis.png
│       ├── user_tab4_summary.png
│       ├── user_tab5_chat.png
│       ├── aauditor_tab1_cases_with_sidebar_Chatbot.png
│       ├── auditor_tab2_claims.png
│       ├── auditor_tab3_fraud.png
│       ├── auditor_tab4_users.png
│       └── auditor_tab5_logs.png
│
├── data/
│   ├── sample_images/
│   │   ├── bike_clean.jpg                # No-damage 2-wheeler test image
│   │   ├── car_damaged_front.jpg         # Front bumper damage (car)
│   │   └── car_fraud_screenshot.jpg      # Screen-capture fraud test image
│   ├── checkpoint_dump.json             # Sample MemorySaver checkpoint for auditor testing
│   ├── final_output.json                # Sample complete pipeline output
│   ├── fraud_audit_report.json          # Sample fraud audit report (human_audit_node output)
│   └── fraud_hash_db.json               # Local pHash fraud database (seed entries for testing)
│
├── notebooks/
│   ├── README.md                        # Notebook quick-start + Gradio live demo guide
│   │                                    # (cell-by-cell walkthrough, tips, troubleshooting)
│   ├── Vehicle_Damage_Agentic_AI_v36_gradio.ipynb  # Main self-contained Colab notebook
│   └── models/
│       ├── detect-best.pt               # Custom YOLO vehicle part detection weights
│       └── seg-best.pt                  # Custom YOLO damage segmentation weights
│
└── src/
    ├── __init__.py
    │
    ├── config/
    │   ├── __init__.py
    │   └── settings.py              # Loads .env, all threshold constants, REPAIR_DATABASE
    │                                # (12 parts + _default), PART_NAME_MAP, COST_TABLE,
    │                                # ZONE_LANGUAGE_MAP, SEVERITY_TO_ACTION
    │
    ├── db/
    │   ├── __init__.py
    │   └── mongo_client.py          # db_upsert  — hybrid write: SQLite first, then Atlas sync
    │                                # db_get     — fetch one case by case_id
    │                                # db_find    — query with filters + limit
    │                                # db_count   — counts by status for stat cards
    │                                # db_mark_auditor — auditor decision write-back
    │                                # db_backend_info — "MongoDB Atlas" or "SQLite (path)"
    │                                # _sqlite_init — idempotent table creation
    │                                # _JSON_FIELDS — fields serialised as JSON strings in SQLite
    │
    ├── models/
    │   ├── __init__.py
    │   ├── gemini_client.py         # _call_gemini(prompt, image_path, schema_hint)
    │   │                            # Uses google.genai SDK (not deprecated google.generativeai)
    │   │                            # Retry logic: primary model → fallback on 429 rate limit
    │   │                            # Strips markdown fences from JSON response
    │   └── groq_client.py           # _call_groq(prompt) → str or GENAI_SENTINEL
    │                                # _template_report(d) — rule-based fallback
    │                                # generate_groq_narrative(damage_list, vehicle_context,
    │                                #   fraud_report, financial_estimate) → 3-section dict
    │
    ├── cv/
    │   ├── __init__.py
    │   ├── fraud_checks.py          # _haversine_km, _parse_exif_gps, _parse_exif_datetime
    │   │                            # _load_fraud_hash_db, _save_fraud_hash_db
    │   │                            # _compute_phash, _check_phash_against_db
    │   │                            # _check_reverse_image_serpapi (SerpAPI Google Lens)
    │   │                            # _detect_screen_capture (FFT Moiré + colour banding)
    │   │                            # perform_ela_check (Error Level Analysis)
    │   │                            # _check_ai_generation_with_fallback (3-stage)
    │   ├── perception.py            # SAHI AutoDetectionModel + get_sliced_prediction
    │   │                            # SamPredictor.predict (SAM mask generation)
    │   │                            # get_damage_location_unified (part/zone routing)
    │   │                            # compute_severity, estimate_cost, severity_to_score
    │   ├── depth.py                 # MiDaS depth inference (MiDaS_small hub model)
    │   │                            # _analyse_image_conditions (V-channel variance +
    │   │                            #   adaptive conf + skip_downsampling flag)
    │   │                            # Relative deformation index per detection mask
    │   └── fusion.py                # NetworkX DiGraph construction (IMAGE/PART/DETECTION)
    │                                # Part-based Golden Record deduplication
    │                                # Image Recycling Fraud loop detection
    │                                # claims_graph global for post-run queries
    │
    ├── graph/
    │   ├── __init__.py
    │   ├── state.py                 # SmartForgeState TypedDict definition
    │   │                            # make_initial_state(image_path) — builds zero state
    │   ├── workflow.py              # StateGraph builder:
    │   │                            #   add_node for all 16 nodes
    │   │                            #   add_edge for fixed edges
    │   │                            #   add_conditional_edges for fraud_router,
    │   │                            #     health_monitor_router
    │   │                            #   compile(checkpointer=MemorySaver(),
    │   │                            #     interrupt_before=["decision"])
    │   └── nodes/
    │       ├── __init__.py
    │       ├── intake.py            # intake_node — file existence check, cv2.imread,
    │       │                        # upscale < 224px, adaptive downsample (cap 4096px),
    │       │                        # _analyse_image_conditions, state update
    │       ├── fraud.py             # fraud_node — 5-check pipeline, trust score, routing
    │       │                        # human_audit_node — terminal; saves fraud_audit_report.json
    │       │                        # fraud_router — "perception" | "map_images" | "human_audit"
    │       ├── map_reduce.py        # map_images_node — Send API fan-out list
    │       │                        # cv_worker_node — per-image SAHI detection + stamp
    │       │                        # fusion_node — NetworkX DiGraph build + reduce phase
    │       ├── verification_v2.py   # verification_v2_node — Golden Frame Batch 3
    │       │                        # get_high_res_crop, _save_crop_to_tmp
    │       │                        # _call_gemini_with_crop (Deep Look prompt)
    │       │                        # VERIFICATION_SCHEMA constant
    │       │                        # Multi-angle secondary crop logic
    │       ├── perception.py        # perception_node — reads adaptive_sahi_conf + retry_count
    │       │                        # SAHI → SAM → MiDaS → part detection → raw_detections
    │       │                        # Clears image_bgr after use (memory management)
    │       ├── gemini_agent.py      # gemini_agent_node — 3-call batched architecture
    │       │                        # Call A: vehicle type classification
    │       │                        # Call B: batch location + low-conf verify (1 call)
    │       │                        # Call C: full-image missed-damage scan
    │       │                        # Gemini positive override, gemini_enriched location type
    │       ├── false_positive_gate.py  # false_positive_gate_node — 4 gates
    │       │                           # Gate 0: gemini_discovery bypass path
    │       │                           # Gates 1-4: YOLO path (conf/area/depth/veto)
    │       │                           # Gemini positive override for FLAT_SURFACE
    │       ├── health_monitor.py    # health_monitor_node — 2 validation checks:
    │       │                        #   Check 1: area_ratio + deformation bounds
    │       │                        #   Check 2: CV-only confidence variance ≤ 0.08
    │       │                        #   (MAJORITY_LOW_CONF check removed — delegated to Gemini)
    │       │                        # pipeline_stability_flag: Stable/Unstable/CircuitBreaker
    │       │                        # health_monitor_router — "reasoning"→"verification_v2"
    │       │                        #   or "perception_retry"
    │       ├── perception_retry.py  # perception_retry_node — retry_count + 1, delegates
    │       ├── reasoning.py         # reasoning_node — Batch 4 financial engine
    │       │                        # compute_severity, estimate_cost, _get_repair_data
    │       │                        # confirmed vs conservative score
    │       │                        # total_loss_flag, financial_estimate assembly
    │       ├── decision.py          # decision_node — is_fraud_flagged check,
    │       │                        # CLM_PENDING / CLM_WORKSHOP / CLM_MANUAL ruling
    │       │                        # HITL interrupt point (interrupt_before=["decision"])
    │       ├── report.py            # report_node — Batch 3/4 final assembly
    │       │                        # generate_groq_narrative (3-section report)
    │       │                        # CONFIRMED_STATUSES set for correct status matching
    │       │                        # Score normalization (0 confirmed → 100)
    │       │                        # Saves /content/final_output.json
    │       └── human_audit.py       # human_audit_node — terminal fraud node
    │                                # Saves /content/fraud_audit_report.json
    │                                # Returns is_fraud=True, final_output → END
    │
    └── ui/
        ├── __init__.py
        ├── theme.py                 # _get_theme(name) — Gradio theme factory
        │                            # _score_badge(score) — coloured HTML badge
        │                            # _ruling_badge(code, status, ruling) — ruling HTML
        │                            # _fraud_badge(fr) — fraud status HTML
        │                            # _pipeline_timeline(agents_run, retry) — node tiles HTML
        │                            # _status_stepper(status) — progress bar HTML
        │                            # _stat_card(label, value, color) — dashboard card HTML
        │                            # All badges use CSS custom properties for dark/light mode
        ├── helpers.py               # handle_intake — Tab 1 save + navigate
        │                            # handle_insurance_preference — Tab 2 save + navigate
        │                            # handle_tab2_save — combined pref + insurance form save
        │                            # handle_analysis — Tab 3 pipeline run + 3-strike logic
        │                            # handle_summary_load — Tab 4 data load from DB
        │                            # _chat_with_session — Tab 5 Groq chat with DB context
        │                            # _run_pipeline_ui — patches globals, streams graph
        │                            # _build_checkpoint_list, _extract_phash
        ├── user_dashboard.py        # user_demo gr.Blocks — 5 tabs (port 7860)
        │                            # Leaflet map iframe (Nominatim autocomplete + GPS)
        │                            # Native HTML5 date picker + JS sync preprocessing
        │                            # 3-strike fraud enforcement in handle_analysis
        │                            # Deferred .then() wiring for Tab 3 → Tab 4 auto-load
        │                            # gr.ChatInterface for Tab 5
        └── auditor_dashboard.py     # auditor_demo gr.Blocks — 5 tabs (port 7861)
                                     # gr.Sidebar with Groq AI bot + live DB context injection
                                     # aud_tabs.select handler for Tab 1 auto-refresh
                                     # _aud_process_claim guard (fraud-flagged → cannot approve)
                                     # audit_search_cases, audit_load_case_detail
                                     # audit_load_claims, audit_load_fraud_cases
                                     # audit_fraud_detail, audit_mark_decision
                                     # audit_load_users, audit_user_history
                                     # audit_load_logs
```

---

## Installation

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | |
| CUDA GPU | T4 or better | Required for YOLOv8 + SAM + MiDaS |
| Google Colab | — | Recommended — free T4 GPU available |
| MongoDB Atlas | Free M0 tier | Optional — SQLite fallback activates automatically |
| Disk space | ~3 GB | Model weights + all pip packages |

### Clone and Install

```bash
git clone https://github.com/your-username/smartforge-agentic-ai.git
cd smartforge-agentic-ai
pip install -r requirements.txt
```

### Model Files

| File | Source | Purpose |
|---|---|---|
| `seg-best.pt` | ✅ Included — `notebooks/models/seg-best.pt` | Custom YOLO damage segmentation model |
| `detect-best.pt` | ✅ Included — `notebooks/models/detect-best.pt` | Custom YOLO vehicle part detection model |
| `sam_vit_b_01ec64.pth` | ⬇️ Auto-downloaded from Meta at runtime (~375 MB) | Meta SAM ViT-B segmentation |

**`seg-best.pt` and `detect-best.pt`** ship with this repository inside `notebooks/models/`. Copy them to `/content/` before running Cell 1:

```python
# Option A — direct copy (models uploaded via Colab Files panel)
!cp notebooks/models/seg-best.pt    /content/seg-best.pt
!cp notebooks/models/detect-best.pt /content/detect-best.pt

# Option B — from mounted Google Drive
import shutil
shutil.copy("/content/drive/MyDrive/smartforge/models/seg-best.pt",    "/content/seg-best.pt")
shutil.copy("/content/drive/MyDrive/smartforge/models/detect-best.pt", "/content/detect-best.pt")
```

**`sam_vit_b_01ec64.pth`** is not stored in this repo (375 MB). The pipeline downloads it automatically:

```python
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
```

No action needed. Subsequent runs in the same Colab session reuse the cached file.

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
gradio>=4.44.0
pandas
pymongo
dnspython
```

---

## Configuration

All user-editable settings live in **Cell 1** of the notebook — this is the **only cell you need to edit**. All other pipeline cells (0–13) and dashboard cells (G1–G4) run unchanged.

For the modular `src/` deployment, copy `.env.example` to `.env`:

```env
# ── AI API Keys ──────────────────────────────────────────────────────────────
GEMINI_API_KEY=AIza...                # https://aistudio.google.com/app/apikey (free tier)
GROQ_API_KEY=gsk_...                  # https://console.groq.com (free tier)

# ── Database ─────────────────────────────────────────────────────────────────
SMARTFORGE_MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/
# Leave empty → automatic SQLite fallback at /content/smartforge_claims.db

# ── Optional Fraud Extensions ─────────────────────────────────────────────────
SERPAPI_KEY=...          # https://serpapi.com — 100 searches/month free
WINSTON_AI_KEY=...       # https://app.gowinston.ai — 2000 credits/month free

# ── Model Paths ──────────────────────────────────────────────────────────────
DAMAGE_MODEL_PATH=/content/seg-best.pt
PART_MODEL_PATH=/content/detect-best.pt
SAM_CHECKPOINT=/content/sam_vit_b_01ec64.pth
SAM_URL=https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# ── SAHI Inference ───────────────────────────────────────────────────────────
SAHI_CONFIDENCE=0.3          # Base confidence (auto-adjusted: 0.25 dark / 0.45 reflective)
SAHI_SLICE_SIZE=640
SAHI_OVERLAP=0.2

# ── Agentic Thresholds ────────────────────────────────────────────────────────
MAX_RETRIES=2                # HealthMonitor retry limit before circuit break
ESCALATION_THRESHOLD=70      # Health score below this → CLM_WORKSHOP
CONFIDENCE_RECHECK_LIMIT=0.45   # YOLO confidence below this → Gemini re-verification
AUTO_APPROVE_THRESHOLD=85    # Retained for reference; AI never auto-approves
HEALTH_SCORE_MIN=0.6

# ── Fraud Layer ───────────────────────────────────────────────────────────────
FRAUD_TRUST_THRESHOLD=40          # Below → SUSPICIOUS_HIGH_RISK
FRAUD_GPS_MAX_DISTANCE_KM=50.0    # Max GPS drift allowed (km)
BYPASS_FRAUD=True                 # False for production
MAX_FRAUD_RETRIES=3               # 3-strike photo retry limit

# ── Fraud Forensics ───────────────────────────────────────────────────────────
WINSTON_AI_THRESHOLD=0.70         # AI probability above this → flagged
PHASH_HAMMING_THRESHOLD=8         # Hamming distance ≤ 8 → near-duplicate
FRAUD_HASH_DB_PATH=/content/fraud_hash_db.json

# ── Golden Frame (Batch 3) ────────────────────────────────────────────────────
GOLDEN_FRAME_CROP_MARGIN=0.25     # 25% context padding around bbox
GOLDEN_FRAME_MIN_CROP_PX=128      # Minimum crop side length
GOLDEN_FRAME_CONFIDENCE_MIN=0.55  # Gemini confidence below this → reject
GOLDEN_FRAME_CROP_DIR=/content/golden_crops

# ── Financial Intelligence (Batch 4) ─────────────────────────────────────────
VEHICLE_VALUE=15000               # USD default (override per claim)
TOTAL_LOSS_THRESHOLD=0.75         # Repair > 75% of vehicle value → TOTALED
USD_TO_INR=83                     # Display conversion rate

# ── AI Models ─────────────────────────────────────────────────────────────────
GEMINI_MODEL=gemini-2.5-flash
GEMINI_FALLBACK_MODEL=gemini-2.5-flash-lite   # Auto-used on 429 rate limit
GROQ_MODEL=llama-3.3-70b-versatile

# ── Output Paths ──────────────────────────────────────────────────────────────
AUDIT_LOG_PATH=/content/audit_log.json
ESCALATION_PATH=/content/escalation_record.json
CHECKPOINT_DUMP_PATH=/content/checkpoint_dump.json
```

### Key Thresholds Reference

| Variable | Default | Effect |
|----------|---------|--------|
| `FRAUD_TRUST_THRESHOLD` | 40 | Trust scores below this → `SUSPICIOUS_HIGH_RISK` → Human Audit |
| `ESCALATION_THRESHOLD` | 70 | Health scores below → `CLM_WORKSHOP` (workshop inspection required) |
| `TOTAL_LOSS_THRESHOLD` | 0.75 | Repair > 75% vehicle value → TOTALED |
| `MAX_RETRIES` | 2 | HealthMonitor retries before circuit break proceeds to `verification_v2` |
| `MAX_FRAUD_RETRIES` | 3 | Photo re-upload attempts before permanent case closure |
| `SAHI_CONFIDENCE` | 0.3 | Base YOLO confidence (auto-raised to 0.45 for high-gloss, lowered to 0.25 for dark) |
| `GOLDEN_FRAME_CONFIDENCE_MIN` | 0.55 | Gemini forensic confidence gate for Golden Frame verification |
| `PHASH_HAMMING_THRESHOLD` | 8 | Hamming distance ≤ 8 = near-duplicate image match in fraud DB |
| `CONFIDENCE_RECHECK_LIMIT` | 0.45 | YOLO confidence below this → `low_confidence_flag = True` → Gemini Call B |

---

## Quick Start

### Option A — Google Colab with Gradio Live Demo (Recommended)

[![Open In Colab](https://img.shields.io/badge/Open%20Notebook%20%7C%20Gradio%20Live%20Demo-Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/your-username/smartforge-agentic-ai/blob/main/notebooks/Vehicle_Damage_Agentic_AI_v36_gradio.ipynb)

1. Click the badge above or open `notebooks/Vehicle_Damage_Agentic_AI_v36_gradio.ipynb` in Colab
2. Set **Runtime → Change runtime type → T4 GPU**
3. Add Colab Secrets via the **🔑 Key icon** in the left sidebar — toggle **Notebook access ON** for each:

   | Secret Name | Where to get it |
   |---|---|
   | `GEMINI_API_KEY` | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) — free tier |
   | `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) — free tier |
   | `SMARTFORGE_MONGO_URI` | MongoDB Atlas connection string *(optional — SQLite fallback if omitted)* |
   | `SERPAPI_KEY` | [serpapi.com](https://serpapi.com) — optional, 100 searches/month free |
   | `WINSTON_AI_KEY` | [app.gowinston.ai](https://app.gowinston.ai) — optional, 2000 credits/month free |

4. Copy model weights to `/content/`:
   ```python
   !cp notebooks/models/seg-best.pt    /content/seg-best.pt
   !cp notebooks/models/detect-best.pt /content/detect-best.pt
   ```
   > SAM weights (`sam_vit_b_01ec64.pth`, ~375 MB) are downloaded **automatically** during Cell 7 — no action needed.

5. Run all cells top-to-bottom: Cell 0 → Cell 1 → Cell 2 → … → Cell G4
   - Total first-run time: ~5–8 minutes (Cell 2 installs all packages)
   - Subsequent sessions: ~2–3 minutes (packages already cached)

6. **Two public Gradio share links appear in Cell G4** — one for users (port 7860), one for auditors (port 7861). Links are valid for 1 week.

**User flow inside the dashboard:**
```
Tab 1 (Vehicle Intake)
  → Tab 2 (Insurance Preference)
  → Tab 3 (Damage Analysis)
  → Tab 4 (Executive Summary)
  → Tab 5 (AI Assistant)
```

> See [`notebooks/README.md`](notebooks/README.md) for the full cell-by-cell walkthrough, live demo instructions, model setup details, and troubleshooting.

### Option B — Local Launch

```bash
# Set environment variables (or fill .env from .env.example)
export GEMINI_API_KEY=AIza...
export GROQ_API_KEY=gsk_...
export SMARTFORGE_MONGO_URI=mongodb+srv://...   # optional

python main.py
# User Dashboard:    http://localhost:7860  (+ public share link if GRADIO_SHARE=True)
# Auditor Dashboard: http://localhost:7861  (+ public share link if GRADIO_SHARE=True)
```

### Option C — Direct Pipeline (No UI)

```python
from src.graph.workflow import graph
from src.graph.state import make_initial_state
import json

state  = make_initial_state("/path/to/car_image.jpg")
thread = {"configurable": {"thread_id": state["job_id"]}}

# Phase 1 — stream to HITL interrupt before decision node
partial = None
for event in graph.stream(state, thread, stream_mode="values"):
    partial = event
print(f"Detections: {len(partial['raw_detections'])}")
print(f"Health score: {partial['health_score']}")

# Phase 2 — resume from interrupt (auto-proceed for scripted runs)
final = None
for event in graph.stream(None, thread, stream_mode="values"):
    final = event

print(final["final_output"]["claim_ruling_code"])
print(final["final_output"]["executive_summary"])
print(final["final_output"]["financial_estimate"]["total_repair_inr_fmt"])

# Dump MemorySaver checkpoint for audit
checkpoint = graph.get_state(thread)
print(json.dumps(checkpoint.values["pipeline_trace"], indent=2, default=str))
```

---

## Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **Orchestration** | LangGraph (StateGraph + MemorySaver) | DCG pipeline, HITL interrupt, retry loop, parallel fan-out |
| **Object Detection** | YOLOv8 (Ultralytics, custom-trained) | Damage segmentation and vehicle part detection |
| **Small Object** | SAHI (Slicing Aided Hyper Inference) | High-recall tiled detection on large images |
| **Segmentation** | SAM (Segment Anything Model ViT-B) | Precise per-detection binary masks |
| **Depth Estimation** | MiDaS (isl-org MiDaS_small) | 3D reasoning, relative deformation index |
| **Multimodal VLM** | Gemini 2.5 Flash | Vehicle type classification, location enrichment, Golden Frame forensics, missed-damage scan |
| **Text Generation** | Groq Llama-3.3-70b-versatile | Executive summary, forensic report, AI auditor bot, user chat |
| **Fraud — Forensics** | ELA, FFT Moiré, Laplacian variance | AI-image generation and screen-capture detection (3-stage unbreakable) |
| **Fraud — Duplicates** | imagehash (pHash + Hamming) | Cross-claim image recycling detection with local hash DB |
| **Fraud — Metadata** | exifread | EXIF temporal, GPS, and software source checks |
| **Fraud — Reverse Search** | SerpAPI Google Lens (optional) | Internet duplicate image search |
| **Fraud — AI Detection** | Winston AI API (optional) | Cloud-based AI-image probability score (Stage 1) |
| **Graph DB** | NetworkX DiGraph | Multi-image detection fusion, deduplication, and recycling loop detection |
| **Persistence** | MongoDB Atlas + SQLite (auto-fallback) | Case documents, chat history, audit logs; hybrid write (SQLite first) |
| **UI — User** | Gradio Blocks | 5-tab claimant dashboard with Leaflet map (port 7860) |
| **UI — Auditor** | Gradio Blocks + gr.Sidebar | 5-tab admin dashboard with AI auditor bot (port 7861) |
| **Map** | Leaflet 1.9.4 + OpenStreetMap + Nominatim | Interactive incident location picker with search autocomplete and GPS |
| **Env** | Google Colab T4 GPU | Primary compute runtime (free tier available) |

---

## Roadmap

- [ ] **Modular Refactor** — migrate monolithic notebook to the `src/` package structure documented above
- [ ] **FastAPI backend** — REST API: `POST /analyze`, `GET /case/{id}`, `PATCH /case/{id}/audit`, `GET /stats`
- [ ] **WebSocket streaming** — real-time agent progress events pushed to dashboard (replace polling / static timeline)
- [ ] **MongoDB full migration** — remove all SQLite fallback paths; add Atlas Search for full-text claim reason queries
- [ ] **YOLOv9 upgrade** — retrain damage and part detection models on expanded labelled dataset
- [ ] **Multi-language reports** — Groq prompt templates for Tamil, Hindi, Telugu narrative output
- [ ] **Mobile PWA** — camera-first intake for field adjusters with offline-capable Service Worker and photo queue
- [ ] **Webhook notifications** — Slack / email alerts on fraud flag, claim status change, or total-loss determination
- [ ] **Unit test coverage** — complete `tests/` suite covering fraud layer, LangGraph routing, financial engine, DB layer; CI/CD via GitHub Actions
- [ ] **Docker** — `docker-compose.yml` with app container + MongoDB + Nginx reverse proxy
- [ ] **Role-based authentication** — JWT or OAuth2 to enforce user/auditor separation at the infrastructure level rather than application logic

---

## Acknowledgements

- [LangGraph](https://github.com/langchain-ai/langgraph) — Directed Cyclic Graph orchestration
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — Object detection backbone
- [SAHI](https://github.com/obss/sahi) — Small object detection framework
- [Segment Anything (Meta AI)](https://github.com/facebookresearch/segment-anything) — Image segmentation
- [MiDaS (isl-org)](https://github.com/isl-org/MiDaS) — Monocular depth estimation
- [Google Gemini](https://ai.google.dev) — Multimodal VLM for verification and forensics
- [Groq](https://groq.com) — Ultra-fast LLM inference
- [Gradio](https://gradio.app) — Dashboard UI framework
- [MongoDB Atlas](https://mongodb.com/atlas) — Cloud database
- [Leaflet](https://leafletjs.com) + [OpenStreetMap](https://www.openstreetmap.org) — Interactive incident location map
- [Nominatim](https://nominatim.org) — Free geocoding and address search API

---

<div align="center">

**Built with 🔬 by the SmartForge team**

*SmartForge v36 · LangGraph DCG · SAHI + SAM + MiDaS · Gemini 2.5 Flash · Groq Llama-3.3-70b · 5-Check Fraud Layer (threshold 40) · 3-Strike Fraud Retry · Leaflet Incident Map · AI Auditor Sidebar · Golden Frame Verification · NetworkX Graph DB · Human-Auditor-Only Approval*
---

## License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.

Copyright (c) 2026 Dasari Likhith
</div>
