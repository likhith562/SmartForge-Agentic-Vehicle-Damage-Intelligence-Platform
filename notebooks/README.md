# 📓 SmartForge Notebook — Gradio Live Demo

> **Open this notebook in Google Colab to launch both interactive Gradio dashboards with public share links directly from your browser — no local setup required.**

[![Open In Colab](https://img.shields.io/badge/Open%20In-Colab%20%7C%20Gradio%20Live%20Demo-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/your-username/smartforge-agentic-ai/blob/main/notebooks/Vehicle_Damage_Agentic_AI_v36_gradio.ipynb)

---

## What the Notebook Launches

Running all cells produces **two public Gradio share links** (valid for 1 week each):

| App | Port | URL format | Audience |
|-----|------|-----------|----------|
| **User Dashboard** | 7860 | `https://xxxxxxxx.gradio.live` | Vehicle owner / claimant |
| **Auditor Dashboard** | 7861 | `https://yyyyyyyy.gradio.live` | Insurance adjuster / auditor |

Both links appear at the bottom of Cell G4. Share the respective links with users and auditors — no authentication is required in the demo setup.

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Google account | To access Colab |
| T4 GPU runtime | Runtime → Change runtime type → T4 GPU |
| `GEMINI_API_KEY` | Free at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) |
| `GROQ_API_KEY` | Free at [console.groq.com/keys](https://console.groq.com/keys) |
| `SMARTFORGE_MONGO_URI` | Optional — SQLite fallback activates automatically if omitted |

---

## Cell-by-Cell Walkthrough

### Setup Cells

| Cell | Name | What it does | Run once? |
|------|------|-------------|-----------|
| **Cell 0** | Drive Mount | Mounts Google Drive for model/log persistence | Optional |
| **Cell 1** | Configuration | Sets all paths and thresholds — **edit this cell** | Every session |
| **Cell 2** | Install Dependencies | Installs langgraph, ultralytics, sahi, gemini, groq, gradio (~3–5 min) | Once per session |
| **Cell 3** | Imports | Imports all libraries, confirms GPU availability | Every session |

> **Cell 1 is the only cell you need to edit.** All other cells run as-is.

### Pipeline Cells

| Cell | Name | What it does |
|------|------|-------------|
| **Cell 4** | State Schema | Defines `SmartForgeState` TypedDict — the shared state for all agents |
| **Cell 5** | Pipeline Helpers | Cost tables, severity mapping, location helpers |
| **Cell 6** | `intake_node` | Image validation, adaptive SAHI confidence, specular highlight detection |
| **Cell 6b** | `fraud_node` | 5-check fraud layer (EXIF + GPS + pHash + ELA + FFT Moiré) |
| **Cell 6c** | Batch 2 Map-Reduce | `map_images_node`, `cv_worker_node`, `fusion_node` + NetworkX graph |
| **Cell 6d** | Batch 3 Verification | `verification_v2_node` — Golden Frame high-res crop + Gemini Deep Look |
| **Cell 7** | `perception_node` | Full SAHI → SAM → MiDaS → part detection pipeline |
| **Cell 7b** | `gemini_agent_node` | Vehicle type, location enrichment, low-conf verification, missed-damage scan |
| **Cell 7c** | `false_positive_gate_node` | 4-gate domain-shift correction for non-car vehicles |
| **Cell 8** | `health_monitor_node` | Validation checks + conditional retry routing |
| **Cell 9** | `perception_retry_node` | Circuit-breaker retry wrapper |
| **Cell 10** | `reasoning_node` | Severity classification + Batch 4 financial estimation |
| **Cell 11** | `decision_node` | Claim ruling (CLM_PENDING / CLM_WORKSHOP / CLM_MANUAL) + HITL interrupt |
| **Cell 12** | `report_node` | Three-section Groq narrative (executive summary + forensic + line-item estimate) |
| **Cell 13** | Build Graph | Assembles LangGraph DCG with MemorySaver checkpointer |

### Dashboard Cells

| Cell | Name | What it does |
|------|------|-------------|
| **Cell G1** | Dashboard Config | Set theme, MongoDB URI, share toggle |
| **Cell G2** | Database Layer | MongoDB Atlas connection + SQLite fallback; defines `db_upsert`, `db_get`, `db_find`, `db_count` |
| **Cell G3** | User Dashboard | Builds `user_demo` Gradio app (5 tabs) — does NOT launch yet |
| **Cell G4** | Auditor Dashboard + Launch | Builds `auditor_demo` + launches both apps → **two public share links appear here** |

---

## User Dashboard Flow (5 Tabs)

```
Tab 1 — Vehicle Intake
  Enter vehicle ID, owner name, upload damage photos, pick incident
  date and location on the Leaflet map, then press "Save & Proceed".

Tab 2 — Insurance Preference
  Choose "Yes – I want to file a claim" (activates fraud checks) or
  "No – damage assessment only" (bypasses fraud checks).
  Fill in policy number and claim reason if filing.

Tab 3 — Damage Analysis
  Press "Run Full Analysis" to execute the complete LangGraph pipeline.
  Watch the agent timeline update in real time.
  Detection table shows every damage with confirmed/rejected status.

Tab 4 — Executive Summary
  Health score badge, claim ruling (CLM_PENDING / CLM_WORKSHOP / CLM_MANUAL),
  line-item repair estimate in USD and INR, fraud badge, forensic integrity report.

Tab 5 — AI Assistant
  Chat with Groq Llama-3.3-70b scoped to your current session only.
  Ask about damages, costs, fraud results, claim status.
```

---

## Auditor Dashboard Flow (5 Tabs)

```
Tab 1 — Case Explorer      Search all cases, click row for full detail + JSON
Tab 2 — Insurance Claims   Filter by status, approve or reject claims directly
Tab 3 — Fraud Review       Forensic detail per flagged case, auditor decision
Tab 4 — User Management    Per-vehicle history, claim frequency stats
Tab 5 — Audit Logs         MemorySaver checkpoint timeline + agent trace JSON
```

An **AI Auditor sidebar** (Groq-powered, collapsible on the right) is available from every tab — ask it about system stats, recent fraud flags, or specific cases.

---

## Model Files

The two custom YOLO weights ship inside `notebooks/models/`:

```
notebooks/models/
├── seg-best.pt       ← damage segmentation model (YOLO custom)
└── detect-best.pt    ← part detection model (YOLO custom)
```

Copy them to `/content/` before running Cell 1:

```python
# Run this once in a Colab cell before Cell 1
!cp /content/drive/MyDrive/smartforge/seg-best.pt    /content/seg-best.pt
!cp /content/drive/MyDrive/smartforge/detect-best.pt /content/detect-best.pt
# Or if models are uploaded directly to Colab Files panel, they are already at /content/
```

The SAM checkpoint (`sam_vit_b_01ec64.pth`, ~375 MB) is **downloaded automatically** from Meta's servers during Cell 7 on first run — no action required.

---

## Tips for a Smooth Demo

- **BYPASS_FRAUD = True** in Cell 1 skips all fraud checks for quick demos without EXIF metadata. Set to `False` for production.
- If you get a **429 rate limit** from Gemini, the system automatically falls back to `gemini-2.5-flash-lite`.
- MongoDB URI is optional — if omitted, all data is stored in `/content/smartforge_claims.db` (SQLite). The data resets when the Colab session ends.
- Both Gradio share links expire after **1 week**. Re-run Cell G4 to get fresh links.
- Use **Cell 16** (if present) to trigger the deliberate unhappy-path demo — shows the HealthMonitor self-correction loop in action.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `CUDA out of memory` | Restart runtime, re-run from Cell 2 |
| `ModuleNotFoundError: sahi` | Re-run Cell 2 (install cell) |
| `SecretNotFoundError: Gemini_API_Key` | Add secret via 🔑 sidebar, toggle Notebook access ON |
| Gradio link expired | Re-run Cell G4 |
| `seg-best.pt not found` | Copy model to `/content/` as shown above |
| MongoDB connection timeout | Leave `SMARTFORGE_MONGO_URI` empty — SQLite fallback activates |
